import json
import os
import re
import time
from collections import defaultdict
from uuid import uuid4
from src.utils.doc_loader import clean_text, extract_meta_from_text, normalize_title
import chromadb
import torch
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

STRIP_TAGS_RE = re.compile(r"</?system>|</?assistant>|</?user>|<\/?[^>]+>", re.IGNORECASE)
WS_RE = re.compile(r"\s+")
SECTION_RE = re.compile(r'(?:Раздел|Глава)\s+([IVXLC\d]+)[\.\:\)]?\s*(.+)?', re.IGNORECASE)
CLAUSE_RE  = re.compile(r'(?:п\.|пункт)\s*([\d]+(?:\.[\d]+)*)', re.IGNORECASE)
END_PUNCT_RE = re.compile(r"[.!?…»)\]]$")
BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•—]|[\d]{1,2}\.)\s+", re.MULTILINE)


def drop_trailing_incomplete_item(text: str) -> str:
    """Если последний буллет оборван (нет завершающей пунктуации) — удаляем его."""
    lines = text.rstrip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if BULLET_LINE_RE.match(lines[i]):
            if not END_PUNCT_RE.search(lines[i].rstrip()):
                del lines[i]
            break
    return "\n".join(lines).rstrip()
 

# --- мягкая нормализация терминов/опечаток в доменных фразах ---
def normalize_terms(text: str) -> str:
    """Правит частые огрехи в ответах (осторожные замены по контексту)."""
    # 1) Нормализуем «ТУ 152» / «ТУ- 152» → «ТУ-152»
    text = re.sub(r"ТУ\s*-?\s*152", "ТУ-152", text, flags=re.IGNORECASE)

    # 2) «в карте дизеля/двигателя» → «в картере ...»
    text = re.sub(r"\bв\s+карте\s+(дизеля|двигателя)\b", r"в картере \1", text, flags=re.IGNORECASE)

    # 3) «изменения/изменений/изменениях ... в журнал(е) ТУ-152» → «замечания/...»
    #   — только если рядом упоминается журнал ТУ-152 (чтобы не ломать другие случаи)
    def _repl_changes(m: re.Match) -> str:
        ending = m.group(1)  # я | й | ях
        return "замечани" + ending  # даёт: замечания/замечаний/замечаниях
    text = re.sub(
        r"\bизменени(я|й|ях)\b(?=[^.\n]{0,80}журнал\w*\s*(?:формы\s*)?ТУ-?\s*152)",
        _repl_changes,
        text,
        flags=re.IGNORECASE,
    )

    return text


def _title_from_meta(meta: dict) -> str:
    t = (meta.get("doc_title") or meta.get("title") or "Документ")
    if isinstance(t, str) and t.endswith((".pdf", ".docx")):
        t = t.rsplit(".", 1)[0]
    return t


def _fallback_clause_section(text: str) -> tuple[str | None, str | None]:
    head = (text or "")[:1200]
    c = CLAUSE_RE.search(head)
    clause = c.group(1) if c else None
    m = SECTION_RE.search(head)
    section = None
    if m:
        roman_or_num, name = m.groups()
        name = (name or "").strip()
        section = f"Раздел {roman_or_num}" + (f" {name}" if name else "")
    return clause, section


def group_sources_by_doc(docs: list[Document], limit: int = 5) -> list[str]:
    from collections import OrderedDict
    groups = OrderedDict()
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        title = _title_from_meta(meta)
        clause = meta.get("clause")
        section = meta.get("section")
        page = meta.get("page") if isinstance(meta.get("page"), int) else None
        if not clause and not section:
            fc, fs = _fallback_clause_section(d.page_content)
            clause = clause or fc 
            section = section or fs
        g = groups.setdefault(title, {"clauses": set(), "sections": set(), "pages": set()})
        if clause: 
            g["clauses"].add(clause)
        elif section: 
            g["sections"].add(section)
        elif page: 
            g["pages"].add(page)

    lines = []
    for title, g in groups.items():
        parts = []
        if g["clauses"]:
            def key(c): return [int(x) for x in c.split(".") if x.isdigit()]
            parts.append("п. " + "; ".join(sorted(g["clauses"], key=key))[:120])
        elif g["sections"]:
            parts.append("; ".join(sorted(g["sections"]))[:120])
        elif g["pages"]:
            pages = sorted(g["pages"])
            parts.append("стр. " + (", ".join(map(str, pages[:3])) + (" …" if len(pages) > 3 else "")))
        lines.append(title if not parts else f"{title}, {', '.join(parts)}")
        if len(lines) >= limit:
            break
    return lines


def postprocess_text(text: str) -> str:
    if not text:
        return ""
    text = STRIP_TAGS_RE.sub("", text)
    m = re.search(r"(источники\s*:?).*$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = text[:m.start()].rstrip()
    for lab in ("Ответ:", "Пункты:", "Human:", "Assistant:", "Пользователь:", "Ассистент:"):
        text = text.replace(lab, "")
    text = WS_RE.sub(" ", text).strip()
    seen, out = set(), []
    for s in re.split(r"(?<=[\.\?\!])\s+", text):
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return " ".join(out)



def _format_source(meta: dict) -> str:
    """Приоритет: пункт > раздел > страница > только название."""
    title = (meta.get("doc_title") or meta.get("title") or "Документ")
    if isinstance(title, str) and title.endswith((".pdf", ".docx")):
        title = title.rsplit(".", 1)[0]
    clause  = meta.get("clause")
    section = meta.get("section")
    page    = meta.get("page")
    if clause:
        return f"{title}, п. {clause}"
    if section:
        return f"{title}, {section}"
    if isinstance(page, int):
        return f"{title}, стр. {page}"
    return title


class RAGService:
    def __init__(self, model_name: str, chroma_path: str, glossary_json: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 0) Загружаем глоссарий (JSON) один раз
        self.glossary: dict[str, str] = {}
        if os.path.exists(glossary_json):
            with open(glossary_json, "r", encoding="utf-8") as f:
                self.glossary = json.load(f)
            print(f"[RAGService] Загружено {len(self.glossary)} терминов из {glossary_json}")
        else:
            print(f"[RAGService] Файл глоссария {glossary_json} не найден")


        print("==" * 50 , self.device, "===" * 30)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                dtype=torch.float32,
            )
        self.model.eval()

        # 2) Сплиттер для создания чанков
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # 3) ChromaDB + эмбеддинги
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        client = chromadb.PersistentClient(path=chroma_path)
        self.vectorstore = Chroma(
            client=client,
            collection_name="rzd_docs",
            embedding_function=self.embeddings,
        )
        # 4) Cross‑encoder для rerank и фильтрации
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda" if torch.cuda.is_available() else "cpu")


        # 5) Системный промпт
        self.SYSTEM_PROMPT = (
            "Ты — ассистент по нормативным документам и технической эксплуатации железнодорожного транспорта (РЖД).\n"
            "Отвечай строго на основе переданных документов (<documents>) и истории (<history>). Не привлекай внешние знания.\n\n"
            "Формат ответа:\n"
            "• Первая строка: краткий вывод (1–2 предложения, ≤120 слов) \n"
            "• Далее до 5 маркеров с действиями/нормами (кратко и по делу, без лишней типографики).\n"
            "• В конце отдельным блоком добавь 'Источники:' — список вида «<Название>, п. X.Y» или «<Название>, Раздел N», если пункта нет — «стр. N».\n\n"
            "Требования:\n"
            "• Числа, нормы, пункты — только из предоставленных фрагментов.\n"
            "• Если данных недостаточно, прямо укажи, чего не хватает.\n"
            "• Не вставляй ссылки/сноски внутри текста — только финальный список источников."
        )

        

        # 6) История диалогов: user_id -> список сообщений {"role":..., "content":...}
        self.conversation_history: dict[str, list[dict]] = defaultdict(list)
        self.MAX_HISTORY = 2  # будем хранить не более 2*MAX_HISTORY последних сообщений

        # Пути к папкам
        self.docs_path = "docs/"
        self.chroma_path = chroma_path
        torch.set_float32_matmul_precision("high")  # TF32 на Ampere/4060

        try:
            self.model.config.attn_implementation = "flash_attention_2"
            print("[RAG] Using FlashAttention-2")
        except Exception:
            try:
                self.model.config.attn_implementation = "sdpa"
                print("[RAG] Using SDPA attention")
            except Exception:
                print("[RAG] Using eager attention")

    def _keyword_boost(self, query: str, meta: dict) -> float:
        q = (query or "").lower()
        kws = (meta or {}).get("keywords") or ""  # строка "kw1, kw2"
        score = 0.0
        for k in [x.strip() for x in kws.split(",") if x.strip()]:
            if k in q:
                score += 1.0
        return score

    # -------------------------
    # 1) Методы для query extension (глоссарий)
    # -------------------------

    def _extend_query(self, query: str) -> str:
        """
        Ищет все термины из self.glossary в исходном запросе (точное вхождение без учёта регистров).
        Если термин найден, врезает его определение (из JSON) в конец запроса.
        Возвращает «расширённый» запрос.
        """
        lowered_query = query.lower()
        extras = []

        for term, definition in self.glossary.items():
            if term.lower() in lowered_query:
                # Добавляем "термин: определение" как дополнительную часть
                extras.append(f"{term}: {definition}")

        if not extras:
            return query
        return query + " " + " ".join(extras)

    # -------------------------
    # 2) Методы для истории диалога и работа с файлами
    # -------------------------

    def update_history(self, user_id: str, role: str, text: str):
        """
        Добавляем запись {"role": role, "content": text} в историю для данного user_id.
        Если превышается 2*MAX_HISTORY, обрезаем старые записи.
        """
        entry = {"role": role, "content": text}
        self.conversation_history[user_id].append(entry)
        if len(self.conversation_history[user_id]) > self.MAX_HISTORY * 2:
            self.conversation_history[user_id] = self.conversation_history[user_id][-self.MAX_HISTORY * 2 :]

    def _format_history(self, user_id: str) -> str:
        """Простой диалоговый формат, понятный большинству LLM."""
        lines = []
        for m in self.conversation_history[user_id]:
            r = m.get("role", "")
            c = (m.get("content", "") or "").strip()
            if not c:
                continue
            if r == "user":
                lines.append(f"Пользователь: {c}")
            elif r == "assistant":
                lines.append(f"Ассистент: {c}")
        return "\n".join(lines)

    def clear_history(self, user_id: str):
        """Очищает всю историю для указанного user_id."""
        self.conversation_history[user_id] = []

    def add_document(self, file_path: str) -> int:
        """
        Добавляет в ChromaDB один документ (PDF или DOCX).
        Логика та же, что в инициализаторе базы: чистка текста, сплит, метаданные.
        Возвращает число добавленных чанков.
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".pdf", ".docx"):
            raise ValueError("Поддерживаются только PDF и DOCX файлы.")

        fname = os.path.basename(file_path)
        print(f"[add_document] Индексация: {fname}")

        # 1) Загрузка документов (pdf постранично, docx целиком)
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split()
            for p in docs:
                p.metadata["source"] = "pdf"
                p.metadata["title"] = fname
                # PyPDFLoader отдаёт page с нуля — приведём к 1-based если это int
                raw = p.metadata.get("page", None)
                p.metadata["page"] = (raw + 1) if isinstance(raw, int) else None
                p.page_content = clean_text(p.page_content)
        else:  # .docx
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = "docx"
                d.metadata["title"] = fname
                d.metadata["page"] = None      # у docx нет страниц
                d.page_content = clean_text(d.page_content)

        if not docs:
            print("[add_document] Пустой документ после загрузки.")
            return 0

        # 2) Сплит
        raw_chunks = self.text_splitter.split_documents(docs)

        # 3) Обогащение метаданных + фильтр коротких
        base_title = normalize_title(file_path)
        chunks: list[Document] = []
        MIN_CHARS = globals().get("MIN_CHUNK_CHARS", 80)  # возьмём глобальную константу, если есть

        for i, ch in enumerate(raw_chunks):
            ch.page_content = clean_text(ch.page_content)
            if len((ch.page_content or "").strip()) < MIN_CHARS:
                continue

            ch.metadata["chunk_id"] = str(uuid4())
            ch.metadata["chunk_index"] = i
            ch.metadata["chunk_start"] = ch.metadata.get("start_index", None)

            ch.metadata.setdefault("title", fname)
            ch.metadata.setdefault("page", None)
            ch.metadata["doc_title"] = base_title

            extra = extract_meta_from_text(ch.page_content)
            if extra.get("section"):
                ch.metadata["section"] = extra["section"]
            if extra.get("clause"):
                ch.metadata["clause"] = extra["clause"]
            if extra.get("keywords"):
                kws = [k.strip() for k in extra["keywords"] if k and isinstance(k, str)]
                ch.metadata["keywords"] = ", ".join(kws) if kws else None

            chunks.append(ch)

        if not chunks:
            print("[add_document] Нет валидных чанков после фильтра.")
            return 0

        # 4) Запись в текущую коллекцию Chroma
        try:
            self.vectorstore.add_documents(chunks)
        except Exception as e:
            print(f"[add_document] Ошибка при сохранении в Chroma: {e}")
            raise

        print(f"[add_document] Добавлено {len(chunks)} чанков из «{fname}».")
        return len(chunks)

    def get_loaded_documents(self) -> list[str]:
        """
        Возвращает список уникальных названий документов (metadata['title']),
        уже добавленных в ChromaDB (текущую коллекцию).
        """
        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None:
            return []
        try:
            docs = collection.get(include=["metadatas"])
        except Exception:
            return []
        titles: set[str] = set()
        for metas in docs.get("metadatas", []):
            if isinstance(metas, dict) and "title" in metas:
                titles.add(metas["title"])
        return sorted(titles)

    def remove_document(self, title: str) -> int:
        """
        Удаляет из ChromaDB все чанки, у которых metadata['title'] == title.
        Возвращает число удалённых записей (по ответу Chroma).
        """
        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None:
            return 0
        try:
            res = collection.delete(where={"title": title})
            # Chroma может вернуть ids удалённых; если нет — просто 0/1 незначимо
            removed = len(res) if isinstance(res, list) else 0
            print(f"[remove_document] Удалено записей: {removed} для «{title}».")
            return removed
        except Exception as e:
            print(f"[remove_document] Ошибка при удалении «{title}»: {e}")
            raise

    # 3) Retrieval + Filtering + Rerank (с учетом расширённого запроса)
    # -------------------------

    def _rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: int = 1,
        score_threshold: float = 0.0
    ) -> list[Document]:
        """
        Перенумеровываем ранжируемые документы через Cross‑Encoder:
        1) Считаем скоры для каждой пары [query, doc.page_content]
        2) Оставляем только те, чей скор >= score_threshold
        3) Если после фильтрации ничего не осталось, просто возвращаем топ‑top_n по скору
        4) Иначе – сортируем отфильтрованные и возвращаем первые top_n
        """
        if not docs:
            return []

        pairs  = [[query, doc.page_content] for doc in docs]
        scores = self.cross_encoder.predict(pairs)

        # Фильтруем
        filtered = [(doc, sc) for doc, sc in zip(docs, scores) if sc >= score_threshold]

        if not filtered:
            # Ничего не прошло фильтрацию → берём top_n по любым скорам
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:top_n]]

        # Иначе сортируем отфильтрованные по убыванию
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in filtered[:top_n]]

    def _search_docs(
        self,
        query: str,
        k: int = 10,
        top_n: int = 3,
        score_threshold: float = 0.3
    ) -> list[Document]:
        """
        1) Сначала расширяем запрос через _extend_query
        2) Получаем k кандидатов через MMR
        3) Фильтруем + rerank через Cross‑Encoder с порогом score_threshold
        4) Возвращаем top_n
        """
        # 1) Расширяем запрос
        extended_query = self._extend_query(query)

        # 2) Берём k кандидатов из MMR
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 6, "lambda_mult": 0.45}
        )
        initial_docs = retriever.invoke(extended_query)

        # --- лёгкий переранж по keyword boost (чуть «выталкиваем» целевые чанки вверх)
        initial_docs = sorted(
            initial_docs,
            key=lambda d: -self._keyword_boost(extended_query, getattr(d, "metadata", {}) or {})
        )

        # 3) Фильтруем + реранжим
        return self._rerank(extended_query, initial_docs, top_n=top_n, score_threshold=score_threshold)

    # -------------------------
    # 5) Генерация ответа (multi‑turn + динамика токенов)
    # -------------------------

    def generate_answer(
        self,
        user_id: str,
        query: str,
        k: int = 4,
        score_threshold: float = 0.3,
        max_new_tokens: int = None,
        min_new_tokens: int = 50
    ) -> str:
        """
        Multi-turn генерация ответа с измерением времени по этапам.
        """

        t_all0 = time.perf_counter()

        # 1) Сохраняем запрос пользователя
        t0 = time.perf_counter()
        self.update_history(user_id, "user", query)
        print(f"[RAG] update_history: {time.perf_counter()-t0:.3f} s")

        # 2) Получаем список документов
        t0 = time.perf_counter()
        docs = self._search_docs(query, k=k, top_n=k, score_threshold=score_threshold)
        if not docs:
            return ("Недостаточно данных в загруженных документах для точного ответа. "
                    "Уточни запрос или добавь источник.\n\nИсточники: —")
        print(f"[RAG] retrieval+rerank: {time.perf_counter()-t0:.3f} s")

        # 3) Готовим цитаты
        t0 = time.perf_counter()
        sources = group_sources_by_doc(docs, limit=5)
        formatted_docs = [
            {
                "doc_id": i,
                "title": sources[i] if i < len(sources) else (d.metadata.get("title") or "Источник"),
                "content": d.page_content
            }
            for i, d in enumerate(docs)
        ]
        print(f"[RAG] citations+format_docs: {time.perf_counter()-t0:.3f} s")


        # 4) Строим историю
        t0 = time.perf_counter()
        history_str = self._format_history(user_id)
        print(f"[RAG] format_history: {time.perf_counter()-t0:.3f} s")

        # 5) Сборка prompt
        t0 = time.perf_counter()
        sample = [
            {"role": "system",    "content": self.SYSTEM_PROMPT},
            {"role": "history",   "content": history_str},
            {"role": "documents", "content": json.dumps(formatted_docs, ensure_ascii=False)},
            {"role": "user",      "content": query}
        ]
        prompt = ""
        for msg in sample:
            prompt += f"<{msg['role']}>\n{msg['content']}\n"
        prompt += "<assistant>\n"
        print(f"[RAG] build_prompt: {time.perf_counter()-t0:.3f} s")

        # 6) Динамический max_new_tokens
        t0 = time.perf_counter()
        if max_new_tokens is None:
            enc = self.tokenizer(prompt, return_tensors="pt")
            curr_len = enc["input_ids"].shape[1]
            model_max = (
                getattr(self.model.config, "n_positions", None)
                or getattr(self.model.config, "max_position_embeddings", None)
                or getattr(self.tokenizer, "model_max_length", 4096)
            )
            available = model_max - curr_len - 64
            max_new_tokens = max(min(available, 330), min_new_tokens)
        print(f"[RAG] calc_max_tokens: {time.perf_counter()-t0:.3f} s")

        # 7) Генерация
        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        if self.device == "cuda":
            torch.cuda.synchronize()
        # ------------------------------------------------------------------------------------
        import psutil
        tag = "before"
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / 1024 ** 2

        print(f"[MEM][{tag}] RAM used: {ram_mb:.2f} MB")

            # видеопамять (если есть CUDA)
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / 1024 ** 2
            vram_res = torch.cuda.memory_reserved() / 1024 ** 2
            print(f"[MEM][{tag}] VRAM allocated: {vram_alloc:.2f} MB, reserved: {vram_res:.2f} MB")
        # -----------------------------------------------------------------------------------------
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # ------------------------------------------------------------------------------------
        tag = "after"
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / 1024 ** 2

        print(f"[MEM][{tag}] RAM used: {ram_mb:.2f} MB")

            # видеопамять (если есть CUDA)
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / 1024 ** 2
            vram_res = torch.cuda.memory_reserved() / 1024 ** 2
            print(f"[MEM][{tag}] VRAM allocated: {vram_alloc:.2f} MB, reserved: {vram_res:.2f} MB")
        # -----------------------------------------------------------------------------------------
        if self.device == "cuda":
            torch.cuda.synchronize()
        print(f"[RAG] generation: {time.perf_counter()-t0:.3f} s")

        # 8) Декодирование
        t0 = time.perf_counter()
        gen_text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )
        if "</assistant>" in gen_text:
            gen_text = gen_text.split("</assistant>")[0]
        answer_body = gen_text.replace("</s>", "").strip()
        print(f"[RAG] decode: {time.perf_counter()-t0:.3f} s")

        # 9) Сохраняем ответ в историю
        t0 = time.perf_counter()
        self.update_history(user_id, "assistant", answer_body)
        print(f"[RAG] update_history(answer): {time.perf_counter()-t0:.3f} s")

        # 10) Формируем финальный ответ
        t0 = time.perf_counter()
        answer_clean = postprocess_text(answer_body)
        answer_clean = normalize_terms(answer_clean)
        answer_clean = drop_trailing_incomplete_item(answer_clean)

        citation_str = "Источники:\n" + "\n".join(f"- {s}" for s in sources) if sources else "Источники: —"
        final_answer = (answer_clean.strip() + "\n\n" + citation_str).strip()
        print(f"[RAG] build_final_answer: {time.perf_counter()-t0:.3f} s")

        print(f"[RAG] TOTAL: {time.perf_counter()-t_all0:.3f} s")

        return final_answer



# === Singleton ===
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
CHROMA_PATH  = "src/chroma_db/"
GLOSSARY_JSON= "jsons/glossary.json"
rag = RAGService(MODEL_NAME, CHROMA_PATH, GLOSSARY_JSON)

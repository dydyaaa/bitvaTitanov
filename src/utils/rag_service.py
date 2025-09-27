import time
import re
import json
import os
import torch
import chromadb

from collections import defaultdict


from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder

from transformers import AutoModelForCausalLM, AutoTokenizer


STRIP_TAGS_RE = re.compile(r"</?system>|</?assistant>|</?user>|<\/?[^>]+>", re.IGNORECASE)
WS_RE = re.compile(r"\s+")

def postprocess_text(text: str) -> str:
    """Сносим служебные/HTML-теги, лишние пробелы и дубликаты предложений."""
    if not text:
        return text
    text = STRIP_TAGS_RE.sub("", text)
    text = text.replace("Ответ:", "").replace("Источники:", "")
    text = WS_RE.sub(" ", text).strip()
    # дедуп предложений (простой хеш по нижнему регистру)
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


def _consolidate_sources(docs: list[Document], limit: int = 5) -> list[str]:
    """Уникализируем и ограничиваем количество источников (сохраняем порядок релевантности)."""
    ordered, seen = [], set()
    for d in docs:
        s = _format_source(getattr(d, "metadata", {}) or {})
        if s not in seen:
            seen.add(s)
            ordered.append(s)
        if len(ordered) >= limit:
            break
    return ordered


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
            "• Короткий вывод (1–2 предложения, ≤120 слов).\n"
            "• До 5 маркеров с действиями/нормами.\n"
            "• В конце отдельным блоком добавь 'Источники:' — список вида «<Название> , п. X.Y» или «<Название>, Раздел N», если пункта нет — «стр. N».\n\n"
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
    # 2) Методы для истории диалога
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
        k: int = 3,
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
        print(f"[RAG] retrieval+rerank: {time.perf_counter()-t0:.3f} s")

        # 3) Готовим цитаты
        t0 = time.perf_counter()
        sources = _consolidate_sources(docs, limit=5)
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
            max_new_tokens = max(min(available, 256), min_new_tokens)
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
                temperature=0.2,
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

        citation_str = "Источники:\n" + "\n".join(f"- {s}" for s in sources) if sources else "Источники: —"
        final_answer = (answer_clean.strip() + "\n\n" + citation_str).strip()
        print(f"[RAG] build_final_answer: {time.perf_counter()-t0:.3f} s")

        print(f"[RAG] TOTAL: {time.perf_counter()-t_all0:.3f} s")

        return final_answer



# === Singleton ===
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
CHROMA_PATH  = "src/chroma_db/"
GLOSSARY_JSON= "glossary.json"
rag = RAGService(MODEL_NAME, CHROMA_PATH, GLOSSARY_JSON)

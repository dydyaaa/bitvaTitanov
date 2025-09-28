import { Fragment, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import MessageItem from "../../components/MessageItem/MessageItem";
import ChatInput from "../../components/ChatInput/ChatInput";
import styles from "./ChatPage.module.scss";
import Button from "../../components/Button/Button";
import Modal from "../../components/Modal/Modal";
import UserBadge from "../../components/UserBadge/UserBadge"; // 👈 добавили импорт
import api from "../../services/api";
import { shouldInsertDateSeparator } from "../../services/chat";
import Notification from "../../components/Notification/Notification";
import { AnimatePresence, motion } from "framer-motion";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);
  const navigate = useNavigate();
  const [showClear, setShowClear] = useState(false);
  const [notification, setNotification] = useState(null);

  // для статуса "ИИ печатает..."
  const [isTyping, setIsTyping] = useState(false);
  const [typingTime, setTypingTime] = useState(0);

  // loading state
  const [loading, setLoading] = useState(true);
  const [isFirstLoad, setIsFirstLoad] = useState(true);

  const scrollToBottom = (smooth = true) => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({
        behavior: smooth ? "smooth" : "auto",
      });
    }
  };

  // Загрузка истории
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const { data } = await api.get("/chat/history");

        const loaded = data.flatMap((m) => {
          const msgs = [];
          if (m.text) {
            msgs.push({
              id: `${m.id}-user`,
              role: "user",
              text: m.text,
              timestamp: m.created_at,
            });
          }
          if (m.response) {
            msgs.push({
              id: `${m.id}-bot`,
              role: "bot",
              text: m.response,
              timestamp: m.created_at,
            });
          }
          return msgs;
        });

        loaded.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        setMessages(loaded);
      } catch (err) {
        console.error("Ошибка загрузки истории", err);
        setMessages([
          {
            id: Date.now(),
            role: "bot",
            text: "Ошибка соединения с сервером",
            timestamp: new Date().toISOString(),
            variant: "error",
            onReload: fetchHistory,
          },
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  // Скролл сразу вниз без дерганья при первой загрузке
  useLayoutEffect(() => {
    if (!loading && isFirstLoad) {
      scrollToBottom(false);
      setIsFirstLoad(false);
    }
  }, [loading, isFirstLoad]);

  // Автоскролл вниз при новых сообщениях или "ИИ печатает..."
  useEffect(() => {
    if (!isFirstLoad) {
      scrollToBottom(true);
    }
  }, [messages.length, isTyping, isFirstLoad]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now(),
      role: "user",
      text: input,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    setIsTyping(true);
    setTypingTime(0);

    try {
      const { data } = await api.post("/chat/send", { text: input });

      setIsTyping(false);

      const botMessage = {
        id: data.id || Date.now() + 1,
        role: "bot",
        text: data.response,
        timestamp: data.created_at,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error("Ошибка отправки", err);
      setIsTyping(false);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 2,
          role: "bot",
          text: "Ошибка соединения с сервером",
          timestamp: new Date().toISOString(),
          variant: "error",
          onReload: handleSend,
        },
      ]);
    }
  };

  const handleClear = async () => {
    setNotification({
      id: Date.now(),
      message: "Очищаем историю...",
      status: "loading",
    });

    try {
      await api.delete("/chat/history");
      setMessages([]);

      setNotification({
        id: Date.now(),
        message: "История успешно очищена",
        status: "success",
      });
    } catch (err) {
      console.error("Ошибка очистки истории", err);
      setNotification({
        id: Date.now(),
        message: "Ошибка при очистке истории",
        status: "error",
      });
    }
  };

  // обновляем время "ИИ печатает..."
  useEffect(() => {
    if (!isTyping) return;
    const interval = setInterval(() => {
      setTypingTime((prev) => prev + 100);
    }, 100);
    return () => clearInterval(interval);
  }, [isTyping]);

  let lastDate = null;

  return (
    <div className={styles.chatPage}>
      <div className={styles.messages}>
        <div className={styles.header}>
          <UserBadge /> {/* 👈 заменили старый userBadge на компонент */}
          <div className={styles.actionsBar}>
            <Button variant="secondary" onClick={() => navigate('/stats')}>
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 10.105A1.105 1.105 0 0 1 11.105 9h1.79A1.105 1.105 0 0 1 14 10.105v9.79A1.105 1.105 0 0 1 12.895 21h-1.79A1.105 1.105 0 0 1 10 19.895zm7-6A1.105 1.105 0 0 1 18.105 3h1.79A1.105 1.105 0 0 1 21 4.105v15.79A1.105 1.105 0 0 1 19.895 21h-1.79A1.105 1.105 0 0 1 17 19.895zM3 19a2 2 0 1 0 4 0a2 2 0 1 0-4 0"/></svg>
            </Button>
            <Button variant="secondary" onClick={() => navigate('/documents')}>
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"><path d="M14 3v4a1 1 0 0 0 1 1h4"/><path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2m-8-4h6m-6-4h6"/></g></svg>
            </Button>
            <Button variant="secondary" onClick={() => setShowClear(true)}>
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7h16m-10 4v6m4-6v6M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-12M9 7V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v3"/></svg>
              Очистить
            </Button>
          </div>
        </div>

        <div className={styles.list}>
          {loading ? (
            <div className={styles.loader}>
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                <g fill="none" stroke="currentColor" strokeWidth="1.5">
                  <circle cx="12" cy="12" r="10" />
                  <circle cx="12" cy="12" r="6" />
                  <circle cx="12" cy="12" r="2" />
                  <path strokeLinecap="round" d="M6 12h4m4 0h4m-9 5.196l2-3.464m2-3.464l2-3.464m0 10.392l-2-3.464m-2-3.464L9 6.804" opacity="0.5"/>
                </g>
              </svg>
            </div>
          ) : messages.length === 0 ? (
            <AnimatePresence>
              <motion.div
                key="empty-state"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.4 }}
                className={styles.emptyState}
              >
                <img
                  src="public/TrafficDataChat.png"
                  height={240}
                  width={240}
                  alt="Пустой чат"
                  className={styles.emptyImage}
                />
                <p className={styles.emptyText}>Начни новый диалог, чтобы увидеть сообщения здесь</p>
              </motion.div>
            </AnimatePresence>
          ) : (
            <>
              {messages.map((msg) => {
                const needDate = shouldInsertDateSeparator(lastDate, msg.timestamp);
                if (needDate) {
                  lastDate = msg.timestamp;
                }
                return (
                  <Fragment key={msg.id}>
                    {needDate && (
                      <div
                        style={{
                          textAlign: "center",
                          opacity: 0.7,
                          fontSize: 12,
                          margin: "8px 0",
                        }}
                      >
                        {new Date(msg.timestamp).toLocaleDateString()}
                      </div>
                    )}
                    <MessageItem
                      role={msg.role}
                      text={msg.text}
                      timestamp={msg.timestamp}
                      onCopy={(text) => {
                        navigator.clipboard.writeText(text);
                        setNotification({
                          id: Date.now(),
                          message: "Скопировано в буфер!",
                          status: "success",
                        });
                      }}
                      variant={msg.variant}
                      onReload={msg.onReload}
                    />
                  </Fragment>
                );
              })}

              {isTyping && (
                <MessageItem
                  role="bot"
                  text="ИИ печатает..."
                  timestamp={new Date().toISOString()}
                  onCopy={() => {}}
                  isTyping
                  typingTime={typingTime}
                />
              )}
            </>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className={styles.chatInputWrapper}>
        <ChatInput value={input} onChange={setInput} onSend={handleSend} />
      </div>

      <Modal
        open={showClear}
        title="Очистить историю чата?"
        description="Это действие нельзя отменить."
        confirmText="Очистить"
        onCancel={() => setShowClear(false)}
        onConfirm={() => {
          setShowClear(false);
          handleClear();
        }}
      />

      {notification && (
        <Notification
          key={notification.id}
          message={notification.message}
          status={notification.status}
          onClose={() => setNotification(null)}
        />
      )}
    </div>
  );
}
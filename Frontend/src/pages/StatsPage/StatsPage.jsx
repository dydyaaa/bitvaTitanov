// StatsPage.jsx
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../../services/api";
import styles from "./StatsPage.module.scss";
import Button from "../../components/Button/Button";
import Notification from "../../components/Notification/Notification";
import { AnimatePresence, motion } from "framer-motion";
import UserBadge from "../../components/UserBadge/UserBadge";

export default function StatsPage() {
  const [uniqueUsers, setUniqueUsers] = useState(null);
  const [topWords, setTopWords] = useState([]);
  const navigate = useNavigate('');
  const [loading, setLoading] = useState({
    users: false,
    words: false
  });
  const [notification, setNotification] = useState(null);
  const [period, setPeriod] = useState("7d");

  const fetchUniqueUsers = async () => {
    setLoading(prev => ({ ...prev, users: true }));
    try {
      const { data } = await api.post("/stats/users", { 
        period: period === "1d" ? "day" : 
                period === "7d" ? "week" : "month" 
      });
      setUniqueUsers(data);
    } catch (err) {
      console.error("Ошибка загрузки статистики пользователей", err);
      setNotification({
        id: Date.now(),
        message: "Ошибка загрузки статистики пользователей",
        status: "error",
      });
    } finally {
      setLoading(prev => ({ ...prev, users: false }));
    }
  };

  const fetchTopWords = async () => {
    setLoading(prev => ({ ...prev, words: true }));
    try {
      const { data } = await api.get("/stats/top-words");
      setTopWords(data.top_words || []);
    } catch (err) {
      console.error("Ошибка загрузки топ слов", err);
      setNotification({
        id: Date.now(),
        message: "Ошибка загрузки топ слов",
        status: "error",
      });
    } finally {
      setLoading(prev => ({ ...prev, words: false }));
    }
  };

  useEffect(() => {
    fetchUniqueUsers();
    fetchTopWords();
  }, []);

  useEffect(() => {
    if (uniqueUsers !== null) {
      fetchUniqueUsers();
    }
  }, [period]);

  const handleRefresh = () => {
    fetchUniqueUsers();
    fetchTopWords();
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('ru-RU', {
      day: 'numeric',
      month: 'long',
      year: 'numeric'
    });
  };

  const periodButtons = [
    { id: "1d", label: "1д" },
    { id: "7d", label: "7д" },
    { id: "30d", label: "30д" }
  ];

  return (
    <div className={styles.statsPage}>
      <div className={styles.stats}>
        <div className={styles.header}>
          <UserBadge />
          <div className={styles.actionsBar}>
             <Button 
              variant="secondary" 
              onClick={() => navigate('/chat')}
              className={styles.chatButton}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 11v.01M8 11v.01m8-.01v.01M18 4a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3h-5l-5 3v-3H6a3 3 0 0 1-3-3V7a3 3 0 0 1 3-3z"/></svg>
            </Button>
            <Button 
            variant="secondary" 
            onClick={handleRefresh}
            disabled={loading.users || loading.words}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24">
              <g fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M19.933 13.041a8 8 0 1 1-9.925-8.788c3.899-1 7.935 1.007 9.425 4.747"/>
                <path d="M20 4v5h-5"/>
              </g>
            </svg>
            
          </Button>
          </div>
        </div>

        <div className={styles.statsGrid}>
          {/* Блок уникальных пользователей */}
          <motion.div 
            className={styles.statCard}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className={styles.cardHeader}>
              <h3>Уникальные пользователи</h3>
              <div className={styles.periodTabs}>
                {periodButtons.map((button) => (
                  <button
                    key={button.id}
                    className={`${styles.periodTab} ${
                      period === button.id ? styles.periodTabActive : ""
                    }`}
                    onClick={() => setPeriod(button.id)}
                  >
                    {button.label}
                  </button>
                ))}
              </div>
            </div>
            
            <div className={styles.cardContent}>
              {loading.users ? (
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
              ) : uniqueUsers ? (
                <div className={styles.usersStats}>
                  <div className={styles.usersCount}>
                    {uniqueUsers.unique_users}
                  </div>
                 
                </div>
              ) : (
                <div className={styles.error}>Не удалось загрузить данные</div>
              )}
            </div>
          </motion.div>

          {/* Блок топ слов */}
          <motion.div 
            className={styles.statCard}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <div className={styles.cardHeader}>
              <h3>Популярные слова</h3>
              <div className={styles.wordsCount}>
                Топ 10 слов из вопросов
              </div>
            </div>
            
            <div className={styles.cardContent}>
              {loading.words ? (
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
              ) : topWords.length > 0 ? (
                <div className={styles.wordsList}>
                  {topWords.map((item, index) => (
                    <motion.div 
                      key={item.word} 
                      className={styles.wordItem}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.05 }}
                    >
                      <div className={styles.wordRank}>
                        #{index + 1}
                      </div>
                      <div className={styles.wordInfo}>
                        <span className={styles.word}>{item.word}</span>
                        <span className={styles.frequency}>
                          {item.frequency} раз
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className={styles.empty}>
                  Нет данных о популярных словах
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>

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
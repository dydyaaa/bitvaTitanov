import { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { login } from "../../services/auth";
import Button from "../../components/Button/Button";
import TextInput from "../../components/TextInput/TextInput";
import Notification from "../../components/Notification/Notification";
import styles from "./LoginPage.module.scss";

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notification, setNotification] = useState(null);

  // Таймер для автоудаления ошибки
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(""), 3000); // 3 секунды
      return () => clearTimeout(timer); // очистка таймера при размонтировании или смене ошибки
    }
  }, [error]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login({ email, password });
      navigate("/"); // если логин успешный
    } catch (err) {
      if (!err.response) {
        setNotification({
          id: Date.now(),
          message: "Сервер недоступен. Попробуйте позже",
          status: "error",
        });
      } else {
        const message = err.response.data.detail || "Ошибка авторизации";
        setError(message);
        setNotification({
          id: Date.now(),
          message: 'Неверный логин или пароль',
          status: "error",
        });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>Вход</h1>
        <div className={styles.alt}>
          <Link to="/register">Зарегистрироваться</Link>
          <svg
            fill="white"
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
          >
            <path
              fill="currentColor"
              fillRule="evenodd"
              d="M9 17a1 1 0 0 0 1.707.707l5-5a1 1 0 0 0 0-1.414l-5-5A1 1 0 0 0 9 7z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      </div>

      <div className={styles.inputs}>
        <TextInput
          id="email"
          label="Email"
          type="email"
          value={email}
          onChange={setEmail}
          required
        />
        <TextInput
          id="password"
          label="Пароль"
          type="password"
          value={password}
          onChange={setPassword}
          required
        />
      </div>

      <div className={styles.actions}>
        <Button variant="primary" disabled={loading}>
          {loading ? "Входим..." : "Войти"}
        </Button>
      </div>

      {notification && (
        <Notification
          key={notification.id}
          message={notification.message}
          status={notification.status}
          onClose={() => setNotification(null)}
        />
      )}
    </form>
  );
}

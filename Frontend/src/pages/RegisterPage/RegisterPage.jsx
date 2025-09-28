import { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { register } from "../../services/auth";
import Button from "../../components/Button/Button";
import TextInput from "../../components/TextInput/TextInput";
import Notification from "../../components/Notification/Notification"; // 👈 импорт уведомлений
import styles from "./RegisterPage.module.scss";

export default function RegisterPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notification, setNotification] = useState(null);

  // Таймер для автоудаления уведомления
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (password !== confirm) {
      setError("Пароли не совпадают");
      setNotification({
        id: Date.now(),
        message: "Пароли не совпадают",
        status: "error",
      });
      return;
    }

    setLoading(true);
    try {
      await register({ email, password });
      setNotification({
        id: Date.now(),
        message: "Регистрация прошла успешно!",
        status: "success",
      });
      navigate("/chat");
    } catch (err) {
      const detail = err.response?.data?.detail;
      const message = Array.isArray(detail)
        ? detail.map((d) => d.msg).join(", ")
        : detail || "Ошибка регистрации. Попробуйте снова.";
      setError(message);
      setNotification({
        id: Date.now(),
        message,
        status: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>Регистрация</h1>
        <div className={styles.alt}>
          <Link to="/login">Войти</Link>
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
        <TextInput id="email" label="Email" type="email" value={email} onChange={setEmail} required />
        <TextInput id="password" label="Пароль" type="password" value={password} onChange={setPassword} required />
        <TextInput id="confirm" label="Повторите пароль" type="password" value={confirm} onChange={setConfirm} required />
      </div>


      <div className={styles.actions}>
        <Button variant="primary" disabled={loading}>
          {loading ? "Регистрируем..." : "Зарегистрироваться"}
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

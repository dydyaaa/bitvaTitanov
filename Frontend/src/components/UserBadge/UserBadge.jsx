// UserBadge.jsx
import { useState } from "react";
import { getCurrentUser, logout } from "../../services/auth";
import Modal from "../Modal/Modal";
import styles from "./UserBadge.module.scss";

export default function UserBadge() {
  const user = getCurrentUser();
  const [showLogout, setShowLogout] = useState(false);

  if (!user) return null;

  const handleLogout = () => {
    setShowLogout(false);
    logout();
    window.location.href = "/login";
  };

  return (
    <>
      <button
        className={styles.userBadge}
        onClick={() => setShowLogout(true)}
      >
        <div className={styles.avatar}>
          {(user.email || "").slice(0, 2).toUpperCase()}
        </div>
        <div className={styles.name}>{user.email}</div>
      </button>

      <Modal
        open={showLogout}
        title="Выйти из аккаунта?"
        description="Вы будете перенаправлены на экран входа."
        confirmText="Выйти"
        onCancel={() => setShowLogout(false)}
        onConfirm={handleLogout}
      />
    </>
  );
}
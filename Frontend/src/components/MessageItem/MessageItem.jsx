import styles from "./MessageItem.module.scss";
import Button from "../Button/Button";

export default function MessageItem({
  role,
  text,
  timestamp,
  onCopy,
  isTyping,
  typingTime,
  variant, // 👈 добавляем тип сообщения
  onReload, // 👈 хэндлер для ошибок
}) {
  const date = new Date(timestamp);
  const formattedTime = date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div
      className={`${styles.message || ""} ${styles[role]} ${
        variant === "error" ? styles.error : ""
      }`}
    >
      <div className={styles.content}>
        <p>
          {isTyping ? (
            <span className={styles.typingWrapper}>
              Печатает {(typingTime / 1000).toFixed(1)}s
              <span className={styles.typingDots}>
                <span> .</span>
                <span>.</span>
                <span>.</span>
              </span>
            </span>
          ) : (
            text
          )}
        </p>

        {!isTyping && variant !== "error" && (
          <div className={styles.footer}>
            <span className={styles.time}>{formattedTime}</span>
            <Button
              onClick={() => onCopy?.(text)}
              variant="ghost"
              className={styles.copyBtn}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 24 24"
              >
                <path
                  fill="currentColor"
                  d="M19.53 8L14 2.47a.75.75 0 0 0-.53-.22H11A2.75 2.75 0 0 0 8.25 5v1.25H7A2.75 2.75 0 0 0 4.25 9v10A2.75 2.75 0 0 0 7 21.75h7A2.75 2.75 0 0 0 16.75 19v-1.25H17A2.75 2.75 0 0 0 19.75 15V8.5a.75.75 0 0 0-.22-.5m-5.28-3.19l2.94 2.94h-2.94Zm1 14.19A1.25 1.25 0 0 1 14 20.25H7A1.25 1.25 0 0 1 5.75 19V9A1.25 1.25 0 0 1 7 7.75h1.25V15A2.75 2.75 0 0 0 11 17.75h4.25ZM17 16.25h-6A1.25 1.25 0 0 1 9.75 15V5A1.25 1.25 0 0 1 11 3.75h1.75V8.5a.76.76 0 0 0 .75.75h4.75V15A1.25 1.25 0 0 1 17 16.25"
                />
              </svg>
            </Button>
          </div>
        )}

        {variant === "error" && (
          <div className={styles.errorFooter}>
            <Button variant="secondary" onClick={onReload}>
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"><path d="M19.933 13.041a8 8 0 1 1-9.925-8.788c3.899-1 7.935 1.007 9.425 4.747"/><path d="M20 4v5h-5"/></g></svg>
              Перезагрузить
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

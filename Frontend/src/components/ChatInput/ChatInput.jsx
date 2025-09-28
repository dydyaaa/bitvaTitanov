import { useRef, useEffect } from "react";
import styles from "./ChatInput.module.scss";
import Button from "../Button/Button";

export default function ChatInput({ value, onChange, onSend }) {
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "24px"; // минимальная высота
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 10 * 24) + "px"; // максимум 10 строк
    }
  }, [value]);

  const handleSend = () => {
    if (!value.trim()) return;
    onSend();
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={styles.wrapper}>
      <textarea
        ref={textareaRef}
        className={styles.textarea}
        placeholder="Спросите что-нибудь..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={1}
      />
      <div className={styles.sendButtonWrapper}>
        <Button variant="secondary" onClick={handleSend}>
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
        </Button>
      </div>
    </div>
  );
}

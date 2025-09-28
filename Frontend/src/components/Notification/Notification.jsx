import { useEffect, useState } from "react";
import styles from "./Notification.module.scss";

export default function Notification({ message, status = "success", onClose, duration = 2000 }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!message) return;
    setVisible(true);

    // при "loading" не закрываем автоматически
    if (status === "loading") return;

    const timer = setTimeout(() => {
      setVisible(false);
      setTimeout(() => onClose(), 300); // ждём завершения анимации
    }, duration);

    return () => clearTimeout(timer);
  }, [message, duration, onClose, status]);

  if (!message) return null;

  let icon;
  if (status === "success") {
    icon = (
      <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 1024 1024">
        <path
          fill="#6DD14B"
          d="M512 64a448 448 0 1 1 0 896a448 448 0 0 1 0-896m-55.808 
             536.384l-99.52-99.584a38.4 38.4 0 1 0-54.336 
             54.336l126.72 126.72a38.27 38.27 0 0 0 54.336 
             0l262.4-262.464a38.4 38.4 0 1 0-54.272-54.336z"
        />
      </svg>
    );
  } else if (status === "error") {
    icon = (
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="#FF4D4F" d="M12 17q.425 0 .713-.288T13 16t-.288-.712T12 15t-.712.288T11 16t.288.713T12 17m0-4q.425 0 .713-.288T13 12V8q0-.425-.288-.712T12 7t-.712.288T11 8v4q0 .425.288.713T12 13m0 9q-2.075 0-3.9-.788t-3.175-2.137T2.788 15.9T2 12t.788-3.9t2.137-3.175T8.1 2.788T12 2t3.9.788t3.175 2.137T21.213 8.1T22 12t-.788 3.9t-2.137 3.175t-3.175 2.138T12 22"/></svg>
    );
  } else if (status === "loading") {
    icon = (
      <svg
        className={styles.spinner}
        xmlns="http://www.w3.org/2000/svg"
        width="28"
        height="28"
        viewBox="0 0 24 24"
      >
       <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><defs><linearGradient id="SVGz0tT9cEa" x1="50%" x2="50%" y1="5.271%" y2="91.793%"><stop offset="0%" stop-color="currentColor"/><stop offset="100%" stop-color="currentColor" stop-opacity="0.55"/></linearGradient><linearGradient id="SVGadeRXbLy" x1="50%" x2="50%" y1="15.24%" y2="87.15%"><stop offset="0%" stop-color="currentColor" stop-opacity="0"/><stop offset="100%" stop-color="currentColor" stop-opacity="0.55"/></linearGradient></defs><g fill="none"><path d="m12.593 23.258l-.011.002l-.071.035l-.02.004l-.014-.004l-.071-.035q-.016-.005-.024.005l-.004.01l-.017.428l.005.02l.01.013l.104.074l.015.004l.012-.004l.104-.074l.012-.016l.004-.017l-.017-.427q-.004-.016-.017-.018m.265-.113l-.013.002l-.185.093l-.01.01l-.003.011l.018.43l.005.012l.008.007l.201.093q.019.005.029-.008l.004-.014l-.034-.614q-.005-.018-.02-.022m-.715.002a.02.02 0 0 0-.027.006l-.006.014l-.034.614q.001.018.017.024l.015-.002l.201-.093l.01-.008l.004-.011l.017-.43l-.003-.012l-.01-.01z"/><path fill="url(#SVGz0tT9cEa)" d="M8.749.021a1.5 1.5 0 0 1 .497 2.958A7.5 7.5 0 0 0 3 10.375a7.5 7.5 0 0 0 7.5 7.5v3c-5.799 0-10.5-4.7-10.5-10.5C0 5.23 3.726.865 8.749.021" transform="translate(1.5 1.625)"/><path fill="url(#SVGadeRXbLy)" d="M15.392 2.673a1.5 1.5 0 0 1 2.119-.115A10.48 10.48 0 0 1 21 10.375c0 5.8-4.701 10.5-10.5 10.5v-3a7.5 7.5 0 0 0 5.007-13.084a1.5 1.5 0 0 1-.115-2.118" transform="translate(1.5 1.625)"/></g></svg>
      </svg>
    );
  }

  return (
    <div className={`${styles.notification} ${visible ? styles.show : styles.hide}`}>
      {icon}
      <span className={styles.text}>{message}</span>
    </div>
  );
}

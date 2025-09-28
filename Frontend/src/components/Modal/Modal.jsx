import React, { useEffect } from "react";
import styles from "./Modal.module.scss";
import Button from "../Button/Button";

export default function Modal({ open, title, description, onConfirm, onCancel, confirmText = "Подтвердить", cancelText = "Отмена" }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "Escape") onCancel?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div className={styles.backdrop} onClick={onCancel}>
      <div className={styles.modal} role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        {title && <h3 className={styles.title}>{title}</h3>}
        {description && <div className={styles.desc}>{description}</div>}
        <div className={styles.actions}>
          <Button variant="secondary" onClick={onCancel}>{cancelText}</Button>
          <Button variant="primary" onClick={onConfirm}>{confirmText}</Button>
        </div>
      </div>
    </div>
  );
}



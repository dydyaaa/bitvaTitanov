import React from "react";
import styles from "./Button.module.scss";

const Button = ({ children, onClick, disabled, variant = "secondary" }) => {
  return (
    <button
      className={`${styles.button} ${styles[variant]}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
};

export default Button;

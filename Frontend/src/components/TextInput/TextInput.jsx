import styles from "./TextInput.module.scss";

export default function TextInput({
  id,
  label,
  type = "text",
  value,
  onChange,
  required = false,
  placeholder = "",
}) {
  return (
    <div className={styles.field}>
      {label && <label htmlFor={id}>{label}</label>}
      <input
        id={id}
        className={styles.input}
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
        placeholder={placeholder}
      />
    </div>
  );
}

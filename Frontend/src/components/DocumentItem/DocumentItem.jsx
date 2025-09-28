// DocumentItem.jsx
import { motion } from "framer-motion";
import styles from "./DocumentItem.module.scss";
import Button from "../Button/Button";

export default function DocumentItem({ id, name, size, type, date, onDelete }) {
  const getFileExtension = (fileName, fileType) => {
    // Пытаемся получить расширение из имени файла
    const extension = fileName?.split('.').pop()?.toLowerCase() || '';
    
    // Если есть расширение и оно короткое (макс 4 символа), используем его
    if (extension && extension.length <= 4) {
      return extension;
    }
    
    // Иначе определяем по MIME-type
    if (fileType.includes('pdf')) return 'pdf';
    if (fileType.includes('word') || fileType.includes('document')) return 'doc';
    if (fileType.includes('sheet') || fileType.includes('excel')) return 'xls';
    if (fileType.includes('powerpoint') || fileType.includes('presentation')) return 'ppt';
    if (fileType.includes('image')) return 'img';
    if (fileType.includes('text')) return 'txt';
    if (fileType.includes('zip') || fileType.includes('archive')) return 'zip';
    if (fileType.includes('json')) return 'json';
    if (fileType.includes('xml')) return 'xml';
    if (fileType.includes('csv')) return 'csv';
    
    return 'file';
  };

  const getFileColor = (extension) => {
    const colors = {
      pdf: '#ff4444',
      doc: '#2b579a',
      docx: '#2b579a',
      xls: '#217346',
      xlsx: '#217346',
      ppt: '#d24726',
      pptx: '#d24726',
      txt: '#666666',
      zip: '#9059ff',
      rar: '#9059ff',
      json: '#ff8c00',
      xml: '#ff8c00',
      csv: '#00a300',
      img: '#9c27b0',
      jpg: '#9c27b0',
      png: '#9c27b0',
      gif: '#9c27b0',
      file: '#666666'
    };
    
    return colors[extension] || colors.file;
  };

  const fileExtension = getFileExtension(name, type);
  const fileColor = getFileColor(fileExtension);

  return (
    <motion.div
      className={styles.documentItem}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3 }}
    >
      <div className={styles.header}>
        <div 
          className={styles.fileAvatar}
          style={{ 
            backgroundColor: fileColor,
            '--file-color': fileColor 
          }}
        >
          <span className={styles.fileExtension}>
            {fileExtension}
          </span>
        </div>
        <div className={styles.fileInfo}>
          <h4 className={styles.name} title={name}>{name}</h4>
          <div className={styles.meta}>
            <span className={styles.date}>{date}</span>
            <span className={styles.size}>{size}</span>
          </div>
        </div>
      </div>
      
      <div className={styles.footer}>
        <Button
          variant="secondary"
          onClick={() => onDelete(id, name)}
          className={styles.deleteBtn}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">
            <path fill="currentColor" d="M7 21q-.825 0-1.412-.587T5 19V6H4V4h5V3h6v1h5v2h-1v13q0 .825-.587 1.413T17 21zM17 6H7v13h10zM9 17h2V8H9zm4 0h2V8h-2zM7 6v13z"/>
          </svg>
          Удалить
        </Button>
      </div>
    </motion.div>
  );
}
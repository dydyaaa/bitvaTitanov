import { useEffect, useState, useRef } from "react";
import api from "../../services/api";
import styles from "./DocumentsPage.module.scss";
import Button from "../../components/Button/Button";
import DocumentItem from "../../components/DocumentItem/DocumentItem";
import Notification from "../../components/Notification/Notification";
import { AnimatePresence, motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Modal from "../../components/Modal/Modal";
import UserBadge from "../../components/UserBadge/UserBadge";

export default function DocumentsPage() {
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [notification, setNotification] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [fileToDelete, setFileToDelete] = useState(null);
  const fileInputRef = useRef(null);

  const fetchFiles = async () => {
    setLoading(true);
    try {
      const { data } = await api.get("/documents/list");
      setFiles(Array.isArray(data?.documents) ? data.documents : []);
    } catch (err) {
      console.error("Ошибка загрузки файлов", err);
      setFiles([]);
      setNotification({
        id: Date.now(),
        message: "Ошибка загрузки документов",
        status: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Проверка размера файла (например, 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setNotification({
        id: Date.now(),
        message: "Файл слишком большой (макс. 10MB)",
        status: "error",
      });
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    try {
      await api.post("/documents/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      
      setNotification({
        id: Date.now(),
        message: "Файл успешно загружен",
        status: "success",
      });
      
      fetchFiles();
    } catch (err) {
      console.error("Ошибка загрузки файла", err);
      setNotification({
        id: Date.now(),
        message: "Ошибка загрузки файла",
        status: "error",
      });
    } finally {
      setUploading(false);
      fileInputRef.current.value = null;
    }
  };

  const handleDeleteClick = (id, name) => {
    setFileToDelete({ id, name });
    setShowDeleteModal(true);
  };

  const handleDeleteConfirm = async () => {
    if (!fileToDelete) return;
    
    try {
      await api.delete(`/documents/${fileToDelete.id}`);
      setFiles((prev) => prev.filter((f) => f.id !== fileToDelete.id));
      
      setNotification({
        id: Date.now(),
        message: "Файл удален",
        status: "success",
      });
    } catch (err) {
      console.error("Ошибка удаления файла", err);
      setNotification({
        id: Date.now(),
        message: "Ошибка удаления файла",
        status: "error",
      });
    } finally {
      setShowDeleteModal(false);
      setFileToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setShowDeleteModal(false);
    setFileToDelete(null);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className={styles.documentsPage}>
      <div className={styles.documents}>
        <div className={styles.header}>
            <UserBadge />
          <div className={styles.actionsBar}>
            <Button 
              variant="secondary" 
              onClick={() => navigate('/chat')}
              className={styles.chatButton}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 11v.01M8 11v.01m8-.01v.01M18 4a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3h-5l-5 3v-3H6a3 3 0 0 1-3-3V7a3 3 0 0 1 3-3z"/></svg>
            </Button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleUpload}
              style={{ display: "none" }}
            />
            <Button 
              onClick={() => fileInputRef.current.click()} 
              disabled={uploading}
              variant="secondary"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"><path d="M14 3v4a1 1 0 0 0 1 1h4"/><path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2m-5-10v6"/><path d="M9.5 13.5L12 11l2.5 2.5"/></g></svg>
              {uploading ? "Загружаем..." : "Загрузить документ"}
            </Button>
          </div>
        </div>

        <div className={styles.content}>
          {loading ? (
            <div className={styles.loader}>
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                <g fill="none" stroke="currentColor" strokeWidth="1.5">
                  <circle cx="12" cy="12" r="10" />
                  <circle cx="12" cy="12" r="6" />
                  <circle cx="12" cy="12" r="2" />
                  <path strokeLinecap="round" d="M6 12h4m4 0h4m-9 5.196l2-3.464m2-3.464l2-3.464m0 10.392l-2-3.464m-2-3.464L9 6.804" opacity="0.5"/>
                </g>
              </svg>
            </div>
          ) : files.length === 0 ? (
            <AnimatePresence>
              <motion.div
                key="empty-state"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.4 }}
                className={styles.emptyState}
              >
                <img
                  src="public/Files.png"
                  height={240}
                  width={240}
                  alt="Нет документов"
                  className={styles.emptyImage}
                />
                <p className={styles.emptyText}>Загрузите первый документ</p>
              </motion.div>
            </AnimatePresence>
          ) : (
            <div className={styles.documentsGrid}>
              {files.map((file) => (
                <DocumentItem
                  key={file.id}
                  id={file.id}
                  name={file.original_name}
                  size={formatFileSize(file.file_size)}
                  type={file.content_type}
                  date={new Date(file.created_at).toLocaleDateString('ru-RU')}
                  onDelete={handleDeleteClick}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Модалка для подтверждения удаления */}
      <Modal
        open={showDeleteModal}
        title="Удалить документ?"
        description={fileToDelete ? `Вы уверены, что хотите удалить файл "${fileToDelete.name}"? Это действие нельзя отменить.` : ""}
        confirmText="Удалить"
        cancelText="Отмена"
        onCancel={handleDeleteCancel}
        onConfirm={handleDeleteConfirm}
      />

      {notification && (
        <Notification
          key={notification.id}
          message={notification.message}
          status={notification.status}
          onClose={() => setNotification(null)}
        />
      )}
    </div>
  );
}
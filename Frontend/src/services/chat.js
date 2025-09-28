const STORAGE_KEYS = {
  chats: "app_chats",
};

function read(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function write(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

export function getMessagesForUser(email) {
  const all = read(STORAGE_KEYS.chats, {});
  return all[email] || [];
}

export function saveMessagesForUser(email, messages) {
  const all = read(STORAGE_KEYS.chats, {});
  all[email] = messages;
  write(STORAGE_KEYS.chats, all);
}

export function clearMessagesForUser(email) {
  const all = read(STORAGE_KEYS.chats, {});
  all[email] = [];
  write(STORAGE_KEYS.chats, all);
}

export function shouldInsertDateSeparator(prevISO, nextISO) {
  if (!prevISO) return true;
  const prev = new Date(prevISO);
  const next = new Date(nextISO);
  return prev.toDateString() !== next.toDateString();
}



import api from "./api";

const TOKEN_KEY = "token";
const USER_KEY = "user";

export function getCurrentUser() {
  return JSON.parse(localStorage.getItem(USER_KEY) || "null");
}

export function isAuthenticated() {
  return !!localStorage.getItem(TOKEN_KEY);
}

export function logout() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

// регистрация → затем автологин
export async function register({ email, password }) {
  await api.post("/users/register", {
    username: email,
    password,
  });

  // сразу авторизация
  return await login({ email, password });
}

export async function login({ email, password }) {
  const res = await api.post("/users/login", {
    username: email,
    password,
  });

  // тут сервер вернёт объект { access_token: "..." }
  const token = res.data.access_token;

  localStorage.setItem("token", `Bearer ${token}`); // сохраняем уже в правильном формате
  localStorage.setItem("user", JSON.stringify({ email }));

  return { email };
}


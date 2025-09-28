# Простой Dockerfile для фронтенда
FROM node:22

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем package.json и package-lock.json
COPY package*.json ./

# Устанавливаем все зависимости (включая dev)
RUN npm install

# Копируем исходный код
COPY . .

# Открываем порт
EXPOSE 5173

# Запускаем dev сервер Vite
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

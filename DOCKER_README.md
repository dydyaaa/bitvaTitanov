# Docker Setup для Chat API

## Требования
- Docker
- Docker Compose
- NVIDIA Docker (для CUDA поддержки в PyTorch)

## Быстрый запуск

### 1. Клонирование и настройка
```bash
git clone <repository>
cd bitvaTitanov
```

### 2. Запуск всех сервисов

#### Вариант A: С GPU поддержкой (CUDA)
```bash
docker-compose up --build
```

#### Вариант B: Только CPU (если проблемы с NVIDIA CUDA образом)
```bash
docker-compose -f docker-compose.cpu.yaml up --build
```

### 3. Доступ к сервисам
- **Фронтенд**: http://localhost:3000
- **Бекенд API**: http://localhost:8000
- **API Документация**: http://localhost:8000/docs
- **База данных**: localhost:5432

## Структура сервисов

### Backend (FastAPI)
- **Порт**: 8000
- **Образ**: Собранный из `backend/Dockerfile`
- **Особенности**: 
  - CUDA поддержка для PyTorch (Ubuntu 24.04 + Python 3.12)
  - Автоматическая установка PyTorch 2.6.0+cu124
  - Горячая перезагрузка в development режиме

### Frontend (React + Vite)
- **Порт**: 3000 (внутри контейнера 5173)
- **Образ**: Собранный из `Frontend/Dockerfile`
- **Особенности**:
  - Vite dev сервер для горячей перезагрузки
  - Простая настройка без nginx
  - Автоматическая перезагрузка при изменениях

### Database (PostgreSQL)
- **Порт**: 5432
- **Образ**: postgres:15
- **Данные**: Персистентное хранение в volume

## Полезные команды

### Запуск в фоновом режиме
```bash
docker-compose up -d
```

### Просмотр логов
```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f db
```

### Остановка сервисов
```bash
docker-compose down
```

### Пересборка без кэша
```bash
docker-compose build --no-cache
```

### Выполнение команд в контейнере
```bash
# Бекенд
docker-compose exec backend bash

# База данных
docker-compose exec db psql -U postgres -d chatdb
```

## Разработка

### Горячая перезагрузка
- **Бекенд**: Исходный код монтируется как volume, изменения применяются автоматически
- **Фронтенд**: Vite dev сервер обеспечивает мгновенную перезагрузку при изменениях

### Миграции базы данных
```bash
# Войти в контейнер бекенда
docker-compose exec backend bash

# Выполнить миграции
alembic upgrade head
```

## Troubleshooting

### Проблемы с CUDA
Если возникают проблемы с CUDA образом (`nvidia/cuda:12.4-devel-ubuntu22.04: not found`):

#### Решение 1: Используйте CPU версию
```bash
docker-compose -f docker-compose.cpu.yaml up --build
```

#### Решение 2: Обновите образ CUDA
Измените в `backend/Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04  # вместо 12.4
```

#### Решение 3: Проверьте NVIDIA Docker
Убедитесь что:
1. Установлен NVIDIA Docker
2. Драйверы NVIDIA актуальны
3. Docker имеет доступ к GPU

### Проблемы с портами
Если порты заняты, измените их в `docker-compose.yaml`:
```yaml
ports:
  - "8001:8000"  # Бекенд на 8001
  - "3001:5173"  # Фронтенд на 3001
```

### Очистка данных
```bash
# Остановить и удалить все контейнеры и volumes
docker-compose down -v

# Удалить все образы проекта
docker-compose down --rmi all
```

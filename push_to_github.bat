@echo off
chcp 65001 >nul
set REPO=https://github.com/chookee/ai_chat_bot.git

echo [1/4] Проверка Git...
if not exist ".git" (
    git init
    echo Инициализирован новый репозиторий.
) else (
    echo Репозиторий уже инициализирован.
)

echo.
echo [2/4] Удалённый репозиторий...
git remote remove origin 2>nul
git remote add origin %REPO%
echo origin = %REPO%

echo.
echo [3/4] Добавление файлов и коммит...
git add .
git status
git commit -m "Initial commit: Telegram AI bot"

echo.
echo [4/4] Отправка на GitHub...
git branch -M main
git push -u origin main

echo.
echo Готово. Репозиторий: https://github.com/chookee/ai_chat_bot
pause

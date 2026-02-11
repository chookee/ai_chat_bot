import asyncio
import logging
from typing import Optional
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.exceptions import AiogramError
from config import (
    BOT_TOKEN,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    DEEPSEEK_MODEL,
    GENAPI_MODEL,
    PROXYAPI_MODEL,
    AI_PROVIDER,
    TEMPERATURE,
    MAX_TOKENS,
)
from context_manager import ContextManager
from ollama_client import OllamaClient
from openai_client import OpenAIClient
from deepseek_client import DeepSeekClient
from genapi_client import GenAPIClient
from proxyapi_client import ProxyAPIClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация компонентов
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
context_manager = ContextManager(max_messages=20)

# Инициализация клиентов AI (с проверкой доступности)
ollama_client: Optional[OllamaClient] = None
openai_client: Optional[OpenAIClient] = None
deepseek_client: Optional[DeepSeekClient] = None
genapi_client: Optional[GenAPIClient] = None
proxyapi_client: Optional[ProxyAPIClient] = None

try:
    ollama_client = OllamaClient()
    logger.info("Ollama клиент успешно инициализирован")
except Exception as e:
    logger.warning("Не удалось инициализировать Ollama клиент: %s", e)

try:
    openai_client = OpenAIClient()
    logger.info("OpenAI клиент успешно инициализирован")
except Exception as e:
    logger.warning("Не удалось инициализировать OpenAI клиент: %s", e)

try:
    deepseek_client = DeepSeekClient()
    logger.info("DeepSeek клиент успешно инициализирован")
except Exception as e:
    logger.warning("Не удалось инициализировать DeepSeek клиент: %s", e)

try:
    genapi_client = GenAPIClient()
    logger.info("GenAPI клиент успешно инициализирован")
except Exception as e:
    logger.warning("Не удалось инициализировать GenAPI клиент: %s", e)

try:
    proxyapi_client = ProxyAPIClient()
    logger.info("ProxyAPI клиент успешно инициализирован")
except Exception as e:
    logger.warning("Не удалось инициализировать ProxyAPI клиент: %s", e)

# Выбор провайдера: из AI_PROVIDER или первый доступный по приоритету
_active_provider = None
if AI_PROVIDER:
    provider_map = {
        "ollama": ("Ollama", ollama_client),
        "openai": ("OpenAI", openai_client),
        "deepseek": ("DeepSeek", deepseek_client),
        "genapi": ("GenAPI", genapi_client),
        "proxyapi": ("ProxyAPI", proxyapi_client),
    }
    name, client = provider_map.get(AI_PROVIDER, (None, None))
    if client:
        _active_provider = (AI_PROVIDER, name, client)
        logger.info("Используется выбранный провайдер: %s", name)
    else:
        if AI_PROVIDER not in provider_map:
            raise RuntimeError(
                f"AI_PROVIDER должен быть один из: ollama, openai, deepseek, genapi, proxyapi (указано: {AI_PROVIDER!r})"
            )
            raise RuntimeError(
                f"Провайдер {name or AI_PROVIDER} выбран в AI_PROVIDER, но не удалось его инициализировать. "
                "Проверьте ключи и настройки в .env."
            )
else:
    for key, name, client in [
        ("ollama", "Ollama", ollama_client),
        ("openai", "OpenAI", openai_client),
        ("deepseek", "DeepSeek", deepseek_client),
        ("genapi", "GenAPI", genapi_client),
        ("proxyapi", "ProxyAPI", proxyapi_client),
    ]:
        if client:
            _active_provider = (key, name, client)
            logger.info("Используется провайдер по умолчанию: %s", name)
            break

if not _active_provider:
    raise RuntimeError(
        "Ни один из AI клиентов не доступен. Укажите AI_PROVIDER и настройте выбранный провайдер в .env."
    )

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    """Обработка команды /start"""
    _key, provider_name, _ = _active_provider
    welcome_text = (
        f"Привет! Я — AI-ассистент (сейчас используется: {provider_name}).\n\n"
        "Отправьте любое сообщение, и я отвечу.\n"
        "Очистить историю: /clear, /reset или напишите «очистить контекст»."
    )
    try:
        await message.answer(welcome_text)
    except AiogramError as e:
        logger.error(f"Ошибка при отправке сообщения пользователю {message.from_user.id}: {e}")

async def _clear_context_handler(message: types.Message):
    """Очистка контекста диалога для пользователя (общий обработчик для /clear и /reset)."""
    user_id = message.from_user.id
    try:
        context_manager.clear_context(user_id)
        await message.answer("Контекст диалога очищен!")
        logger.info(f"Контекст для пользователя {user_id} очищен")
    except Exception as e:
        logger.error(f"Ошибка при очистке контекста для пользователя {user_id}: {e}")
        await message.answer("Произошла ошибка при очистке контекста. Пожалуйста, попробуйте еще раз.")


@dp.message(Command("clear"))
@dp.message(Command("reset"))
async def clear_context(message: types.Message):
    await _clear_context_handler(message)

@dp.message()
async def handle_message(message: types.Message):
    """Обработка входящих сообщений"""
    user_id = message.from_user.id
    user_message = message.text.strip()
    
    try:
        # Проверка команды очистки контекста
        if user_message.lower() == "очистить контекст":
            context_manager.clear_context(user_id)
            await message.answer("Контекст диалога очищен!")
            logger.info(f"Контекст для пользователя {user_id} очищен по команде в тексте")
            return
        
        # Добавляем сообщение пользователя в контекст
        context_manager.add_message(user_id, "user", user_message)
        
        # Получаем полный контекст для генерации ответа
        context = context_manager.get_context(user_id)
        _key, provider_name, client = _active_provider

        logger.info("Использую %s для пользователя %s", provider_name, user_id)
        if _key == "ollama":
            response = client.generate_response(
                messages=context,
                model=OLLAMA_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        elif _key == "openai":
            response = client.generate_response(
                messages=context,
                model=OPENAI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        elif _key == "deepseek":
            response = client.generate_response(
                messages=context,
                model=DEEPSEEK_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        elif _key == "genapi":
            response = client.generate_response(
                messages=context,
                model=GENAPI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        elif _key == "proxyapi":
            response = client.generate_response(
                messages=context,
                model=PROXYAPI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        else:
            response = None

        if response:
            # Добавляем ответ бота в контекст
            context_manager.add_message(user_id, "assistant", response)
            
            # Отправляем ответ пользователю
            await message.answer(response)
            logger.info(f"Отправлен ответ пользователю {user_id}")
        else:
            error_msg = "Извините, возникла ошибка при генерации ответа. Пожалуйста, попробуйте еще раз."
            await message.answer(error_msg)
            logger.error(f"Не удалось получить ответ для пользователя {user_id}")
    
    except AiogramError as e:
        logger.error(f"Aiogram ошибка при обработке сообщения от пользователя {user_id}: {e}")
    except Exception as e:
        logger.error(f"Неизвестная ошибка при обработке сообщения от пользователя {user_id}: {e}")
        try:
            error_msg = "Произошла ошибка при обработке вашего сообщения. Пожалуйста, попробуйте еще раз."
            await message.answer(error_msg)
        except AiogramError as send_error:
            logger.error(f"Не удалось отправить сообщение об ошибке пользователю {user_id}: {send_error}")

async def main():
    """Запуск бота"""
    logger.info("Запуск Telegram-бота...")
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
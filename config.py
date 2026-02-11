import os
from dotenv import load_dotenv

load_dotenv()

# Токены и ключи
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
GENAPI_API_KEY = os.getenv("GENAPI_API_KEY") or os.getenv("GENAPI_KEY")
GENAPI_BASE_URL = os.getenv("GENAPI_BASE_URL", "https://api.gen-api.ru/api/v1")
PROXYAPI_API_KEY = os.getenv("PROXYAPI_API_KEY") or os.getenv("PROXYAPI_KEY")
PROXYAPI_BASE_URL = os.getenv("PROXYAPI_BASE_URL", "https://api.proxyapi.ru/openai/v1")

# Выбор провайдера при старте: ollama | openai | deepseek | genapi | proxyapi
AI_PROVIDER = (os.getenv("AI_PROVIDER") or "").strip().lower() or None

# Параметры генерации (для всех провайдеров)
def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except (ValueError, TypeError):
        return default

def _optional_int_env(name: str) -> int | None:
    s = (os.getenv(name) or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None

TEMPERATURE = _float_env("TEMPERATURE", 0.7)
MAX_TOKENS = _optional_int_env("MAX_TOKENS")

# Модели
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
GENAPI_MODEL = os.getenv("GENAPI_MODEL", "gpt-4o-mini")
PROXYAPI_MODEL = os.getenv("PROXYAPI_MODEL", "gpt-4o-mini")

# Проверка обязательных переменных
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не установлен в .env файле")

if not OPENAI_API_KEY:
    print("Предупреждение: OPENAI_API_KEY не установлен в .env файле")
if not DEEPSEEK_API_KEY:
    print("Предупреждение: DEEPSEEK_API_KEY не установлен в .env файле")
if not GENAPI_API_KEY:
    print("Предупреждение: GENAPI_API_KEY не установлен в .env файле")
if not PROXYAPI_API_KEY:
    print("Предупреждение: PROXYAPI_API_KEY не установлен в .env файле")
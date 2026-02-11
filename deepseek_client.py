import logging
from typing import List, Dict, Optional
import httpx
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Клиент для взаимодействия с DeepSeek API (OpenAI-совместимый)."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        timeout: int = 60,
    ):
        """
        Args:
            api_key: API ключ DeepSeek
            base_url: Базовый URL API (по умолчанию официальный DeepSeek)
            timeout: Время ожидания запроса (секунды)
        """
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = (base_url or DEEPSEEK_BASE_URL or "").strip() or "https://api.deepseek.com"
        if not self.api_key:
            raise ValueError("API ключ DeepSeek не предоставлен")

        http_client = httpx.Client(trust_env=False, timeout=timeout)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            http_client=http_client,
        )

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        Генерация ответа через DeepSeek API.

        Args:
            messages: Список сообщений в формате {"role": "...", "content": "..."}
            model: Модель (например, deepseek-chat, deepseek-reasoner)
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов в ответе

        Returns:
            Текст ответа или None при ошибке.
        """
        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            logger.info(
                "DeepSeek запрос: model=%s, temperature=%s, max_tokens=%s, context_size=%d",
                model, temperature, max_tokens, len(messages),
            )
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content

        except AuthenticationError:
            logger.error("Ошибка аутентификации DeepSeek: проверьте API ключ")
            return None
        except RateLimitError:
            logger.error("Превышено ограничение запросов к DeepSeek")
            return None
        except APIError as e:
            logger.error("Ошибка API DeepSeek: %s", e)
            return None
        except Exception as e:
            logger.error("Ошибка при запросе к DeepSeek: %s", e)
            return None

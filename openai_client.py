import logging
from typing import List, Dict, Optional
import httpx
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str = None, timeout: int = 30):
        """
        Клиент для взаимодействия с OpenAI
        
        Args:
            api_key: API ключ OpenAI
            timeout: Время ожидания запроса
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("API ключ OpenAI не предоставлен")
        
        http_client = httpx.Client(trust_env=False, timeout=timeout)
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=timeout,
            http_client=http_client,
        )
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Генерация ответа от OpenAI
        
        Args:
            messages: Список сообщений в формате {"role": "...", "content": "..."}
            model: Название модели OpenAI
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов в ответе
            
        Returns:
            Ответ от модели или None в случае ошибки
        """
        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            logger.info(
                "OpenAI запрос: model=%s, temperature=%s, max_tokens=%s, context_size=%d",
                model, temperature, max_tokens, len(messages),
            )
            response = self.client.chat.completions.create(**params)
            
            return response.choices[0].message.content
            
        except AuthenticationError:
            logger.error("Ошибка аутентификации при запросе к OpenAI: проверьте API ключ")
            return None
        except RateLimitError:
            logger.error("Превышено ограничение на количество запросов к OpenAI")
            return None
        except APIError as e:
            logger.error(f"API ошибка при запросе к OpenAI: {e}")
            return None
        except Exception as e:
            logger.error(f"Ошибка при запросе к OpenAI: {e}")
            return None

# Пример использования:
# client = OpenAIClient()
# messages = [
#     {"role": "user", "content": "Привет!"}
# ]
# response = client.generate_response(messages, model="gpt-4o-mini")
# print(response)
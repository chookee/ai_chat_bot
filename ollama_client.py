import logging
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434/api/chat", timeout: int = 300):
        """
        Клиент для взаимодействия с Ollama
        
        Args:
            base_url: URL для API Ollama
            timeout: Время ожидания запроса по умолчанию
        """
        self.base_url = base_url
        self.timeout = timeout
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Генерация ответа от Ollama
        
        Args:
            messages: Список сообщений в формате {"role": "...", "content": "..."}
            model: Название модели Ollama
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов в ответе (опционально)
            
        Returns:
            Ответ от модели или None в случае ошибки
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "options": {
                    "temperature": temperature
                },
                "stream": False
            }
            
            if max_tokens is not None:
                payload["options"]["num_predict"] = max_tokens

            logger.info(
                "Ollama запрос: model=%s, temperature=%s, max_tokens=%s, context_size=%d",
                model, temperature, max_tokens, len(messages),
            )
            response = requests.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content")
            
        except requests.exceptions.Timeout:
            logger.error(f"Таймаут при запросе к Ollama: превышено время ожидания {self.timeout} секунд")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Ошибка подключения к Ollama: убедитесь, что сервер Ollama запущен")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP ошибка при запросе к Ollama: {e}")
            logger.error(f"Статус код: {response.status_code}, текст: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к Ollama: {e}")
            return None
        except KeyError as e:
            logger.error(f"Ошибка при парсинге ответа от Ollama: {e}, полученный ответ: {result}")
            return None
        except Exception as e:
            logger.error(f"Неизвестная ошибка при работе с Ollama: {e}")
            return None

# Пример использования:
# client = OllamaClient()
# messages = [
#     {"role": "user", "content": "Привет!"}
# ]
# response = client.generate_response(messages, model="llama3.2")
# print(response)
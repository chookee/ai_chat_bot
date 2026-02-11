"""
Клиент для нативного API GenAPI (gen-api.ru).
Документация: https://gen-api.ru/docs/schema-work
"""
import logging
import time
from typing import List, Dict, Optional, Any

import httpx

from config import GENAPI_API_KEY, GENAPI_BASE_URL

logger = logging.getLogger(__name__)

# Базовый URL нативного API GenAPI (без openai/v1)
GENAPI_NATIVE_BASE = "https://api.gen-api.ru/api/v1"


def _messages_to_genapi(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Преобразует сообщения в формате OpenAI в формат GenAPI (content — массив частей)."""
    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        result.append({"role": role, "content": content})
    return result


def _extract_text_from_output(output: Any) -> Optional[str]:
    """Достаёт текст ответа из поля output ответа GenAPI."""
    if output is None:
        return None
    if isinstance(output, str):
        return output.strip() or None
    if isinstance(output, list):
        # Список строк (фрагменты ответа)
        if output and isinstance(output[0], str):
            return " ".join(str(x) for x in output).strip() or None
        for item in output:
            if isinstance(item, dict):
                # GenAPI: [{"message": {"role": "assistant", "content": "..."}, ...}]
                msg = item.get("message")
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str):
                        return c.strip() or None
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    return text.strip() or None
                if isinstance(text, list):
                    for part in text:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t = part.get("text")
                            if isinstance(t, str):
                                return t.strip() or None
        # Попытка: список сообщений в формате {"role": "assistant", "content": [...]}
        for item in output:
            if isinstance(item, dict) and item.get("role") == "assistant":
                c = item.get("content")
                if isinstance(c, str):
                    return c.strip() or None
                if isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t = part.get("text")
                            if isinstance(t, str):
                                return t.strip() or None
        return None
    if isinstance(output, dict):
        # Прямые поля (в т.ч. message — иногда API возвращает текст там)
        text = output.get("text") or output.get("content") or output.get("message")
        if isinstance(text, str):
            return text.strip() or None
        # OpenAI-подобный формат: choices[0].message.content
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str):
                    return c.strip() or None
        # Вложенный content как массив частей
        content = output.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        return t.strip() or None
        return None
    return None


class GenAPIClient:
    """Клиент для нативного API GenAPI (gen-api.ru)."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        timeout: int = 60,
        poll_interval: float = 2.0,
        poll_max_wait: float = 120.0,
    ):
        self.api_key = api_key or GENAPI_API_KEY
        self.base_url = (base_url or GENAPI_BASE_URL or "").strip().rstrip("/") or GENAPI_NATIVE_BASE
        # Если в конфиге оставили старый openai-путь — подменяем на нативный
        if "openai" in self.base_url:
            self.base_url = GENAPI_NATIVE_BASE
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.poll_max_wait = poll_max_wait
        if not self.api_key:
            raise ValueError("API ключ GenAPI не предоставлен")
        self._client = httpx.Client(
            trust_env=False,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

    def _request_url(self, model: str) -> str:
        """URL для запроса генерации: POST .../networks/{model}"""
        base = self.base_url.rstrip("/")
        if "/networks/" in base:
            return base
        return f"{base}/networks/{model}"

    def _status_url(self, request_id: int) -> str:
        """URL для опроса статуса: GET .../request/get/{request_id} (документация GenAPI)"""
        base = self.base_url.rstrip("/")
        return f"{base}/request/get/{request_id}"

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        Генерация ответа через нативный API GenAPI.
        Сначала пробуется синхронный режим (is_sync=true), при необходимости — long-polling.
        """
        try:
            genapi_messages = _messages_to_genapi(messages)
            body = {
                "messages": genapi_messages,
                "is_sync": True,
            }
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            if temperature is not None:
                body["temperature"] = temperature

            logger.info(
                "GenAPI запрос: model=%s, temperature=%s, max_tokens=%s, context_size=%d",
                model, temperature, max_tokens, len(messages),
            )
            url = self._request_url(model)
            resp = self._client.post(url, json=body)

            if resp.status_code == 404:
                logger.error("GenAPI: модель или endpoint не найден (404). URL: %s", url)
                return None
            if resp.status_code == 401:
                logger.error("Ошибка аутентификации GenAPI: проверьте API ключ")
                return None
            if resp.status_code == 402:
                logger.error("GenAPI: недостаточно средств на балансе")
                return None
            if resp.status_code >= 400:
                logger.error("GenAPI HTTP %s: %s", resp.status_code, resp.text[:500])
                return None

            data = resp.json()

            # Синхронный ответ: в теле уже есть status и output
            status = data.get("status")
            if status == "success":
                text = _extract_text_from_output(data.get("output"))
                if text:
                    return text
                logger.warning("GenAPI: status=success, но не удалось извлечь текст из output: %s", data.get("output"))
                return None

            if status == "failed":
                logger.error("GenAPI: задача завершилась с ошибкой: %s", data)
                return None

            # starting / processing — нужен long-polling
            request_id = data.get("request_id")
            if request_id is None:
                logger.error("GenAPI: нет request_id в ответе: %s", data)
                return None

            return self._poll_until_done(request_id)

        except httpx.TimeoutException:
            logger.error("GenAPI: таймаут запроса")
            return None
        except httpx.RequestError as e:
            logger.error("GenAPI: ошибка запроса: %s", e)
            return None
        except Exception as e:
            logger.error("Ошибка при запросе к GenAPI: %s", e)
            return None

    def _poll_until_done(self, request_id: int) -> Optional[str]:
        """Ожидание результата по request_id (long-polling)."""
        status_url = self._status_url(request_id)
        # Даём нейросети время принять задачу в работу (документация GenAPI)
        time.sleep(self.poll_interval)
        deadline = time.monotonic() + self.poll_max_wait
        while time.monotonic() < deadline:
            try:
                resp = self._client.get(status_url)
                if resp.status_code != 200:
                    logger.error("GenAPI poll HTTP %s: %s", resp.status_code, resp.text[:300])
                    return None
                data = resp.json()
                status = data.get("status")
                if status == "success":
                    # GenAPI возвращает текст в result или full_response, не в output
                    for key in ("result", "output", "full_response", "response"):
                        raw = data.get(key)
                        text = _extract_text_from_output(raw)
                        if text:
                            return text
                    logger.warning(
                        "GenAPI: status=success, не удалось извлечь текст. Ключи: %s. result type=%s, фрагмент: %s",
                        list(data.keys()),
                        type(data.get("result")).__name__,
                        str(data.get("result"))[:400] if data.get("result") is not None else "None",
                    )
                    return None
                if status == "failed":
                    logger.error("GenAPI: задача не выполнена: %s", data)
                    return None
            except Exception as e:
                logger.warning("GenAPI poll ошибка: %s", e)
            time.sleep(self.poll_interval)
        logger.error("GenAPI: превышено время ожидания результата (request_id=%s)", request_id)
        return None

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ContextManager:
    def __init__(self, max_messages: int = 20):
        """
        Менеджер контекста для хранения истории диалогов
        
        Args:
            max_messages: Максимальное количество сообщений в контексте
        """
        self.contexts: Dict[int, List[Dict[str, str]]] = {}
        self.max_messages = max_messages
    
    def get_context(self, user_id: int) -> List[Dict[str, str]]:
        """
        Получить контекст для пользователя
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Список сообщений в формате {"role": "...", "content": "..."}
        """
        if user_id not in self.contexts:
            self.contexts[user_id] = []
        return self.contexts[user_id]
    
    def add_message(self, user_id: int, role: str, content: str):
        """
        Добавить сообщение в контекст пользователя
        
        Args:
            user_id: ID пользователя
            role: Роль ("user", "assistant", "system")
            content: Содержание сообщения
        """
        if user_id not in self.contexts:
            self.contexts[user_id] = []
        
        self.contexts[user_id].append({
            "role": role,
            "content": content
        })
        
        # Ограничиваем длину контекста
        if len(self.contexts[user_id]) > self.max_messages:
            # Удаляем самые старые сообщения, но оставляем системные сообщения в начале
            # На случай, если есть системные сообщения в начале
            system_messages = []
            other_messages = []
            
            for msg in self.contexts[user_id]:
                if msg.get("role") == "system":
                    system_messages.append(msg)
                else:
                    other_messages.append(msg)
            
            # Сохраняем системные сообщения и последние other_messages
            if len(other_messages) > self.max_messages:
                other_messages = other_messages[-(self.max_messages - len(system_messages)):]
            
            self.contexts[user_id] = system_messages + other_messages
    
    def clear_context(self, user_id: int):
        """
        Очистить контекст для пользователя
        
        Args:
            user_id: ID пользователя
        """
        if user_id in self.contexts:
            del self.contexts[user_id]
            logger.info(f"Контекст для пользователя {user_id} очищен")
    
    def get_user_ids(self) -> List[int]:
        """
        Получить список всех ID пользователей, для которых есть контекст
        
        Returns:
            Список ID пользователей
        """
        return list(self.contexts.keys())
    
    def get_context_length(self, user_id: int) -> int:
        """
        Получить длину контекста для пользователя
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Количество сообщений в контексте
        """
        return len(self.contexts.get(user_id, []))
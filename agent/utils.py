from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

def init_memory():
    """Initialize the memory for contextual conversation."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"  # Updated from "answer" to "output"
    )

MEMORY = init_memory()
CHAT_HISTORY = MessagesPlaceholder(variable_name="chat_history")
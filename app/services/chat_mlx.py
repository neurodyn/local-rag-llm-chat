from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from app.services.mlx_pipeline import MLXPipeline
from pydantic import Field

class ChatMLX(BaseChatModel):
    """Chat model wrapper for MLX pipeline."""
    
    llm: MLXPipeline = Field(..., description="The MLX pipeline to use for generation")
    
    def __init__(self, **kwargs):
        """Initialize the ChatMLX model."""
        super().__init__(**kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mlx"
        
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the model."""
        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages)
        
        # Generate response
        response = self.llm._call(prompt, stop=stop, **kwargs)
        
        # Create chat generation
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        
        # Return chat result
        return ChatResult(generations=[generation])
    
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert a list of messages to a prompt string."""
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            else:
                prompt_parts.append(f"{message.type}: {message.content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:" 
from typing import Any, Dict, List, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field, ConfigDict
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
import mlx_lm

class MLXPipeline(BaseLLM):
    """MLX pipeline for text generation."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_id: str = Field(..., description="The model ID to use")
    pipeline_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional arguments for the pipeline")
    model: Any = Field(default=None, exclude=True)
    tokenizer: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        """Initialize the MLX pipeline."""
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        # Remove unsupported sampling parameters
        if 'temperature' in self.pipeline_kwargs:
            del self.pipeline_kwargs['temperature']
        if 'top_p' in self.pipeline_kwargs:
            del self.pipeline_kwargs['top_p']
        if 'repetition_penalty' in self.pipeline_kwargs:
            del self.pipeline_kwargs['repetition_penalty']
            
        self.model, self.tokenizer = mlx_lm.load(self.model_id)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mlx"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate responses for the given prompts."""
        # Remove unsupported sampling parameters
        if 'temperature' in kwargs:
            del kwargs['temperature']
        if 'top_p' in kwargs:
            del kwargs['top_p']
        if 'repetition_penalty' in kwargs:
            del kwargs['repetition_penalty']
            
        # Combine pipeline kwargs with any additional kwargs
        generation_kwargs = {**self.pipeline_kwargs, **kwargs}
        
        # Generate responses for each prompt
        responses = []
        for prompt in prompts:
            response = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt,
                **generation_kwargs
            )
            responses.append(response)
            
        # Format the response according to LangChain's expected format
        return {
            "generations": [
                {"text": response, "generation_info": None}
                for response in responses
            ],
            "llm_output": None,
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from the model."""
        # Remove unsupported sampling parameters
        if 'temperature' in kwargs:
            del kwargs['temperature']
        if 'top_p' in kwargs:
            del kwargs['top_p']
        if 'repetition_penalty' in kwargs:
            del kwargs['repetition_penalty']
            
        # Combine pipeline kwargs with any additional kwargs
        generation_kwargs = {**self.pipeline_kwargs, **kwargs}
        
        # Generate response
        response = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt,
            **generation_kwargs
        )
        
        return response
    
    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "MLXPipeline":
        """Create a pipeline from a model ID."""
        if pipeline_kwargs is None:
            pipeline_kwargs = {}
            
        # Remove unsupported sampling parameters
        if 'temperature' in pipeline_kwargs:
            del pipeline_kwargs['temperature']
        if 'top_p' in pipeline_kwargs:
            del pipeline_kwargs['top_p']
        if 'repetition_penalty' in pipeline_kwargs:
            del pipeline_kwargs['repetition_penalty']
            
        return cls(model_id=model_id, pipeline_kwargs=pipeline_kwargs, **kwargs) 
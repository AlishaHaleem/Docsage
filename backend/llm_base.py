from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class GenerationConfig(BaseModel):
    """Configuration for text generation"""
    temperature: float = Field(0.7, ge=0.0, le=2.0, 
                             description="Controls randomness. Lower = more deterministic")
    max_tokens: int = Field(1024, gt=0, 
                           description="Maximum number of tokens to generate")
    top_p: float = Field(1.0, ge=0.0, le=1.0, 
                        description="Nucleus sampling parameter")
    frequency_penalty: float = Field(0.0, ge=0.0, le=2.0, 
                                   description="Penalize new tokens based on frequency")
    presence_penalty: float = Field(0.0, ge=0.0, le=2.0, 
                                  description="Penalize new tokens based on presence")

class LLMBase(ABC):
    @abstractmethod
    def generate_text(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
from typing import List, Dict, Optional
import json
from .llm_base import LLMBase, GenerationConfig
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, llm_client: LLMBase):
        self.llm_client = llm_client
        self.default_config = GenerationConfig(
            temperature=0.7,
            max_tokens=512
        )

    def query_answer(
        self, 
        question: str, 
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Get answer to question with configurable parameters"""
        config = config or self.default_config
        return self.llm_client.generate_text(question, config)

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text"""
        return self.llm_client.get_embedding(text)

    def summarize_text(
        self, 
        text: str, 
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Summarize text with configurable parameters"""
        config = config or GenerationConfig(max_tokens=256)
        prompt = f"Summarize the following in under 200 words:\n{text}"
        return self.llm_client.generate_text(prompt, config)

    def generate_quiz(
        self, 
        text: str, 
        num_questions: int = 3,
        config: Optional[GenerationConfig] = None
    ) -> List[Dict]:
        """Generate quiz with configurable parameters"""
        config = config or GenerationConfig(max_tokens=1024)
        return self.llm_client.generate_quiz(text, num_questions, config)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return self.llm_client.count_tokens(text)

    def validate_prompt_length(
        self, 
        prompt: str, 
        max_tokens: int = 512
    ) -> bool:
        """Check if prompt + max_tokens is within model limits"""
        try:
            tokens = self.count_tokens(prompt)
            return tokens + max_tokens <= 8192  # Adjust based on model
        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}")
            return False
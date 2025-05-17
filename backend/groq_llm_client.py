import requests
from typing import List, Dict, Optional
import json
from tiktoken import encoding_for_model
from .llm_base import LLMBase, GenerationConfig
import logging

logger = logging.getLogger(__name__)

class GroqLLMClient(LLMBase):
    def __init__(self, api_key: str, default_model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.endpoint = "https://api.groq.com/openai/v1"
        self.default_model = default_model
        try:
            self.encoder = encoding_for_model("gpt-4")  # Using GPT-4 tokenizer as approximation
        except KeyError:
            logger.warning("GPT-4 tokenizer not found, using default")
            self.encoder = encoding_for_model("gpt-3.5-turbo")

    def count_tokens(self, text: str) -> int:
        """Count tokens using the encoder"""
        return len(self.encoder.encode(text))

    def _validate_prompt_length(self, prompt: str, max_tokens: int) -> None:
        """Validate prompt + max_tokens doesn't exceed model limits"""
        prompt_tokens = self.count_tokens(prompt)
        total = prompt_tokens + max_tokens
        model_limit = 8192 if "8192" in self.default_model else 4096
        
        if total > model_limit:
            raise ValueError(
                f"Prompt ({prompt_tokens} tokens) + max_tokens ({max_tokens}) "
                f"exceeds model limit of {model_limit} tokens"
            )
        if prompt_tokens > model_limit:
            raise ValueError(
                f"Prompt is too long ({prompt_tokens} tokens), "
                f"model limit is {model_limit} tokens"
            )

    def generate_text(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate text from prompt with configurable parameters
        
        Args:
            prompt: Input text prompt
            config: Generation parameters (temperature, max_tokens etc)
            
        Returns:
            Generated text
            
        Raises:
            ValueError: If prompt is too long
            RuntimeError: If API request fails
        """
        config = config or GenerationConfig()
        self._validate_prompt_length(prompt, config.max_tokens)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty
        }

        try:
            response = requests.post(
                f"{self.endpoint}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise RuntimeError(f"Failed to generate text: {str(e)}")

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": text,
            "model": "text-embedding-3-small"  # Using smaller embedding model
        }

        try:
            response = requests.post(
                f"{self.endpoint}/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise RuntimeError(f"Failed to get embedding: {str(e)}")

    def summarize_chunk(self, chunk: str, config: Optional[GenerationConfig] = None) -> str:
        """Summarize a text chunk with configurable parameters"""
        prompt = f"Summarize the following text in 3-5 sentences:\n{chunk}"
        return self.generate_text(prompt, config)

    def generate_quiz(
        self, 
        content: str, 
        num_questions: int = 5,
        config: Optional[GenerationConfig] = None
    ) -> List[Dict]:
        """Generate quiz questions with configurable parameters"""
        prompt = (
            f"Generate {num_questions} multiple-choice questions from the text below. "
            "Each question should have 4 options (A-D) and indicate the correct answer.\n"
            "Respond in this JSON format:\n"
            "[{\"question\": \"...\", \"options\": [\"...\", \"...\", \"...\", \"...\"], \"answer\": \"...\"}]\n"
            f"Text:\n{content}"
        )
        
        quiz_text = self.generate_text(prompt, config)
        try:
            return json.loads(quiz_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse quiz JSON: {str(e)}")
            raise RuntimeError("Failed to parse quiz response")
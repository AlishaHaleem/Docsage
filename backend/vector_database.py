import os
from typing import List, Dict
import pinecone
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
import structlog
from backend.groq_llm_client import GroqLLMClient
from langchain.embeddings import OpenAIEmbeddings  # Assuming you're using OpenAI for embeddings

# Initialize logger (using structlog or any other logging framework)
logger = structlog.get_logger()

# Load environment variables from .env file
load_dotenv()

class VectorDB:
    def __init__(self):
        # Load environment variables
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Check if all necessary environment variables are provided
        if not all([self.api_key, self.environment, self.index_name, self.groq_api_key, self.openai_api_key]):
            raise ValueError("Missing required environment variables.")

        # Initialize Pinecone
        try:
            pinecone.init(api_key=self.api_key, environment=self.environment)
            if self.index_name not in pinecone.list_indexes():
                raise ValueError(f"Pinecone index '{self.index_name}' does not exist.")
            self.index = pinecone.Index(self.index_name)
            self.groq_client = GroqLLMClient(api_key=self.groq_api_key)
            self.embedding_client = OpenAIEmbeddings(api_key=self.openai_api_key)
            logger.info("Connected to Pinecone, Groq, and OpenAI.")
        except Exception as e:
            logger.error("Failed to initialize VectorDB", error=str(e))
            raise

    def upsert(self, documents: List[Dict[str, str]]) -> None:
        try:
            vectors = []
            for i, doc in enumerate(documents):
                # Get embedding using OpenAI for document text
                embedding = self.embedding_client.get_embedding(doc["text"])
                vectors.append({
                    "id": doc["id"],
                    "values": embedding,
                    "metadata": {"text": doc["text"]}
                })
            # Upsert vectors into Pinecone
            self.index.upsert(vectors=vectors)
            logger.info("Upserted documents into Pinecone", count=len(documents))
        except Exception as e:
            logger.error("Failed to upsert documents", error=str(e))
            raise

    def query(self, question: str, top_k: int = 5) -> List[Dict]:
        try:
            # Get embedding for the question
            query_embedding = self.embedding_client.get_embedding(question)
            # Query Pinecone with the question embedding
            response = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            # Process results from Pinecone
            results = [{"text": match.metadata["text"], "score": match.score} for match in response.matches]
            logger.info("Query successful", query=question, results_found=len(results))
            return results
        except Exception as e:
            logger.error("Failed to query documents", error=str(e))
            raise

# app/rag_pipeline.py

import logging
from typing import List, Dict
from backend.groq_llm_client import GroqLLMClient
from backend.vector_database import VectorDB
from backend.document_processor import DocumentProcessor
from backend.llm_service import LLMService

class RAG:
    def __init__(self, groq_client: GroqLLMClient, vector_db: VectorDB, doc_processor: DocumentProcessor):
        self.groq_client = groq_client
        self.vector_db = vector_db
        self.doc_processor = doc_processor
        self.llm_service = LLMService(groq_client)

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            query_embedding = self.groq_client.get_embedding(query)
            relevant_chunks = self.vector_db.query(query=query, top_k=top_k)
            return relevant_chunks
        except Exception as e:
            logging.error(f"Error retrieving relevant chunks: {e}")
            raise

    def generate_answer(self, query: str) -> str:
        try:
            relevant_chunks = self.retrieve_relevant_chunks(query)
            context = "\n".join([chunk["text"] for chunk in relevant_chunks])
            prompt = f"Context:\n{context}\nQuestion: {query}\nAnswer:"
            return self.groq_client.generate_text(prompt)
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            raise

    def process_and_store_documents(self, documents: List[str], doc_type: str = "txt") -> None:
        try:
            if doc_type == "pdf":
                processed_docs = self.doc_processor.process_pdf(documents)
            else:
                processed_docs = self.doc_processor.process_txt(documents)
            self.doc_processor.store_documents(processed_docs)
        except Exception as e:
            logging.error(f"Error processing and storing documents: {e}")
            raise

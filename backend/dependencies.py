from fastapi import Depends
from backend.groq_llm_client import GroqLLMClient
from backend.vector_database import VectorDB
from backend.document_processor import DocumentProcessor
from backend.rag_pipeline import RAG
import os

def get_groq_client():
    return GroqLLMClient(api_key=os.getenv("GROQ_API_KEY"))

def get_vector_db():
    return VectorDB()

def get_document_processor(vector_db: VectorDB = Depends(get_vector_db)):
    return DocumentProcessor(vector_db=vector_db)

def get_rag(
    groq_client: GroqLLMClient = Depends(get_groq_client),
    vector_db: VectorDB = Depends(get_vector_db),
    doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    return RAG(groq_client=groq_client, vector_db=vector_db, doc_processor=doc_processor)

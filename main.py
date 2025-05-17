import os
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from backend.vector_database import VectorDB
from backend.document_processor import DocumentProcessor
from backend.groq_llm_client import GroqLLMClient
from backend.llm_service import LLMService
from backend.rag_pipeline import RAG
from backend.api import router as api_router
from backend.dependencies import get_rag
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables once
load_dotenv()

app = FastAPI(title="RAG Chatbot API",
             description="Retrieval-Augmented Generation Chatbot",
             version="1.0.0")

app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG chatbot!"}

def initialize_components():
    """Initialize and return all RAG components"""
    try:
        # Initialize LLM client with configurable parameters
        llm_client = GroqLLMClient(
            api_key=os.getenv("GROQ_API_KEY"),
            default_model="llama3-8b-8192"
        )
        
        # Initialize services
        vector_db = VectorDB()
        doc_processor = DocumentProcessor(vector_db=vector_db)
        llm_service = LLMService(llm_client)
        
        # Create RAG pipeline
        rag_pipeline = RAG(
            groq_client=llm_client,
            vector_db=vector_db,
            doc_processor=doc_processor
        )
        
        logger.info("All components initialized successfully")
        return rag_pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

def main():
    """Example usage of the RAG pipeline"""
    try:
        rag = initialize_components()
        
        # Example document processing
        document_paths = ['example.pdf', 'example.txt']
        
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                logger.warning(f"Document not found: {doc_path}")
                continue
                
            file_ext = os.path.splitext(doc_path)[1].lower()
            doc_type = "pdf" if file_ext == ".pdf" else "txt"
            
            try:
                logger.info(f"Processing {doc_path}...")
                rag.process_and_store_documents([doc_path], doc_type=doc_type)
                logger.info(f"Successfully processed {doc_path}")
            except Exception as e:
                logger.error(f"Failed to process {doc_path}: {str(e)}")

        # Example query
        question = "What is the impact of climate change on marine ecosystems?"
        try:
            answer = rag.generate_answer(question)
            logger.info(f"Question: {question}")
            logger.info(f"Answer: {answer}")
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )
    
    # Alternatively, to run the example usage:
    # main()
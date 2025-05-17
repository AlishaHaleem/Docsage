import os
import re
from typing import List, Dict, Optional
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.vector_database import VectorDB
import structlog
from unicodedata import normalize

# Initialize logger
logger = structlog.get_logger()

class DocumentProcessor:
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db  # Instance of VectorDB to interact with Pinecone
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

    def _validate_file(self, file_path: str) -> None:
        """Validate file exists and is accessible."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Unable to read file: {file_path}")

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text before processing.
        """
        # Normalize unicode characters
        text = normalize("NFKC", text)
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters (keep alphanumeric, basic punctuation)
        text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)
        
        # Standardize quotes
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace("‘", "'").replace("’", "'")
        
        return text

    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Parse a PDF file, clean and chunk it into documents.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of document chunks with IDs and text
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
            ValueError: If PDF parsing fails
        """
        try:
            self._validate_file(pdf_path)
            
            text = extract_text(pdf_path)
            if not text.strip():
                raise ValueError(f"PDF appears to be empty or couldn't be parsed: {pdf_path}")
                
            logger.info(f"Extracted text from PDF {pdf_path}", char_count=len(text))
            return self._process_text(text)
        except Exception as e:
            logger.error("Failed to process PDF", error=str(e), file_path=pdf_path)
            raise

    def process_txt(self, txt_path: str) -> List[Dict[str, str]]:
        """
        Process a text file (TXT).
        
        Args:
            txt_path: Path to the text file
            
        Returns:
            List of document chunks with IDs and text
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
            UnicodeDecodeError: If file encoding is invalid
        """
        try:
            self._validate_file(txt_path)
            
            with open(txt_path, "r", encoding="utf-8") as file:
                text = file.read()
                
            if not text.strip():
                raise ValueError(f"Text file is empty: {txt_path}")
                
            logger.info(f"Loaded text from {txt_path}", char_count=len(text))
            return self._process_text(text)
        except UnicodeDecodeError:
            # Try with different encoding if utf-8 fails
            try:
                with open(txt_path, "r", encoding="latin-1") as file:
                    text = file.read()
                logger.warning(f"Used latin-1 encoding for {txt_path}")
                return self._process_text(text)
            except Exception as e:
                logger.error("Failed to process TXT with fallback encoding", 
                           error=str(e), file_path=txt_path)
                raise
        except Exception as e:
            logger.error("Failed to process TXT", error=str(e), file_path=txt_path)
            raise

    def _process_text(self, text: str) -> List[Dict[str, str]]:
        """
        Clean, split, and prepare text for embedding.
        
        Args:
            text: Raw text to process
            
        Returns:
            List of document chunks with IDs and cleaned text
        """
        try:
            # Clean text first
            cleaned_text = self._clean_text(text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Create document structure
            documents = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():  # Skip empty chunks
                    continue
                    
                documents.append({
                    "id": f"chunk_{i}_{hash(chunk)}",  # Include hash for uniqueness
                    "text": chunk
                })
                
            logger.info(
                "Processed text into chunks",
                original_length=len(text),
                cleaned_length=len(cleaned_text),
                chunk_count=len(documents)
            )
            return documents
        except Exception as e:
            logger.error("Failed to process text", error=str(e))
            raise

    def store_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Store processed documents into the VectorDB.
        
        Args:
            documents: List of document chunks to store
            
        Raises:
            ValueError: If documents list is empty
            RuntimeError: If vector DB operation fails
        """
        try:
            if not documents:
                raise ValueError("No documents to store")
                
            # Batch upsert to avoid timeouts with large documents
            batch_size = 100  # Adjust based on your vector DB's limits
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vector_db.upsert(batch)
                
            logger.info(
                "Stored documents into VectorDB",
                total_count=len(documents),
                batch_size=batch_size
            )
        except Exception as e:
            logger.error("Failed to store documents in VectorDB", error=str(e))
            raise
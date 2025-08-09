import os
import requests
import numpy as np
import pickle
import hashlib
import asyncio
import re
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import google.generativeai as genai
import PyPDF2
from io import BytesIO
import logging
from functools import lru_cache
import threading
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCXGyMsqh7yDcqk7CAqGBMh-owevThyPAQ")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model only
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Lightweight TF-IDF vectorizer (much faster than sentence transformers)
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,  # Limit features for speed
    stop_words='english',
    ngram_range=(1, 2),
    lowercase=True,
    max_df=0.85,
    min_df=2
)

# Security
security = HTTPBearer()
EXPECTED_TOKEN = "Bearer 1fd4ee76a5f7d0249cf4262bd779267a6e246992896f2ee373d16e9a19254ef5"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory caches (with size limits for memory efficiency)
DOCUMENT_CACHE = {}  # URL -> processed document data
ANSWER_CACHE = {}    # query_hash -> answer (max 50 items)
CACHE_LOCK = threading.Lock()
MAX_ANSWER_CACHE = 50

# Define lifespan context manager BEFORE FastAPI app initialization
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Lightweight Query-Retrieval System")
    logger.info(f"TF-IDF vectorizer initialized with {tfidf_vectorizer.max_features} features")
    yield
    # Shutdown (if needed)
    logger.info("Shutting down Lightweight Query-Retrieval System")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Lightweight Query-Retrieval System", lifespan=lifespan)

# Pydantic models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class LightweightDocumentProcessor:
    def __init__(self):
        self.chunk_size = 500  # Smaller chunks for faster processing
        self.chunk_overlap = 25
    
    def get_url_hash(self, url: str) -> str:
        """Generate hash for URL to use as cache key"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def simple_text_splitter(self, text: str) -> List[str]:
        """Lightweight text splitting without langchain"""
        # Split by sentences and group into chunks
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def extract_text_from_pdf_url(self, url: str) -> str:
        """Extract text from PDF URL with caching"""
        url_hash = self.get_url_hash(url)
        
        with CACHE_LOCK:
            if url_hash in DOCUMENT_CACHE:
                logger.info(f"Using cached document for URL hash: {url_hash}")
                return DOCUMENT_CACHE[url_hash]['text']
        
        try:
            # Download with timeout and size limit
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            # Limit file size to prevent memory issues
            pdf_content = BytesIO()
            total_size = 0
            max_size = 10 * 1024 * 1024  # 10MB limit
            
            for chunk in response.iter_content(chunk_size=4096):
                total_size += len(chunk)
                if total_size > max_size:
                    logger.warning("PDF too large, truncating...")
                    break
                pdf_content.write(chunk)
            
            pdf_content.seek(0)
            
            pdf_reader = PyPDF2.PdfReader(pdf_content)
            text = ""
            # Limit pages for faster processing
            max_pages = min(30, len(pdf_reader.pages))
            
            for i, page in enumerate(pdf_reader.pages[:max_pages]):
                page_text = page.extract_text()
                text += page_text + "\n"
                
                # Process incrementally to avoid memory spikes
                if i % 5 == 0 and i > 0:
                    logger.info(f"Processed page {i+1}/{max_pages}")
            
            # Clean the text
            text = self.preprocess_text(text)
            
            # Cache the document
            with CACHE_LOCK:
                DOCUMENT_CACHE[url_hash] = {
                    'text': text,
                    'timestamp': time.time(),
                    'url': url
                }
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")
    
    def get_processed_document(self, url: str):
        """Get or create processed document with TF-IDF vectorization"""
        url_hash = self.get_url_hash(url)
        
        with CACHE_LOCK:
            if url_hash in DOCUMENT_CACHE and 'tfidf_matrix' in DOCUMENT_CACHE[url_hash]:
                logger.info("Using cached TF-IDF vectors")
                return DOCUMENT_CACHE[url_hash]
        
        # Extract text
        text = self.extract_text_from_pdf_url(url)
        
        # Split into chunks
        chunks = self.simple_text_splitter(text)
        logger.info(f"Created {len(chunks)} chunks")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks extracted")
        
        # Create TF-IDF vectors (much faster than sentence transformers)
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)
        except Exception as e:
            logger.error(f"TF-IDF vectorization failed: {e}")
            # Fallback: use simple word matching
            tfidf_matrix = None
        
        # Cache everything
        with CACHE_LOCK:
            DOCUMENT_CACHE[url_hash].update({
                'chunks': chunks,
                'tfidf_matrix': tfidf_matrix,
                'vectorized': True
            })
        
        logger.info(f"Processed document: {len(chunks)} chunks, {len(text)} chars")
        return DOCUMENT_CACHE[url_hash]
    
    def keyword_search(self, chunks: List[str], query: str, top_k: int = 2) -> List[str]:
        """Simple keyword-based search as fallback"""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks[:top_k]]
    
    def tfidf_search(self, doc_data: dict, query: str, top_k: int = 2) -> List[str]:
        """Fast TF-IDF based search"""
        if doc_data['tfidf_matrix'] is None:
            # Fallback to keyword search
            return self.keyword_search(doc_data['chunks'], query, top_k)
        
        try:
            # Transform query using the same vectorizer
            query_vector = tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, doc_data['tfidf_matrix']).flatten()
            
            # Get top k indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Filter by minimum similarity threshold
            relevant_chunks = []
            for idx in top_indices:
                if similarities[idx] > 0.05:  # Minimum threshold
                    relevant_chunks.append(doc_data['chunks'][idx])
            
            return relevant_chunks if relevant_chunks else doc_data['chunks'][:2]
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            # Fallback to keyword search
            return self.keyword_search(doc_data['chunks'], query, top_k)

class FastQueryAnswerer:
    @staticmethod
    def manage_answer_cache():
        """Keep answer cache size under control"""
        if len(ANSWER_CACHE) >= MAX_ANSWER_CACHE:
            # Remove oldest 10 entries
            sorted_items = sorted(ANSWER_CACHE.items())
            for i in range(10):
                if sorted_items:
                    key = sorted_items[i][0]
                    ANSWER_CACHE.pop(key, None)
    
    @staticmethod
    def get_cached_answer(query_hash: str, context_hash: str) -> Optional[str]:
        """Check cache for answer"""
        cache_key = f"{query_hash}_{context_hash}"
        return ANSWER_CACHE.get(cache_key)
    
    @staticmethod
    def cache_answer(query_hash: str, context_hash: str, answer: str):
        """Cache answer with size management"""
        FastQueryAnswerer.manage_answer_cache()
        cache_key = f"{query_hash}_{context_hash}"
        ANSWER_CACHE[cache_key] = answer
    
    @staticmethod
    def create_concise_prompt(context: str, query: str) -> str:
        """Ultra-concise prompt for faster processing"""
        # Limit context to save tokens and processing time
        limited_context = context[:800]
        
        return f"""Context: {limited_context}

Question: {query}

Give a direct answer in 1-2 sentences. If not in context, say "Not mentioned in document."

Answer:"""
    
    @staticmethod
    def extract_key_info(query: str, context: str) -> str:
        """Extract key information without full LLM processing for simple queries"""
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Simple pattern matching for common query types
        if 'grace period' in query_lower and 'premium' in query_lower:
            # Look for specific patterns
            pattern = r'grace period[^.]*?(\d+)[^.]*?days?[^.]*?premium'
            match = re.search(pattern, context_lower)
            if match:
                return f"A grace period of {match.group(1)} days is provided for premium payment."
        
        if 'waiting period' in query_lower:
            pattern = r'waiting period[^.]*?(\d+)[^.]*?(months?|years?)'
            match = re.search(pattern, context_lower)
            if match:
                return f"There is a waiting period of {match.group(1)} {match.group(2)}."
        
        # Return None to fallback to full LLM processing
        return None
    
    @staticmethod
    def generate_fast_answer(context: str, query: str) -> str:
        """Generate answer with multiple optimization layers"""
        # Create hashes for caching
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]  # Shorter hash
        context_hash = hashlib.md5(context[:500].encode()).hexdigest()[:8]  # Shorter hash
        
        # Check cache first
        cached = FastQueryAnswerer.get_cached_answer(query_hash, context_hash)
        if cached:
            return cached
        
        # Try pattern-based extraction first
        quick_answer = FastQueryAnswerer.extract_key_info(query, context)
        if quick_answer:
            FastQueryAnswerer.cache_answer(query_hash, context_hash, quick_answer)
            return quick_answer
        
        # Fallback to LLM
        prompt = FastQueryAnswerer.create_concise_prompt(context, query)
        
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=50,   # Very limited output
                    temperature=0,          # Deterministic
                    top_p=0.8,             # Focus on most likely tokens
                )
            )
            answer = response.text.strip()
            
            # Cache the answer
            FastQueryAnswerer.cache_answer(query_hash, context_hash, answer)
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Sorry, couldn't generate an answer due to an error."

# Global processor
doc_processor = LightweightDocumentProcessor()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme.lower() != "bearer" or credentials.credentials != EXPECTED_TOKEN.split(" ")[1]:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_lightweight_query_retrieval(request: QueryRequest):
    """Optimized main endpoint for Render free tier"""
    start_time = time.time()
    
    try:
        # Get processed document (cached if available)
        doc_data = doc_processor.get_processed_document(request.documents)
        
        answers = []
        
        # Process queries efficiently
        for query in request.questions:
            logger.info(f"Processing: {query[:20]}...")
            
            # Use lightweight TF-IDF search
            relevant_chunks = doc_processor.tfidf_search(doc_data, query, top_k=2)
            
            if not relevant_chunks:
                answers.append("No relevant information found in document.")
                continue
            
            context = "\n".join(relevant_chunks)
            answer = FastQueryAnswerer.generate_fast_answer(context, query)
            answers.append(answer)
        
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check with minimal cache stats"""
    return {
        "status": "healthy",
        "cached_docs": len(DOCUMENT_CACHE),
        "cached_answers": len(ANSWER_CACHE),
        "message": "Lightweight Query-Retrieval System is running"
    }

@app.post("/clear-cache", dependencies=[Depends(verify_token)])
async def clear_cache():
    """Clear all caches to free memory"""
    with CACHE_LOCK:
        DOCUMENT_CACHE.clear()
        ANSWER_CACHE.clear()
    
    return {"message": "All caches cleared"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Lightweight Query-Retrieval API is running"}

@app.head("/")
async def head_root():
    """Handle HEAD requests for health checks"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Default to 8000, Render will set PORT
    print(f"Starting server on host 0.0.0.0 port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
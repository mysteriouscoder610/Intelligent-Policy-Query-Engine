import os
import requests
import numpy as np
import faiss
import pickle
import hashlib
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import PyPDF2
from io import BytesIO
import logging
from functools import lru_cache
import threading
import time
from dotenv import load_dotenv

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Structured JSON Query-Retrieval System", version="3.0.0")

# Configure Gemini API
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables or .env file")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize models (load once)
sentence_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Security
security = HTTPBearer()
EXPECTED_TOKEN = "Bearer 1fd4ee76a5f7d0249cf4262bd779267a6e246992896f2ee373d16e9a19254ef5"

# In-memory caches
DOCUMENT_CACHE = {}
EMBEDDING_CACHE = {}
ANSWER_CACHE = {}
CACHE_LOCK = threading.Lock()

# Updated Pydantic models for structured responses
class DecisionResponse(BaseModel):
    decision: str  # "approved", "rejected", "pending", "requires_review"
    amount: Optional[float] = None  # Amount if applicable
    justification: str  # Detailed explanation
    relevant_clauses: List[str]  # List of clause references/numbers
    confidence_score: Optional[float] = None  # Confidence in the decision

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[DecisionResponse]

class OptimizedDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
            length_function=len,
        )
    
    def get_url_hash(self, url: str) -> str:
        """Generate hash for URL to use as cache key"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def extract_text_from_pdf_url(self, url: str) -> str:
        """Extract text from PDF URL with caching"""
        url_hash = self.get_url_hash(url)
        
        with CACHE_LOCK:
            if url_hash in DOCUMENT_CACHE:
                logger.info(f"Using cached document for URL hash: {url_hash}")
                return DOCUMENT_CACHE[url_hash]['text']
        
        try:
            response = requests.get(url, timeout=15, stream=True)
            response.raise_for_status()
            
            pdf_content = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                pdf_content.write(chunk)
            pdf_content.seek(0)
            
            pdf_reader = PyPDF2.PdfReader(pdf_content)
            text = ""
            max_pages = min(50, len(pdf_reader.pages))
            
            for i, page in enumerate(pdf_reader.pages[:max_pages]):
                text += page.extract_text() + "\n"
                if i % 10 == 0:
                    logger.info(f"Processed page {i+1}/{max_pages}")
            
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
        """Get or create processed document with vector store"""
        url_hash = self.get_url_hash(url)
        
        with CACHE_LOCK:
            if url_hash in DOCUMENT_CACHE and 'vector_data' in DOCUMENT_CACHE[url_hash]:
                logger.info("Using cached vector store")
                return DOCUMENT_CACHE[url_hash]
        
        text = self.extract_text_from_pdf_url(url)
        chunks = self.text_splitter.split_text(text)
        
        # Create embeddings
        embeddings = sentence_model.encode(
            chunks,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        with CACHE_LOCK:
            DOCUMENT_CACHE[url_hash].update({
                'chunks': chunks,
                'index': index,
                'vector_data': True
            })
        
        logger.info(f"Processed document: {len(chunks)} chunks, {len(text)} chars")
        return DOCUMENT_CACHE[url_hash]
    
    def semantic_search(self, doc_data: dict, query: str, top_k: int = 3) -> List[tuple]:
        """Enhanced semantic search returning chunks with scores"""
        query_embedding = sentence_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        scores, indices = doc_data['index'].search(query_embedding.astype('float32'), top_k)
        
        relevant_chunks = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(doc_data['chunks']) and score > 0.1:
                relevant_chunks.append((doc_data['chunks'][idx], float(score)))
        
        return relevant_chunks

class StructuredAnswerGenerator:
    @staticmethod
    def extract_clause_references(text: str) -> List[str]:
        """Extract clause references from text"""
        # Common patterns for clause references
        patterns = [
            r'(?i)clause\s+(\d+(?:\.\d+)*)',
            r'(?i)section\s+(\d+(?:\.\d+)*)',
            r'(?i)article\s+(\d+(?:\.\d+)*)',
            r'(?i)paragraph\s+(\d+(?:\.\d+)*)',
            r'(?i)subsection\s+(\d+(?:\.\d+)*)',
            r'(?i)provision\s+(\d+(?:\.\d+)*)',
            r'\b(\d+(?:\.\d+)*)\s*(?:of|under|per|as per)',
            r'(?i)(?:clause|section|article|para)\s*[:\-]\s*(\d+(?:\.\d+)*)'
        ]
        
        clauses = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    clauses.add(match[0])
                else:
                    clauses.add(match)
        
        return list(clauses)
    
    @staticmethod
    def extract_amount(text: str) -> Optional[float]:
        """Extract monetary amounts from text"""
        # Patterns for different currency formats
        amount_patterns = [
            r'(?i)(?:rs\.?|rupees?|inr|₹)\s*([0-9,]+(?:\.[0-9]{2})?)',
            r'(?i)\$\s*([0-9,]+(?:\.[0-9]{2})?)',
            r'(?i)([0-9,]+(?:\.[0-9]{2})?)\s*(?:rs\.?|rupees?|inr|₹|\$)',
            r'(?i)(?:amount|sum|value|worth|cost|price|fee|premium)\s*(?:of|is|:)?\s*(?:rs\.?|rupees?|inr|₹|\$)?\s*([0-9,]+(?:\.[0-9]{2})?)',
            r'\b([0-9,]+(?:\.[0-9]{2})?)\s*(?:lakh|crore|thousand|million|billion)',
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    # Take the first match and clean it
                    amount_str = matches[0].replace(',', '')
                    return float(amount_str)
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def create_structured_prompt(context: str, query: str, chunks_with_scores: List[tuple]) -> str:
        """Create prompt for structured decision response"""
        
        # Prepare context with clause identification
        enriched_context = ""
        for i, (chunk, score) in enumerate(chunks_with_scores):
            enriched_context += f"\n[Context {i+1} - Relevance: {score:.2f}]\n{chunk}\n"
        
        prompt = f"""You are an expert document analyzer specializing in policy, legal, and compliance decisions. 

Based on the provided context from the document, analyze the query and provide a structured decision response.

CONTEXT FROM DOCUMENT:
{enriched_context}

QUERY: {query}

INSTRUCTIONS:
1. Make a clear DECISION: Choose one of: "approved", "rejected", "pending", "requires_review", or "insufficient_information"
2. Extract any AMOUNT mentioned (monetary values, quantities, etc.) - return as number only, no currency symbols
3. Provide detailed JUSTIFICATION explaining your decision based on the document content
4. Identify RELEVANT_CLAUSES: Extract specific clause numbers, sections, or article references from the context
5. Assign CONFIDENCE_SCORE (0.0 to 1.0) based on how clear the document is about this decision

RESPONSE FORMAT - Return ONLY a valid JSON object:
{{
    "decision": "approved|rejected|pending|requires_review|insufficient_information",
    "amount": number_or_null,
    "justification": "Detailed explanation of the decision with specific references to document content",
    "relevant_clauses": ["clause1", "clause2", "section3"],
    "confidence_score": 0.95
}}

IMPORTANT:
- Base your decision ONLY on the provided context
- If information is unclear or missing, use "insufficient_information" as decision
- Include specific quotes or references from the document in justification
- Be precise about clause numbers and references
- Return ONLY the JSON object, no additional text

JSON Response:"""
        
        return prompt
    
    @staticmethod
    def generate_structured_answer(context_chunks: List[tuple], query: str) -> DecisionResponse:
        """Generate structured answer with decision, amount, and justification"""
        
        # Create cache key
        context_text = " ".join([chunk for chunk, _ in context_chunks])
        query_hash = hashlib.md5(query.encode()).hexdigest()
        context_hash = hashlib.md5(context_text.encode()).hexdigest()
        cache_key = f"{query_hash}_{context_hash}"
        
        # Check cache
        if cache_key in ANSWER_CACHE:
            return ANSWER_CACHE[cache_key]
        
        # Generate structured prompt
        prompt = StructuredAnswerGenerator.create_structured_prompt(context_text, query, context_chunks)
        
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.1,
                    top_p=0.9,
                )
            )
            
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            try:
                # Find JSON object in response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                
                # Create structured response
                structured_answer = DecisionResponse(
                    decision=parsed_response.get("decision", "insufficient_information"),
                    amount=parsed_response.get("amount"),
                    justification=parsed_response.get("justification", "Unable to parse decision from document"),
                    relevant_clauses=parsed_response.get("relevant_clauses", []),
                    confidence_score=parsed_response.get("confidence_score", 0.5)
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                # Fallback: try to extract information manually
                decision = "insufficient_information"
                if any(word in response_text.lower() for word in ["approved", "accept", "granted"]):
                    decision = "approved"
                elif any(word in response_text.lower() for word in ["rejected", "denied", "refused"]):
                    decision = "rejected"
                elif any(word in response_text.lower() for word in ["pending", "review", "further"]):
                    decision = "requires_review"
                
                # Try to extract amount
                amount = StructuredAnswerGenerator.extract_amount(response_text)
                
                # Extract clauses
                relevant_clauses = StructuredAnswerGenerator.extract_clause_references(context_text)
                
                structured_answer = DecisionResponse(
                    decision=decision,
                    amount=amount,
                    justification=response_text,
                    relevant_clauses=relevant_clauses,
                    confidence_score=0.6
                )
            
            # Cache the result
            ANSWER_CACHE[cache_key] = structured_answer
            return structured_answer
            
        except Exception as e:
            logger.error(f"Error generating structured answer: {e}")
            return DecisionResponse(
                decision="insufficient_information",
                amount=None,
                justification=f"Error generating decision: {str(e)}",
                relevant_clauses=[],
                confidence_score=0.0
            )

# Global processor
doc_processor = OptimizedDocumentProcessor()
answer_generator = StructuredAnswerGenerator()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme.lower() != "bearer" or credentials.credentials != EXPECTED_TOKEN.split(" ")[1]:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_structured_query_retrieval(request: QueryRequest):
    """Main endpoint returning structured JSON decisions"""
    start_time = time.time()
    
    try:
        # Get processed document (cached if available)
        doc_data = doc_processor.get_processed_document(request.documents)
        
        structured_answers = []
        
        # Process queries efficiently
        for query in request.questions:
            logger.info(f"Processing structured query: {query[:50]}...")
            
            # Get relevant chunks with scores
            relevant_chunks = doc_processor.semantic_search(doc_data, query, top_k=3)
            
            if not relevant_chunks:
                structured_answers.append(DecisionResponse(
                    decision="insufficient_information",
                    amount=None,
                    justification="No relevant information found in the document for this query.",
                    relevant_clauses=[],
                    confidence_score=0.0
                ))
                continue
            
            # Generate structured answer
            structured_answer = answer_generator.generate_structured_answer(relevant_chunks, query)
            structured_answers.append(structured_answer)
        
        processing_time = time.time() - start_time
        logger.info(f"Structured processing time: {processing_time:.2f} seconds")
        
        return QueryResponse(answers=structured_answers)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check with cache stats"""
    return {
        "status": "healthy",
        "cached_documents": len(DOCUMENT_CACHE),
        "cached_answers": len(ANSWER_CACHE),
        "response_format": "Structured JSON with Decision, Amount, Justification, and Clause Mapping",
        "message": "Structured JSON Query-Retrieval System is running"
    }

@app.post("/clear-cache")
async def clear_cache(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Clear all caches"""
    verify_token(credentials)
    
    with CACHE_LOCK:
        DOCUMENT_CACHE.clear()
        ANSWER_CACHE.clear()
    
    return {"message": "All caches cleared"}

@app.get("/cache-stats")
async def get_cache_stats():
    """Get cache statistics"""
    return {
        "document_cache_size": len(DOCUMENT_CACHE),
        "answer_cache_size": len(ANSWER_CACHE),
        "embedding_cache_size": len(EMBEDDING_CACHE)
    }

# Example endpoint to test structured response
@app.post("/test-structured-response")
async def test_response():
    """Test endpoint showing expected response format"""
    return {
        "example_response": {
            "answers": [
                {
                    "decision": "approved",
                    "amount": 50000.0,
                    "justification": "The claim meets all criteria specified in clause 4.2 and the amount is within the policy limit of Rs. 1,00,000 as per section 3.1.",
                    "relevant_clauses": ["4.2", "3.1", "7.5"],
                    "confidence_score": 0.95
                }
            ]
        }
    }

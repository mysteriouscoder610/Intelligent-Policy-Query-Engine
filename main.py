import os
import requests
import numpy as np
import hashlib
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import PyPDF2
from io import BytesIO
import logging
import threading
import time
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import string

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Fast Structured JSON Query-Retrieval System", version="4.0.0")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables or .env file")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model only
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Security
security = HTTPBearer()
EXPECTED_TOKEN = "Bearer 1fd4ee76a5f7d0249cf4262bd779267a6e246992896f2ee373d16e9a19254ef5"

# In-memory caches
DOCUMENT_CACHE = {}
ANSWER_CACHE = {}
CACHE_LOCK = threading.Lock()

# Pydantic models for structured responses
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

class FastSearchEngine:
    """Fast search engine using TF-IDF and keyword matching without transformers"""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunks = []
        
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase for better matching
        return text.lower()
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract important keywords from text"""
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common stopwords to filter out
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Filter out stopwords and count frequency
        filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
        word_freq = Counter(filtered_words)
        
        return [word for word, _ in word_freq.most_common(top_k)]
    
    def keyword_similarity(self, query: str, chunk: str) -> float:
        """Calculate similarity based on keyword overlap"""
        query_keywords = set(self.extract_keywords(query, 15))
        chunk_keywords = set(self.extract_keywords(chunk, 20))
        
        if not query_keywords or not chunk_keywords:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_keywords.intersection(chunk_keywords))
        union = len(query_keywords.union(chunk_keywords))
        jaccard_score = intersection / union if union > 0 else 0.0
        
        # Boost for exact phrase matches
        query_clean = self.preprocess_text(query)
        chunk_clean = self.preprocess_text(chunk)
        
        phrase_bonus = 0.0
        if query_clean in chunk_clean:
            phrase_bonus = 0.5
        elif any(word in chunk_clean for word in query_clean.split() if len(word) > 3):
            phrase_bonus = 0.2
        
        return jaccard_score + phrase_bonus
    
    def tfidf_similarity(self, query: str, chunks: List[str]) -> List[float]:
        """Calculate TF-IDF similarity scores"""
        try:
            # Combine query with chunks for vectorization
            all_texts = [query] + chunks
            
            # Create TF-IDF vectorizer with optimized parameters
            vectorizer = TfidfVectorizer(
                max_features=500,  # Limit features for speed
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                max_df=0.8,
                min_df=1
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity between query and chunks
            query_vector = tfidf_matrix[0]
            chunk_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            return similarities.tolist()
            
        except Exception as e:
            logger.warning(f"TF-IDF calculation failed: {e}, using zeros")
            return [0.0] * len(chunks)
    
    def hybrid_search(self, query: str, chunks: List[str], top_k: int = 3) -> List[tuple]:
        """Combine keyword and TF-IDF search for best results"""
        if not chunks:
            return []
        
        # Get TF-IDF scores
        tfidf_scores = self.tfidf_similarity(query, chunks)
        
        # Calculate combined scores
        combined_scores = []
        for i, chunk in enumerate(chunks):
            # Keyword similarity
            keyword_score = self.keyword_similarity(query, chunk)
            
            # TF-IDF similarity
            tfidf_score = tfidf_scores[i] if i < len(tfidf_scores) else 0.0
            
            # Combine scores (you can adjust weights)
            combined_score = 0.6 * tfidf_score + 0.4 * keyword_score
            
            # Additional boost for chunks with numbers (amounts, dates, etc.)
            if re.search(r'\d+', chunk):
                combined_score += 0.1
            
            combined_scores.append((combined_score, chunk, i))
        
        # Sort by combined score and return top results
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top chunks with scores
        results = []
        for score, chunk, idx in combined_scores[:top_k]:
            if score > 0.05:  # Minimum threshold
                results.append((chunk, score))
        
        return results

class OptimizedDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Smaller chunks for faster processing
            chunk_overlap=40,
            length_function=len,
        )
        self.search_engine = FastSearchEngine()
    
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
            max_pages = min(30, len(pdf_reader.pages))  # Limit pages for speed
            
            for i, page in enumerate(pdf_reader.pages[:max_pages]):
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    text += page_text + "\n"
                if i % 10 == 0:
                    logger.info(f"Processed page {i+1}/{max_pages}")
            
            # Clean the text
            text = re.sub(r'\s+', ' ', text.strip())
            
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
        """Get or create processed document with chunks"""
        url_hash = self.get_url_hash(url)
        
        with CACHE_LOCK:
            if url_hash in DOCUMENT_CACHE and 'chunks' in DOCUMENT_CACHE[url_hash]:
                logger.info("Using cached document chunks")
                return DOCUMENT_CACHE[url_hash]
        
        # Extract text
        text = self.extract_text_from_pdf_url(url)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Cache the chunks
        with CACHE_LOCK:
            DOCUMENT_CACHE[url_hash].update({
                'chunks': chunks,
                'processed': True
            })
        
        logger.info(f"Processed document: {len(chunks)} chunks, {len(text)} chars")
        return DOCUMENT_CACHE[url_hash]
    
    def fast_search(self, doc_data: dict, query: str, top_k: int = 3) -> List[tuple]:
        """Fast search using hybrid approach"""
        chunks = doc_data.get('chunks', [])
        if not chunks:
            return []
        
        return self.search_engine.hybrid_search(query, chunks, top_k)

class StructuredAnswerGenerator:
    @staticmethod
    def extract_clause_references(text: str) -> List[str]:
        """Extract clause references from text"""
        patterns = [
            r'(?i)clause\s+(\d+(?:\.\d+)*)',
            r'(?i)section\s+(\d+(?:\.\d+)*)',
            r'(?i)article\s+(\d+(?:\.\d+)*)',
            r'(?i)paragraph\s+(\d+(?:\.\d+)*)',
            r'(?i)subsection\s+(\d+(?:\.\d+)*)',
            r'(?i)provision\s+(\d+(?:\.\d+)*)',
            r'(?i)rule\s+(\d+(?:\.\d+)*)',
            r'\b(\d+(?:\.\d+)*)\s*(?:of|under|per|as per)',
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
        amount_patterns = [
            r'(?i)(?:rs\.?|rupees?|inr|₹)\s*([0-9,]+(?:\.[0-9]{1,2})?)',
            r'(?i)\$\s*([0-9,]+(?:\.[0-9]{1,2})?)',
            r'(?i)([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:rs\.?|rupees?|inr|₹|\$)',
            r'(?i)(?:amount|sum|value|worth|cost|price|fee|premium|limit)\s*(?:of|is|:)?\s*(?:rs\.?|rupees?|inr|₹|\$)?\s*([0-9,]+(?:\.[0-9]{1,2})?)',
            r'\b([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:lakh|crore|thousand|million|billion)',
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    amount_str = matches[0].replace(',', '')
                    amount = float(amount_str)
                    
                    # Convert lakh/crore to actual numbers if mentioned
                    if 'lakh' in text.lower():
                        amount *= 100000
                    elif 'crore' in text.lower():
                        amount *= 10000000
                    
                    return amount
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def create_structured_prompt(context: str, query: str, chunks_with_scores: List[tuple]) -> str:
        """Create prompt for structured decision response"""
        
        # Prepare context with relevance scores
        enriched_context = ""
        for i, (chunk, score) in enumerate(chunks_with_scores):
            enriched_context += f"\n[Context {i+1} - Relevance: {score:.3f}]\n{chunk}\n"
        
        prompt = f"""You are an expert document analyzer for insurance, legal, and policy decisions.

Analyze the query based on the document context and provide a structured JSON response.

CONTEXT FROM DOCUMENT:
{enriched_context}

QUERY: {query}

ANALYSIS REQUIREMENTS:
1. DECISION: Choose exactly one: "approved", "rejected", "pending", "requires_review", "insufficient_information"
2. AMOUNT: Extract any monetary value as a number (no currency symbols)
3. JUSTIFICATION: Explain your decision with specific document references
4. RELEVANT_CLAUSES: List clause/section numbers found in the context
5. CONFIDENCE_SCORE: Rate confidence 0.0-1.0 based on context clarity

OUTPUT FORMAT - Return ONLY valid JSON:
{{
    "decision": "one_of_the_five_options",
    "amount": number_or_null,
    "justification": "Detailed explanation with document quotes and references",
    "relevant_clauses": ["clause_numbers", "section_numbers"],
    "confidence_score": 0.85
}}

RULES:
- Use ONLY information from the provided context
- Quote specific text from document in justification
- If context is unclear/missing, use "insufficient_information"
- Be precise with clause numbers
- Return ONLY the JSON object

JSON Response:"""
        
        return prompt
    
    @staticmethod
    def generate_structured_answer(context_chunks: List[tuple], query: str) -> DecisionResponse:
        """Generate structured answer with decision, amount, and justification"""
        
        # Create cache key
        context_text = " ".join([chunk for chunk, _ in context_chunks])
        query_hash = hashlib.md5(query.encode()).hexdigest()
        context_hash = hashlib.md5(context_text[:500].encode()).hexdigest()  # Use first 500 chars
        cache_key = f"{query_hash}_{context_hash}"
        
        # Check cache
        if cache_key in ANSWER_CACHE:
            logger.info("Using cached structured answer")
            return ANSWER_CACHE[cache_key]
        
        # Generate structured prompt
        prompt = StructuredAnswerGenerator.create_structured_prompt(context_text, query, context_chunks)
        
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=400,
                    temperature=0.05,  # Low temperature for consistency
                    top_p=0.8,
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
                logger.warning(f"Failed to parse JSON response: {e}, falling back to manual extraction")
                
                # Fallback: manual extraction
                decision = "insufficient_information"
                response_lower = response_text.lower()
                
                if any(word in response_lower for word in ["approved", "accept", "granted", "eligible"]):
                    decision = "approved"
                elif any(word in response_lower for word in ["rejected", "denied", "refused", "not eligible"]):
                    decision = "rejected"
                elif any(word in response_lower for word in ["pending", "review", "further", "additional"]):
                    decision = "requires_review"
                
                # Extract amount and clauses
                amount = StructuredAnswerGenerator.extract_amount(response_text + " " + context_text)
                relevant_clauses = StructuredAnswerGenerator.extract_clause_references(context_text)
                
                structured_answer = DecisionResponse(
                    decision=decision,
                    amount=amount,
                    justification=response_text,
                    relevant_clauses=relevant_clauses[:5],  # Limit to 5 clauses
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
async def run_fast_structured_query_retrieval(request: QueryRequest):
    """Main endpoint returning structured JSON decisions - NO TRANSFORMERS"""
    start_time = time.time()
    
    try:
        logger.info("Processing document without sentence transformers...")
        
        # Get processed document (cached if available)
        doc_data = doc_processor.get_processed_document(request.documents)
        
        structured_answers = []
        
        # Process queries efficiently
        for query in request.questions:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Fast search using TF-IDF + keyword matching
            relevant_chunks = doc_processor.fast_search(doc_data, query, top_k=3)
            
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
        logger.info(f"Fast processing completed in: {processing_time:.2f} seconds")
        
        return QueryResponse(answers=structured_answers)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check with system info"""
    return {
        "status": "healthy",
        "search_method": "TF-IDF + Keyword Matching (No Transformers)",
        "cached_documents": len(DOCUMENT_CACHE),
        "cached_answers": len(ANSWER_CACHE),
        "response_format": "Structured JSON with Decision, Amount, Justification, and Clause Mapping",
        "message": "Fast Structured JSON Query-Retrieval System is running"
    }

@app.post("/clear-cache")
async def clear_cache(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Clear all caches"""
    verify_token(credentials)
    
    with CACHE_LOCK:
        DOCUMENT_CACHE.clear()
        ANSWER_CACHE.clear()
    
    return {"message": "All caches cleared successfully"}

@app.get("/cache-stats")
async def get_cache_stats():
    """Get detailed cache statistics"""
    with CACHE_LOCK:
        doc_cache_info = {}
        for key, value in DOCUMENT_CACHE.items():
            doc_cache_info[key] = {
                "has_text": "text" in value,
                "has_chunks": "chunks" in value,
                "chunk_count": len(value.get("chunks", [])),
                "timestamp": value.get("timestamp", 0)
            }
    
    return {
        "document_cache_size": len(DOCUMENT_CACHE),
        "answer_cache_size": len(ANSWER_CACHE),
        "document_details": doc_cache_info
    }

@app.post("/test-structured-response")
async def test_response():
    """Test endpoint showing expected response format"""
    return {
        "example_response": {
            "answers": [
                {
                    "decision": "approved",
                    "amount": 75000.0,
                    "justification": "The medical claim is covered under hospitalization benefits as per clause 5.3. The treatment amount of Rs. 75,000 is within the annual limit of Rs. 2,00,000 specified in section 3.1.",
                    "relevant_clauses": ["5.3", "3.1", "7.2"],
                    "confidence_score": 0.92
                }
            ]
        },
        "search_info": {
            "method": "TF-IDF + Keyword Matching",
            "no_transformers": True,
            "processing_speed": "~5-10x faster than transformer-based systems"
        }
    }
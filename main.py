import os
import requests
import numpy as np
import pickle
import hashlib
import asyncio
import re
import json
from typing import List, Dict, Any, Optional, Tuple
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv
from collections import defaultdict
import string

# Load environment variables from .env
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Enhanced TF-IDF with better parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=2000,  # Increased for better coverage
    stop_words='english',
    ngram_range=(1, 3),  # Include trigrams for better context
    lowercase=True,
    max_df=0.8,
    min_df=1,  # Keep rare terms that might be important
    sublinear_tf=True,  # Better scaling
    use_idf=True
)

# BM25-like scoring vectorizer
bm25_vectorizer = CountVectorizer(
    max_features=1500,
    stop_words='english',
    ngram_range=(1, 2),
    lowercase=True,
    max_df=0.8,
    min_df=1
)

# Security
security = HTTPBearer()
EXPECTED_TOKEN = os.getenv("API_ACCESS_TOKEN")
if not EXPECTED_TOKEN:
    raise RuntimeError("API_ACCESS_TOKEN not found in environment variables.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced caches
DOCUMENT_CACHE = {}
ANSWER_CACHE = {}
QUESTION_PATTERNS_CACHE = {}
CACHE_LOCK = threading.Lock()
MAX_ANSWER_CACHE = 100

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enhanced Query-Retrieval System")
    logger.info(f"TF-IDF vectorizer initialized with {tfidf_vectorizer.max_features} features")
    yield
    logger.info("Shutting down Enhanced Query-Retrieval System")

app = FastAPI(title="Enhanced Query-Retrieval System", lifespan=lifespan)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class EnhancedDocumentProcessor:
    def __init__(self):
        self.chunk_size = 800  # Larger chunks for more context
        self.chunk_overlap = 100  # More overlap for continuity
        self.question_patterns = self._load_question_patterns()
    
    def _load_question_patterns(self) -> Dict[str, List[str]]:
        """Load common question patterns for insurance/financial documents"""
        return {
            'waiting_period': [
                r'waiting period.*?(\d+).*?(months?|years?|days?)',
                r'(\d+).*?(months?|years?|days?).*?waiting period',
                r'wait.*?(\d+).*?(months?|years?|days?)',
                r'(\d+).*?(months?|years?|days?).*?before.*?(cover|claim)'
            ],
            'grace_period': [
                r'grace period.*?(\d+).*?(days?|months?)',
                r'(\d+).*?(days?|months?).*?grace period',
                r'premium.*?grace.*?(\d+).*?(days?|months?)',
                r'(\d+).*?(days?|months?).*?premium.*?grace'
            ],
            'sum_insured': [
                r'sum insured.*?(?:rs\.?|₹|inr)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakhs?|crores?|thousands?)?',
                r'coverage.*?(?:rs\.?|₹|inr)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakhs?|crores?|thousands?)?',
                r'insured amount.*?(?:rs\.?|₹|inr)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakhs?|crores?|thousands?)?'
            ],
            'premium': [
                r'premium.*?(?:rs\.?|₹|inr)?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
                r'(?:rs\.?|₹|inr)?\s*(\d+(?:,\d+)*(?:\.\d+)?).*?premium',
                r'cost.*?(?:rs\.?|₹|inr)?\s*(\d+(?:,\d+)*(?:\.\d+)?)'
            ],
            'age_limit': [
                r'age.*?(\d+).*?(?:years?|yrs?)',
                r'(\d+).*?(?:years?|yrs?).*?age',
                r'minimum.*?age.*?(\d+)',
                r'maximum.*?age.*?(\d+)'
            ]
        }
    
    def get_url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def enhanced_text_preprocessing(self, text: str) -> str:
        """Enhanced preprocessing that preserves important structure"""
        # Preserve important formatting while cleaning
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        # Keep important punctuation and numbers
        # Don't remove all special characters - keep currency symbols, percentages, etc.
        text = re.sub(r'[^\w\s.,!?%-₹$]', ' ', text)
        
        return text.strip()
    
    def smart_text_splitter(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced chunking that preserves context and adds metadata"""
        # First, split by major sections (if present)
        sections = re.split(r'\n\s*(?=[A-Z][^a-z]*[:\n])', text)
        
        chunks = []
        chunk_id = 0
        
        for section_idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Try to identify section type
            section_type = self._identify_section_type(section)
            
            # Split long sections into smaller chunks
            if len(section) <= self.chunk_size:
                chunks.append({
                    'text': section,
                    'id': chunk_id,
                    'section_type': section_type,
                    'section_idx': section_idx,
                    'keywords': self._extract_keywords(section)
                })
                chunk_id += 1
            else:
                # Smart splitting for long sections
                sentences = re.split(r'[.!?]+', section)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append({
                                'text': current_chunk.strip(),
                                'id': chunk_id,
                                'section_type': section_type,
                                'section_idx': section_idx,
                                'keywords': self._extract_keywords(current_chunk)
                            })
                            chunk_id += 1
                        
                        # Start new chunk with overlap from previous
                        overlap_words = current_chunk.split()[-10:] if current_chunk else []
                        current_chunk = " ".join(overlap_words) + " " + sentence + ". "
                
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'id': chunk_id,
                        'section_type': section_type,
                        'section_idx': section_idx,
                        'keywords': self._extract_keywords(current_chunk)
                    })
                    chunk_id += 1
        
        logger.info(f"Created {len(chunks)} enhanced chunks")
        return chunks
    
    def _identify_section_type(self, text: str) -> str:
        """Identify the type of section for better retrieval"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['benefit', 'coverage', 'cover']):
            return 'benefits'
        elif any(word in text_lower for word in ['exclusion', 'not covered', 'except']):
            return 'exclusions'
        elif any(word in text_lower for word in ['premium', 'cost', 'price', 'payment']):
            return 'premium'
        elif any(word in text_lower for word in ['claim', 'settlement', 'procedure']):
            return 'claims'
        elif any(word in text_lower for word in ['waiting', 'grace', 'period']):
            return 'periods'
        else:
            return 'general'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        text_lower = text.lower()
        keywords = []
        
        # Extract numbers with units
        number_patterns = [
            r'\d+\s*(?:days?|months?|years?|lakhs?|crores?|%|percent)',
            r'rs\.?\s*\d+(?:,\d+)*',
            r'₹\s*\d+(?:,\d+)*'
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.extend(matches)
        
        # Extract important terms
        important_terms = [
            'premium', 'sum insured', 'waiting period', 'grace period',
            'pre-existing', 'coverage', 'benefit', 'exclusion', 'claim',
            'deductible', 'copay', 'cashless', 'reimbursement'
        ]
        
        for term in important_terms:
            if term in text_lower:
                keywords.append(term)
        
        return list(set(keywords))
    
    def extract_text_from_pdf_url(self, url: str) -> str:
        """Enhanced PDF extraction with better error handling"""
        url_hash = self.get_url_hash(url)
        
        with CACHE_LOCK:
            if url_hash in DOCUMENT_CACHE:
                logger.info(f"Using cached document for URL hash: {url_hash}")
                return DOCUMENT_CACHE[url_hash]['text']
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=45, stream=True, headers=headers, verify=True)
            response.raise_for_status()
            
            pdf_content = BytesIO()
            total_size = 0
            max_size = 25 * 1024 * 1024  # 25MB limit
            
            for chunk in response.iter_content(chunk_size=8192):
                total_size += len(chunk)
                if total_size > max_size:
                    logger.warning("PDF too large, truncating...")
                    break
                pdf_content.write(chunk)
            
            pdf_content.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_content)
            text = ""
            
            # Process more pages for better coverage
            max_pages = min(100, len(pdf_reader.pages))
            
            for i, page in enumerate(pdf_reader.pages[:max_pages]):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += page_text + "\n\n"
                except Exception as e:
                    logger.warning(f"Error extracting page {i}: {e}")
                    continue
            
            text = self.enhanced_text_preprocessing(text)
            
            if not text or len(text) < 200:
                raise ValueError("Insufficient text extracted from PDF")
            
            with CACHE_LOCK:
                DOCUMENT_CACHE[url_hash] = {
                    'text': text,
                    'timestamp': time.time(),
                    'url': url
                }
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF from {url}: {e}")
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")
    
    def get_processed_document(self, url: str) -> Dict[str, Any]:
        """Enhanced document processing with multiple retrieval methods"""
        url_hash = self.get_url_hash(url)
        
        with CACHE_LOCK:
            if url_hash in DOCUMENT_CACHE and 'enhanced_chunks' in DOCUMENT_CACHE[url_hash]:
                logger.info("Using cached enhanced document")
                return DOCUMENT_CACHE[url_hash]
        
        text = self.extract_text_from_pdf_url(url)
        chunks = self.smart_text_splitter(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks extracted")
        
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        try:
            # Create TF-IDF vectors
            tfidf_matrix = tfidf_vectorizer.fit_transform(chunk_texts)
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            
            # Create BM25-like vectors
            bm25_matrix = bm25_vectorizer.fit_transform(chunk_texts)
            
            # Optional: Apply LSA for semantic understanding
            svd = TruncatedSVD(n_components=min(100, tfidf_matrix.shape[1]))
            lsa_matrix = svd.fit_transform(tfidf_matrix)
            
        except Exception as e:
            logger.error(f"Vectorization failed: {e}")
            tfidf_matrix = None
            bm25_matrix = None
            lsa_matrix = None
            svd = None
        
        with CACHE_LOCK:
            DOCUMENT_CACHE[url_hash].update({
                'enhanced_chunks': chunks,
                'chunk_texts': chunk_texts,
                'tfidf_matrix': tfidf_matrix,
                'bm25_matrix': bm25_matrix,
                'lsa_matrix': lsa_matrix,
                'svd_model': svd,
                'vectorized': True
            })
        
        logger.info(f"Enhanced processing complete: {len(chunks)} chunks")
        return DOCUMENT_CACHE[url_hash]
    
    def hybrid_search(self, doc_data: Dict[str, Any], query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Hybrid search combining multiple methods"""
        chunks = doc_data['enhanced_chunks']
        chunk_texts = doc_data['chunk_texts']
        
        if not chunks:
            return []
        
        scored_chunks = []
        
        # Method 1: Enhanced TF-IDF search
        tfidf_scores = self._tfidf_search_scores(doc_data, query)
        
        # Method 2: BM25-like search
        bm25_scores = self._bm25_search_scores(doc_data, query)
        
        # Method 3: Keyword and pattern matching
        keyword_scores = self._keyword_search_scores(chunks, query)
        
        # Method 4: Section-type matching
        section_scores = self._section_type_scores(chunks, query)
        
        # Combine scores with weights
        for i, chunk in enumerate(chunks):
            combined_score = (
                0.4 * (tfidf_scores[i] if tfidf_scores else 0) +
                0.3 * (bm25_scores[i] if bm25_scores else 0) +
                0.2 * keyword_scores[i] +
                0.1 * section_scores[i]
            )
            
            scored_chunks.append((chunk['text'], combined_score, chunk))
        
        # Sort by combined score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k with minimum threshold
        min_threshold = 0.1
        return [(text, score, metadata) for text, score, metadata in scored_chunks[:top_k] 
                if score >= min_threshold]
    
    def _tfidf_search_scores(self, doc_data: Dict[str, Any], query: str) -> List[float]:
        """Get TF-IDF similarity scores"""
        if doc_data['tfidf_matrix'] is None:
            return [0.0] * len(doc_data['enhanced_chunks'])
        
        try:
            query_vector = tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, doc_data['tfidf_matrix']).flatten()
            return similarities.tolist()
        except:
            return [0.0] * len(doc_data['enhanced_chunks'])
    
    def _bm25_search_scores(self, doc_data: Dict[str, Any], query: str) -> List[float]:
        """Get BM25-like similarity scores"""
        if doc_data['bm25_matrix'] is None:
            return [0.0] * len(doc_data['enhanced_chunks'])
        
        try:
            query_vector = bm25_vectorizer.transform([query])
            # Simple BM25-like scoring
            similarities = cosine_similarity(query_vector, doc_data['bm25_matrix']).flatten()
            return similarities.tolist()
        except:
            return [0.0] * len(doc_data['enhanced_chunks'])
    
    def _keyword_search_scores(self, chunks: List[Dict], query: str) -> List[float]:
        """Enhanced keyword matching with patterns"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores = []
        for chunk in chunks:
            chunk_text = chunk['text'].lower()
            chunk_words = set(chunk_text.split())
            chunk_keywords = set(chunk.get('keywords', []))
            
            # Basic word overlap
            word_overlap = len(query_words.intersection(chunk_words))
            total_query_words = len(query_words)
            word_score = word_overlap / total_query_words if total_query_words > 0 else 0
            
            # Keyword boost
            keyword_boost = len(query_words.intersection(chunk_keywords)) * 0.3
            
            # Pattern matching boost
            pattern_boost = self._pattern_match_score(query_lower, chunk_text)
            
            final_score = word_score + keyword_boost + pattern_boost
            scores.append(min(final_score, 1.0))  # Cap at 1.0
        
        return scores
    
    def _section_type_scores(self, chunks: List[Dict], query: str) -> List[float]:
        """Score based on section type relevance"""
        query_lower = query.lower()
        
        # Map query terms to preferred section types
        section_preferences = {
            'premium': ['premium', 'general'],
            'benefit': ['benefits', 'general'],
            'coverage': ['benefits', 'general'],
            'waiting period': ['periods', 'general'],
            'grace period': ['periods', 'general'],
            'exclusion': ['exclusions', 'general'],
            'claim': ['claims', 'general']
        }
        
        # Determine preferred section types for this query
        preferred_types = set()
        for term, types in section_preferences.items():
            if term in query_lower:
                preferred_types.update(types)
        
        if not preferred_types:
            preferred_types = {'general'}
        
        scores = []
        for chunk in chunks:
            section_type = chunk.get('section_type', 'general')
            if section_type in preferred_types:
                scores.append(0.5)  # Boost for relevant section
            else:
                scores.append(0.0)
        
        return scores
    
    def _pattern_match_score(self, query: str, text: str) -> float:
        """Advanced pattern matching for specific question types"""
        score = 0.0
        
        # Check for specific patterns based on question type
        for pattern_type, patterns in self.question_patterns.items():
            if any(keyword in query for keyword in pattern_type.split('_')):
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        score += 0.3
                        break
        
        return min(score, 0.9)  # Cap pattern boost

class SmartQueryAnswerer:
    @staticmethod
    def get_answer_cache_key(query: str, context_texts: List[str]) -> str:
        """Create cache key from query and context"""
        query_part = hashlib.md5(query.encode()).hexdigest()[:8]
        context_part = hashlib.md5(' '.join(context_texts[:2]).encode()).hexdigest()[:8]
        return f"{query_part}_{context_part}"
    
    @staticmethod
    def create_enhanced_prompt(context_chunks: List[Tuple[str, float, Dict]], query: str) -> str:
        """Create a more sophisticated prompt with structured context"""
        
        # Sort chunks by score and take top ones
        sorted_chunks = sorted(context_chunks, key=lambda x: x[1], reverse=True)[:4]
        
        context_parts = []
        for i, (text, score, metadata) in enumerate(sorted_chunks):
            section_info = f"[Section: {metadata.get('section_type', 'general')}]"
            context_parts.append(f"Context {i+1} {section_info}:\n{text}")
        
        combined_context = "\n\n".join(context_parts)
        
        # Limit context to avoid token limits but keep it substantial
        if len(combined_context) > 3000:
            combined_context = combined_context[:3000] + "..."
        
        return f"""You are analyzing an insurance/financial document. Use the provided contexts to answer the question accurately and specifically.

{combined_context}

Question: {query}

Instructions:
1. Answer based ONLY on the provided contexts
2. Be specific with numbers, periods, amounts when mentioned
3. If information is not in the contexts, say "Information not found in the provided document"
4. Keep the answer concise but complete
5. Include relevant details like amounts, time periods, conditions

Answer:"""
    
    @staticmethod
    def extract_structured_answer(query: str, contexts: List[str]) -> Optional[str]:
        """Extract answers using rule-based extraction for common patterns"""
        query_lower = query.lower()
        combined_text = " ".join(contexts).lower()
        
        # Grace period for premium
        if 'grace period' in query_lower and 'premium' in query_lower:
            patterns = [
                r'grace period.*?(?:of\s+)?(\d+)\s*(days?|months?)',
                r'(\d+)\s*(days?|months?).*?grace period.*?premium',
                r'premium.*?grace period.*?(\d+)\s*(days?|months?)',
                r'(?:within|after)\s+(\d+)\s*(days?|months?).*?premium'
            ]
            for pattern in patterns:
                match = re.search(pattern, combined_text)
                if match:
                    number = match.group(1)
                    unit = match.group(2)
                    return f"The grace period for premium payment is {number} {unit}."
        
        # Waiting period for pre-existing diseases
        if 'waiting period' in query_lower and ('pre-existing' in query_lower or 'ped' in query_lower):
            patterns = [
                r'pre-existing.*?waiting period.*?(\d+)\s*(years?|months?)',
                r'waiting period.*?(\d+)\s*(years?|months?).*?pre-existing',
                r'ped.*?waiting period.*?(\d+)\s*(years?|months?)',
                r'(\d+)\s*(years?|months?).*?waiting.*?pre-existing'
            ]
            for pattern in patterns:
                match = re.search(pattern, combined_text)
                if match:
                    number = match.group(1)
                    unit = match.group(2)
                    return f"The waiting period for pre-existing diseases is {number} {unit}."
        
        # Sum insured
        if 'sum insured' in query_lower or 'coverage amount' in query_lower:
            patterns = [
                r'sum insured.*?(?:rs\.?|₹|inr)?\s*(\d+(?:,\d+)*)\s*(lakhs?|crores?)?',
                r'coverage.*?(?:rs\.?|₹|inr)?\s*(\d+(?:,\d+)*)\s*(lakhs?|crores?)?',
                r'insured amount.*?(?:rs\.?|₹|inr)?\s*(\d+(?:,\d+)*)\s*(lakhs?|crores?)?'
            ]
            for pattern in patterns:
                match = re.search(pattern, combined_text)
                if match:
                    amount = match.group(1)
                    unit = match.group(2) if match.group(2) else ""
                    return f"The sum insured is Rs. {amount} {unit}".strip() + "."
        
        return None
    
    @staticmethod
    def generate_enhanced_answer(context_chunks: List[Tuple[str, float, Dict]], query: str) -> str:
        """Generate answer using multiple methods"""
        if not context_chunks:
            return "No relevant information found in the document."
        
        # Extract context texts
        context_texts = [chunk[0] for chunk in context_chunks]
        
        # Check cache first
        cache_key = SmartQueryAnswerer.get_answer_cache_key(query, context_texts)
        if cache_key in ANSWER_CACHE:
            logger.info("Using cached answer")
            return ANSWER_CACHE[cache_key]
        
        # Try rule-based extraction first
        structured_answer = SmartQueryAnswerer.extract_structured_answer(query, context_texts)
        if structured_answer:
            ANSWER_CACHE[cache_key] = structured_answer
            return structured_answer
        
        # Fallback to LLM
        prompt = SmartQueryAnswerer.create_enhanced_prompt(context_chunks, query)
        
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,  # Slightly more for better answers
                    temperature=0.1,       # Very low for consistency
                    top_p=0.8,
                )
            )
            
            answer = response.text.strip()
            
            # Cache the answer
            if len(ANSWER_CACHE) >= MAX_ANSWER_CACHE:
                # Remove oldest entries
                oldest_keys = list(ANSWER_CACHE.keys())[:20]
                for key in oldest_keys:
                    ANSWER_CACHE.pop(key, None)
            
            ANSWER_CACHE[cache_key] = answer
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Sorry, couldn't generate an answer due to an error."

# Global processor
doc_processor = EnhancedDocumentProcessor()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    
    expected_token = EXPECTED_TOKEN
    if expected_token.startswith("Bearer "):
        expected_token = expected_token.split(" ")[1]
    
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return credentials

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_enhanced_query_retrieval(request: QueryRequest):
    """Enhanced main endpoint with better accuracy"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        if not request.documents or not request.questions:
            raise HTTPException(status_code=400, detail="Documents URL and questions are required")
        
        # Get processed document
        doc_data = doc_processor.get_processed_document(request.documents)
        
        answers = []
        
        for i, query in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {query[:60]}...")
            
            try:
                # Use hybrid search for better retrieval
                relevant_chunks = doc_processor.hybrid_search(doc_data, query, top_k=5)
                
                if not relevant_chunks:
                    answers.append("No relevant information found in the document.")
                    continue
                
                # Generate answer using enhanced method
                answer = SmartQueryAnswerer.generate_enhanced_answer(relevant_chunks, query)
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                answers.append("Error processing this question.")
        
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        logger.info(f"Generated {len(answers)} answers")
        
        return QueryResponse(answers=answers)
    except Exception as e:
        logger.error(f"Unexpected error in query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cache_size": len(DOCUMENT_CACHE),
        "answer_cache_size": len(ANSWER_CACHE),
        "timestamp": time.time()
    }

@app.get("/cache/stats")
async def cache_stats(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get cache statistics - requires authentication"""
    verify_token(credentials)
    
    with CACHE_LOCK:
        doc_cache_info = {}
        for url_hash, data in DOCUMENT_CACHE.items():
            doc_cache_info[url_hash] = {
                "url": data.get("url", "unknown"),
                "text_length": len(data.get("text", "")),
                "chunks_count": len(data.get("enhanced_chunks", [])),
                "timestamp": data.get("timestamp", 0),
                "vectorized": data.get("vectorized", False)
            }
    
    return {
        "document_cache": doc_cache_info,
        "answer_cache_size": len(ANSWER_CACHE),
        "total_cached_documents": len(DOCUMENT_CACHE)
    }

@app.delete("/cache/clear")
async def clear_cache(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Clear all caches - requires authentication"""
    verify_token(credentials)
    
    with CACHE_LOCK:
        DOCUMENT_CACHE.clear()
        ANSWER_CACHE.clear()
    
    logger.info("All caches cleared")
    return {"message": "All caches cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn server
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False,  # Set to True for development
        workers=1      # Single worker due to in-memory caching
    )
    
    server = uvicorn.Server(config)
    
    try:
        logger.info("Starting Enhanced Query-Retrieval System server...")
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
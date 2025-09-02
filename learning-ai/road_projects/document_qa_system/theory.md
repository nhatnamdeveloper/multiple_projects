# 📄 Document Q&A System - Lý thuyết

> **Mục tiêu**: Xây dựng hệ thống hỏi đáp tài liệu thông minh với RAG (Retrieval-Augmented Generation)

## 🧠 **Lý thuyết cơ bản**

### **1. Document Q&A System Overview**

**Khái niệm cốt lõi:**
- **Retrieval-Augmented Generation (RAG)**: Kết hợp retrieval và generation
- **Document Processing**: Xử lý và chunking tài liệu
- **Vector Search**: Tìm kiếm semantic similarity
- **Question Answering**: Trả lời câu hỏi dựa trên context

### **2. RAG Architecture Components**

**A. Document Processing:**
- **Text Extraction**: OCR, PDF parsing, document parsing
- **Chunking**: Chia tài liệu thành chunks nhỏ
- **Embedding**: Chuyển đổi text thành vectors
- **Indexing**: Lưu trữ vectors trong vector database

**B. Query Processing:**
- **Question Understanding**: Phân tích câu hỏi
- **Query Embedding**: Chuyển đổi câu hỏi thành vector
- **Retrieval**: Tìm kiếm chunks liên quan
- **Reranking**: Sắp xếp lại kết quả theo relevance

**C. Answer Generation:**
- **Context Assembly**: Ghép context từ chunks
- **Prompt Engineering**: Tạo prompt cho LLM
- **Answer Generation**: Sinh câu trả lời
- **Answer Validation**: Kiểm tra tính chính xác

### **3. Key Technologies**

**A. Embedding Models:**
- **Sentence Transformers**: BERT-based sentence embeddings
- **OpenAI Embeddings**: text-embedding-ada-002
- **Cohere Embeddings**: Multilingual embeddings
- **Custom Fine-tuned**: Domain-specific embeddings

**B. Vector Databases:**
- **Pinecone**: Managed vector database
- **Weaviate**: Open-source vector database
- **Qdrant**: High-performance vector search
- **FAISS**: Facebook AI Similarity Search

**C. Language Models:**
- **GPT Models**: OpenAI's GPT series
- **Claude**: Anthropic's conversational AI
- **Llama**: Meta's open-source LLM
- **Local Models**: LlamaCpp, GPT4All

## 🔧 **Technical Architecture**

### **1. Document Q&A System Architecture**

```python
class DocumentQAArchitecture:
    """Architecture cho Document Q&A System"""
    
    def __init__(self):
        self.components = {
            'document_processing': ['Text Extraction', 'Chunking', 'Embedding', 'Indexing'],
            'query_processing': ['Question Analysis', 'Query Embedding', 'Retrieval', 'Reranking'],
            'answer_generation': ['Context Assembly', 'Prompt Engineering', 'LLM Generation'],
            'knowledge_base': ['Vector Database', 'Document Store', 'Metadata Management'],
            'api_layer': ['REST API', 'WebSocket', 'Streaming Responses']
        }
    
    def explain_data_flow(self):
        """Explain data flow trong hệ thống"""
        print("""
        **Document Q&A System Data Flow:**
        
        1. **Document Ingestion Layer:**
           - Document upload (PDF, DOCX, TXT, etc.)
           - Text extraction và preprocessing
           - Document chunking (sliding window, semantic)
           - Metadata extraction (title, author, date)
        
        2. **Embedding & Indexing Layer:**
           - Text embedding generation
           - Vector storage trong database
           - Index creation cho fast retrieval
           - Metadata indexing
        
        3. **Query Processing Layer:**
           - Question preprocessing và analysis
           - Query embedding generation
           - Vector similarity search
           - Result reranking và filtering
        
        4. **Answer Generation Layer:**
           - Context assembly từ retrieved chunks
           - Prompt engineering cho LLM
           - Answer generation với citations
           - Answer validation và confidence scoring
        
        5. **Response Layer:**
           - Answer formatting và presentation
           - Source citation và references
           - Confidence scores và explanations
           - Follow-up question suggestions
        """)
```

### **2. Document Processing Implementation**

**Document Chunker:**
```python
import re
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib

@dataclass
class DocumentChunk:
    """Represents a chunk of document"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_char: int
    end_char: int
    embedding: List[float] = None

class DocumentProcessor:
    """Document processing and chunking"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            
            doc = Document(docx_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text
        except Exception as e:
            raise ValueError(f"Error extracting text from DOCX: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}]', '', text)
        
        # Normalize unicode
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text.strip()
    
    def create_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Create overlapping chunks from text"""
        chunks = []
        
        # Clean text
        text = self.clean_text(text)
        
        # Create chunks with overlap
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Adjust end to not break words
            if end < len(text):
                # Find last space before end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                # Create chunk ID
                chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()[:8]
                chunk_id_str = f"chunk_{chunk_id}_{chunk_hash}"
                
                # Create chunk
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata=metadata or {},
                    chunk_id=chunk_id_str,
                    start_char=start,
                    end_char=end
                )
                
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            chunk_id += 1
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def create_semantic_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Create semantic chunks based on paragraphs and sections"""
        chunks = []
        
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_start = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk_hash = hashlib.md5(current_chunk.encode()).hexdigest()[:8]
                chunk_id_str = f"semantic_chunk_{chunk_id}_{chunk_hash}"
                
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    metadata=metadata or {},
                    chunk_id=chunk_id_str,
                    start_char=chunk_start,
                    end_char=chunk_start + len(current_chunk)
                )
                
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = paragraph
                chunk_start = chunk_start + len(current_chunk)
                chunk_id += 1
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunk_hash = hashlib.md5(current_chunk.encode()).hexdigest()[:8]
            chunk_id_str = f"semantic_chunk_{chunk_id}_{chunk_hash}"
            
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata=metadata or {},
                chunk_id=chunk_id_str,
                start_char=chunk_start,
                end_char=chunk_start + len(current_chunk)
            )
            
            chunks.append(chunk)
        
        return chunks
```

### **3. Vector Search Implementation**

**FAISS Vector Search:**
```python
import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
import pickle

class VectorSearchEngine:
    """Vector search engine using FAISS"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', dimension=384):
        self.model_name = model_name
        self.dimension = dimension
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.metadata = []
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create embeddings for document chunks"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, chunks: List[DocumentChunk], index_type='IVFFlat'):
        """Build FAISS index from chunks"""
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Store chunks and metadata
        self.chunks = chunks
        self.metadata = [chunk.metadata for chunk in chunks]
        
        # Create FAISS index
        if index_type == 'IVFFlat':
            # IVF index for large datasets
            nlist = min(100, len(chunks) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train index
            self.index.train(embeddings)
        else:
            # Simple flat index for small datasets
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        print(f"Built index with {len(chunks)} chunks")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 10  # For IVF index
        
        scores, indices = self.index.search(query_embedding, k)
        
        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append((chunk, float(score)))
        
        return results
    
    def search_with_filter(self, query: str, k: int = 5, filter_func=None) -> List[Tuple[DocumentChunk, float]]:
        """Search with custom filtering"""
        # Get more results for filtering
        all_results = self.search(query, k * 3)
        
        # Apply filter
        if filter_func:
            filtered_results = []
            for chunk, score in all_results:
                if filter_func(chunk):
                    filtered_results.append((chunk, score))
                    if len(filtered_results) >= k:
                        break
            return filtered_results
        
        return all_results[:k]
    
    def save_index(self, filepath: str):
        """Save index to file"""
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save chunks and metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata,
                'model_name': self.model_name
            }, f)
    
    def load_index(self, filepath: str):
        """Load index from file"""
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load chunks and metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            self.model_name = data['model_name']
```

### **4. RAG Answer Generation**

**RAG Pipeline:**
```python
from typing import List, Dict, Any, Optional
import openai
from dataclasses import dataclass

@dataclass
class AnswerResult:
    """Result of answer generation"""
    answer: str
    sources: List[DocumentChunk]
    confidence: float
    reasoning: str
    citations: List[Dict[str, Any]]

class RAGPipeline:
    """RAG pipeline for question answering"""
    
    def __init__(self, vector_search: VectorSearchEngine, llm_client=None):
        self.vector_search = vector_search
        self.llm_client = llm_client or openai.OpenAI()
    
    def generate_answer(self, question: str, max_chunks: int = 5) -> AnswerResult:
        """Generate answer using RAG pipeline"""
        try:
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.vector_search.search(question, max_chunks)
            
            if not retrieved_chunks:
                return AnswerResult(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    confidence=0.0,
                    reasoning="No relevant documents found",
                    citations=[]
                )
            
            # Step 2: Assemble context
            context = self._assemble_context(retrieved_chunks)
            
            # Step 3: Generate answer
            answer, reasoning = self._generate_answer_with_context(question, context)
            
            # Step 4: Create citations
            citations = self._create_citations(retrieved_chunks)
            
            # Step 5: Calculate confidence
            confidence = self._calculate_confidence(retrieved_chunks, answer)
            
            return AnswerResult(
                answer=answer,
                sources=[chunk for chunk, _ in retrieved_chunks],
                confidence=confidence,
                reasoning=reasoning,
                citations=citations
            )
            
        except Exception as e:
            return AnswerResult(
                answer=f"Error generating answer: {str(e)}",
                sources=[],
                confidence=0.0,
                reasoning="Error occurred during processing",
                citations=[]
            )
    
    def _assemble_context(self, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Assemble context from retrieved chunks"""
        context_parts = []
        
        for i, (chunk, score) in enumerate(retrieved_chunks):
            context_parts.append(f"Document {i+1} (Relevance: {score:.3f}):\n{chunk.content}\n")
        
        return "\n".join(context_parts)
    
    def _generate_answer_with_context(self, question: str, context: str) -> Tuple[str, str]:
        """Generate answer using LLM with context"""
        prompt = f"""
        Based on the following context, please answer the question. 
        If the answer cannot be found in the context, say "I cannot answer this question based on the provided information."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Generate reasoning
            reasoning = f"Generated answer using {len(context.split())} words of context from {len(context.split('Document'))} documents."
            
            return answer, reasoning
            
        except Exception as e:
            return f"Error generating answer: {str(e)}", "LLM error occurred"
    
    def _create_citations(self, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> List[Dict[str, Any]]:
        """Create citations for sources"""
        citations = []
        
        for i, (chunk, score) in enumerate(retrieved_chunks):
            citation = {
                'id': chunk.chunk_id,
                'content': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                'relevance_score': score,
                'metadata': chunk.metadata,
                'position': i + 1
            }
            citations.append(citation)
        
        return citations
    
    def _calculate_confidence(self, retrieved_chunks: List[Tuple[DocumentChunk, float]], answer: str) -> float:
        """Calculate confidence score for answer"""
        if not retrieved_chunks:
            return 0.0
        
        # Average relevance score
        avg_relevance = sum(score for _, score in retrieved_chunks) / len(retrieved_chunks)
        
        # Answer length factor (longer answers might be more confident)
        length_factor = min(len(answer.split()) / 50, 1.0)
        
        # Number of sources factor
        source_factor = min(len(retrieved_chunks) / 5, 1.0)
        
        # Combine factors
        confidence = (avg_relevance * 0.6 + length_factor * 0.2 + source_factor * 0.2)
        
        return min(confidence, 1.0)
    
    def generate_follow_up_questions(self, question: str, answer: str, context: str) -> List[str]:
        """Generate follow-up questions"""
        prompt = f"""
        Based on the original question, answer, and context, generate 3 relevant follow-up questions.
        
        Original Question: {question}
        Answer: {answer}
        Context: {context[:500]}...
        
        Generate 3 follow-up questions:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates relevant follow-up questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            questions_text = response.choices[0].message.content.strip()
            
            # Parse questions (assuming they're numbered or separated by newlines)
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('Original Question:') and not line.startswith('Answer:'):
                    # Remove numbering if present
                    if line[0].isdigit() and line[1] in ['.', ')']:
                        line = line[2:].strip()
                    questions.append(line)
            
            return questions[:3]  # Return max 3 questions
            
        except Exception as e:
            return [f"Error generating follow-up questions: {str(e)}"]
```

## 📊 **Evaluation Framework**

### **1. RAG Evaluation Metrics**

**Evaluation Implementation:**
```python
class RAGEvaluator:
    """Evaluation framework for RAG systems"""
    
    def __init__(self, test_questions: List[Dict[str, Any]]):
        self.test_questions = test_questions
    
    def evaluate_retrieval(self, rag_pipeline: RAGPipeline) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        total_precision = 0
        total_recall = 0
        total_ndcg = 0
        
        for test_case in self.test_questions:
            question = test_case['question']
            relevant_chunks = set(test_case['relevant_chunk_ids'])
            
            # Get retrieved chunks
            retrieved_chunks = rag_pipeline.vector_search.search(question, k=10)
            retrieved_ids = set(chunk.chunk_id for chunk, _ in retrieved_chunks)
            
            # Calculate metrics
            precision = len(relevant_chunks & retrieved_ids) / len(retrieved_ids) if retrieved_ids else 0
            recall = len(relevant_chunks & retrieved_ids) / len(relevant_chunks) if relevant_chunks else 0
            
            total_precision += precision
            total_recall += recall
        
        avg_precision = total_precision / len(self.test_questions)
        avg_recall = total_recall / len(self.test_questions)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score
        }
    
    def evaluate_answer_quality(self, rag_pipeline: RAGPipeline) -> Dict[str, float]:
        """Evaluate answer quality"""
        total_accuracy = 0
        total_relevance = 0
        total_completeness = 0
        
        for test_case in self.test_questions:
            question = test_case['question']
            expected_answer = test_case['expected_answer']
            
            # Generate answer
            result = rag_pipeline.generate_answer(question)
            
            # Calculate metrics (simplified)
            accuracy = self._calculate_answer_accuracy(result.answer, expected_answer)
            relevance = self._calculate_answer_relevance(result.answer, question)
            completeness = self._calculate_answer_completeness(result.answer, expected_answer)
            
            total_accuracy += accuracy
            total_relevance += relevance
            total_completeness += completeness
        
        return {
            'accuracy': total_accuracy / len(self.test_questions),
            'relevance': total_relevance / len(self.test_questions),
            'completeness': total_completeness / len(self.test_questions)
        }
    
    def _calculate_answer_accuracy(self, generated_answer: str, expected_answer: str) -> float:
        """Calculate answer accuracy (simplified)"""
        # This is a simplified implementation
        # In practice, you might use more sophisticated metrics
        generated_words = set(generated_answer.lower().split())
        expected_words = set(expected_answer.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(generated_words & expected_words)
        return overlap / len(expected_words)
    
    def _calculate_answer_relevance(self, answer: str, question: str) -> float:
        """Calculate answer relevance to question"""
        # Simplified relevance calculation
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words & answer_words)
        return min(overlap / len(question_words), 1.0)
    
    def _calculate_answer_completeness(self, answer: str, expected_answer: str) -> float:
        """Calculate answer completeness"""
        # Simplified completeness calculation
        if not expected_answer:
            return 0.0
        
        expected_length = len(expected_answer.split())
        actual_length = len(answer.split())
        
        if expected_length == 0:
            return 0.0
        
        return min(actual_length / expected_length, 1.0)
```

## 🎯 **Business Impact**

### **Expected Outcomes:**
- **Knowledge Discovery**: 90% faster information retrieval
- **Accuracy Improvement**: 85%+ accuracy in answer generation
- **User Productivity**: 70% reduction in time spent searching documents
- **Knowledge Sharing**: Centralized knowledge base accessible to all
- **Cost Reduction**: 60% reduction in support and training costs

---

**📚 References:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Lewis et al.
- "Dense Passage Retrieval for Open-Domain Question Answering" by Karpukhin et al.
- "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" by Reimers and Gurevych
- "FAISS: A Library for Efficient Similarity Search" by Johnson et al.
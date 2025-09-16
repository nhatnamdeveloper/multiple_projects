# 🔧 Setup Guide - AI Chatbot với RAG

> **Mục tiêu**: Hướng dẫn cài đặt và cấu hình môi trường phát triển cho dự án RAG Chatbot

## 📚 **1. Bảng ký hiệu (Notation)**

### **System Requirements:**
- **OS**: Ubuntu 20.04+ / macOS 10.15+ / Windows 10+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.9+
- **Node.js**: 16+ (for frontend)

### **Environment Variables:**
- **API Keys**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- **Database**: `VECTOR_DB_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`
- **Server**: `HOST`, `PORT`, `DEBUG`

### **File Paths:**
- **Project root**: `./rag_chatbot/`
- **Backend**: `./rag_chatbot/backend/`
- **Frontend**: `./rag_chatbot/frontend/`
- **Data**: `./rag_chatbot/data/`

## 📖 **2. Glossary (Định nghĩa cốt lõi)**

### **Development Environment:**
- **Virtual Environment**: Isolated Python environment
- **Dependencies**: External libraries required by project
- **Package Manager**: Tool to install dependencies (pip, npm)
- **Version Control**: System to track code changes (Git)

### **System Components:**
- **Backend**: Server-side application (FastAPI)
- **Frontend**: Client-side application (React)
- **Database**: Storage system (FAISS, ChromaDB)
- **API**: Interface for communication between components

### **Deployment Terms:**
- **Docker**: Containerization platform
- **Kubernetes**: Container orchestration
- **CI/CD**: Continuous Integration/Deployment
- **Production**: Live environment for users

## 📋 **Yêu cầu hệ thống**

### **Minimum Requirements:**
- **OS**: Ubuntu 20.04+ / macOS 10.15+ / Windows 10+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.9+
- **Node.js**: 16+ (for frontend)

### **Recommended Setup:**
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU với CUDA support (optional)
- **Storage**: SSD với 20GB+ free space

## 🐍 **3. Thẻ thuật toán - Python Environment Setup**

### **1. Bài toán & dữ liệu:**
- **Bài toán**: Tạo môi trường Python isolated cho development
- **Dữ liệu**: Python interpreter, project dependencies
- **Ứng dụng**: Development, testing, production deployment

### **2. Mô hình & công thức:**
**Virtual Environment Creation:**
```bash
python -m venv rag_env
```

**Environment Activation:**
```bash
# Linux/macOS:
source rag_env/bin/activate
# Windows:
rag_env\Scripts\activate
```

### **3. Loss & mục tiêu:**
- **Mục tiêu**: Tạo isolated environment để tránh conflicts
- **Loss**: Không có loss, là setup step

### **4. Tối ưu hoá & cập nhật:**
- **Algorithm**: Create virtual environment
- **Cập nhật**: Activate environment khi cần

### **5. Hyperparams:**
- **Python version**: 3.9+
- **Environment name**: rag_env
- **Path**: ./rag_env/

### **6. Độ phức tạp:**
- **Time**: $O(1)$ cho creation
- **Space**: $O(\text{dependencies})$ cho storage

### **7. Metrics đánh giá:**
- **Environment isolation**: Không conflict với system Python
- **Dependency management**: Clean install/uninstall
- **Reproducibility**: Same environment across machines

### **8. Ưu / Nhược:**
**Ưu điểm:**
- Isolated dependencies
- Easy to reproduce
- Clean uninstall

**Nhược điểm:**
- Additional setup step
- Memory overhead
- Need to remember activation

### **9. Bẫy & mẹo:**
- **Bẫy**: Quên activate environment → install globally
- **Bẫy**: Không add to .gitignore → commit large files
- **Mẹo**: Use virtualenvwrapper for easier management
- **Mẹo**: Add environment to .gitignore

### **10. Pseudocode:**
```bash
# Create virtual environment
python -m venv rag_env

# Activate environment
source rag_env/bin/activate  # Linux/macOS
# rag_env\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### **11. Code mẫu:**
```bash
# Tạo virtual environment
python -m venv rag_env

# Activate environment
# Linux/macOS:
source rag_env/bin/activate
# Windows:
rag_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### **12. Checklist kiểm tra nhanh:**
- [ ] Virtual environment có được tạo?
- [ ] Environment có được activate?
- [ ] Python version có đúng?
- [ ] Pip có được upgrade?
- [ ] Environment có isolated?

---

## 🗄️ **4. Thẻ thuật toán - Vector Database Setup**

### **1. Bài toán & dữ liệu:**
- **Bài toán**: Cài đặt và cấu hình vector database cho RAG
- **Dữ liệu**: FAISS, ChromaDB, embedding vectors
- **Ứng dụng**: Vector similarity search, RAG systems

### **2. Mô hình & công thức:**
**FAISS Installation:**
```bash
pip install faiss-cpu  # CPU version
pip install faiss-gpu  # GPU version (if available)
```

**ChromaDB Setup:**
```python
import chromadb
client = chromadb.Client(Settings(persist_directory="./chroma_db"))
```

### **3. Loss & mục tiêu:**
- **Mục tiêu**: Setup vector database cho fast similarity search
- **Loss**: Không có loss, là infrastructure setup

### **4. Tối ưu hoá & cập nhật:**
- **Algorithm**: Install and configure database
- **Cập nhật**: Không có parameter learning

### **5. Hyperparams:**
- **Index type**: FlatIP, HNSW, IVF
- **Dimension**: 384, 768, 1536
- **Persist directory**: ./vector_db/

### **6. Độ phức tạp:**
- **Time**: $O(n \times d)$ cho exact search
- **Space**: $O(n \times d)$ cho storing vectors

### **7. Metrics đánh giá:**
- **Installation success**: Package installed correctly
- **Performance**: Search speed and accuracy
- **Memory usage**: Storage efficiency

### **8. Ưu / Nhược:**
**Ưu điểm:**
- Fast similarity search
- Scalable với large datasets
- Multiple index types

**Nhược điểm:**
- Complex setup với GPU
- Memory intensive
- Learning curve

### **9. Bẫy & mẹo:**
- **Bẫy**: GPU version không compatible → use CPU
- **Bẫy**: Memory issues với large datasets
- **Mẹo**: Start với CPU version
- **Mẹo**: Use appropriate index type

### **10. Pseudocode:**
```python
# FAISS Setup
import faiss
import numpy as np

def test_faiss_installation():
    # Create sample data
    dimension = 384
    num_vectors = 1000
    vectors = np.random.randn(num_vectors, dimension).astype('float32')
    
    # Normalize vectors
    faiss.normalize_L2(vectors)
    
    # Create index
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    
    # Test search
    query = np.random.randn(1, dimension).astype('float32')
    faiss.normalize_L2(query)
    scores, indices = index.search(query, 5)
    
    return len(indices[0]) > 0
```

### **11. Code mẫu:**
```python
# test_faiss.py
import faiss
import numpy as np

def test_faiss_installation():
    """Test FAISS installation"""
    # Create sample data
    dimension = 384
    num_vectors = 1000
    
    # Generate random vectors
    vectors = np.random.randn(num_vectors, dimension).astype('float32')
    
    # Normalize vectors
    faiss.normalize_L2(vectors)
    
    # Create index
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    
    # Test search
    query = np.random.randn(1, dimension).astype('float32')
    faiss.normalize_L2(query)
    
    scores, indices = index.search(query, 5)
    
    print("FAISS test successful!")
    print(f"Found {len(indices[0])} similar vectors")
    
    return True

if __name__ == "__main__":
    test_faiss_installation()
```

### **12. Checklist kiểm tra nhanh:**
- [ ] FAISS có được install?
- [ ] Test script có chạy thành công?
- [ ] Memory usage có acceptable?
- [ ] Search có fast enough?
- [ ] Index có được tạo?

---

## 🌐 **5. Thẻ thuật toán - Frontend Setup**

### **1. Bài toán & dữ liệu:**
- **Bài toán**: Setup React frontend cho RAG chatbot
- **Dữ liệu**: Node.js, React, dependencies
- **Ứng dụng**: Web interface, real-time chat

### **2. Mô hình & công thức:**
**React App Creation:**
```bash
npx create-react-app rag-frontend
```

**Dependencies Installation:**
```bash
npm install axios react-router-dom @mui/material
```

### **3. Loss & mục tiêu:**
- **Mục tiêu**: Tạo responsive web interface
- **Loss**: Không có loss, là UI development

### **4. Tối ưu hoá & cập nhật:**
- **Algorithm**: Create React app and install dependencies
- **Cập nhật**: Hot reload during development

### **5. Hyperparams:**
- **Node.js version**: 16+
- **React version**: Latest stable
- **Port**: 3000 (default)

### **6. Độ phức tạp:**
- **Time**: $O(\text{dependencies})$ cho installation
- **Space**: $O(\text{node_modules})$ cho storage

### **7. Metrics đánh giá:**
- **Build success**: App compiles without errors
- **Performance**: Load time and responsiveness
- **User experience**: Intuitive interface

### **8. Ưu / Nhược:**
**Ưu điểm:**
- Fast development với hot reload
- Rich ecosystem
- Good performance

**Nhược điểm:**
- Large bundle size
- Complex setup
- Learning curve

### **9. Bẫy & mẹo:**
- **Bẫy**: Node.js version incompatible
- **Bẫy**: Dependencies conflicts
- **Mẹo**: Use nvm for Node.js version management
- **Mẹo**: Check compatibility matrix

### **10. Pseudocode:**
```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Create React app
npx create-react-app rag-frontend
cd rag-frontend

# Install dependencies
npm install axios react-router-dom @mui/material

# Start development server
npm start
```

### **11. Code mẫu:**
```bash
# Install Node.js (nếu chưa có)
# Ubuntu/Debian:
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS:
brew install node

# Windows: Download từ https://nodejs.org/

# Verify installation
node --version
npm --version

# Tạo React app
npx create-react-app rag-frontend
cd rag-frontend

# Install dependencies
npm install axios react-router-dom
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material

# Development dependencies
npm install --save-dev @types/react @types/react-dom
```

### **12. Checklist kiểm tra nhanh:**
- [ ] Node.js có được install?
- [ ] React app có được tạo?
- [ ] Dependencies có được install?
- [ ] Development server có chạy?
- [ ] Browser có hiển thị app?

---

## 🔧 **Development Tools Setup**

### **1. Code Quality Tools**

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
EOF
```

### **2. Testing Setup**

```bash
# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov
pip install httpx  # for FastAPI testing

# Create pytest.ini
cat > pytest.ini << EOF
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=app --cov-report=html
EOF
```

### **3. Docker Setup**

```bash
# Install Docker
# Ubuntu:
sudo apt-get update
sudo apt-get install docker.io docker-compose

# macOS: Download Docker Desktop
# Windows: Download Docker Desktop

# Verify installation
docker --version
docker-compose --version
```

## 📁 **Project Structure**

```bash
# Tạo cấu trúc thư mục
mkdir -p rag_chatbot/{backend,frontend,tests,docs,data}

# Backend structure
mkdir -p rag_chatbot/backend/{app,models,services,utils}
mkdir -p rag_chatbot/backend/app/{api,core,db}

# Frontend structure
mkdir -p rag_chatbot/frontend/src/{components,pages,services,utils}

# Data directories
mkdir -p rag_chatbot/data/{documents,embeddings,vector_db}

# Create necessary files
touch rag_chatbot/backend/requirements.txt
touch rag_chatbot/backend/main.py
touch rag_chatbot/frontend/package.json
touch rag_chatbot/docker-compose.yml
touch rag_chatbot/README.md
```

## 🚀 **Quick Start**

### **1. Backend Setup**

```bash
cd rag_chatbot/backend

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_key_here"

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **2. Frontend Setup**

```bash
cd rag_chatbot/frontend

# Install dependencies
npm install

# Start development server
npm start
```

### **3. Test Installation**

```bash
# Test backend
cd backend
python -m pytest tests/ -v

# Test frontend
cd frontend
npm test

# Test API
curl http://localhost:8000/health
```

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **FAISS Installation Error:**
```bash
# Ubuntu/Debian
sudo apt-get install libblas-dev liblapack-dev
pip install faiss-cpu --no-cache-dir

# macOS
brew install openblas
export LDFLAGS="-L/usr/local/opt/openblas/lib"
export CPPFLAGS="-I/usr/local/opt/openblas/include"
pip install faiss-cpu
```

2. **CUDA Issues:**
```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
# https://pytorch.org/get-started/locally/
```

3. **Memory Issues:**
```bash
# Reduce batch size in config
CHUNK_SIZE=256
BATCH_SIZE=32
```

4. **API Key Issues:**
```bash
# Verify API key
python -c "import openai; openai.api_key='your_key'; print('Valid')"
```

## 📊 **Performance Monitoring**

### **1. System Monitoring**

```bash
# Install monitoring tools
pip install psutil memory-profiler

# Monitor resource usage
python -m memory_profiler your_script.py
```

### **2. API Monitoring**

```python
# Add to FastAPI app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {process_time:.2f}s")
    return response
```

## 🔒 **Security Setup**

### **1. API Security**

```python
# Add to main.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Implement your token verification logic
    if not is_valid_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return token
```

### **2. Environment Security**

```bash
# Create .env.example (không chứa sensitive data)
cp .env .env.example
# Edit .env.example để remove sensitive data

# Add .env to .gitignore
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
echo "vector_db/" >> .gitignore
```

## 📚 **Next Steps**

1. **Complete Setup Verification:**
   - Test all components individually
   - Verify API connectivity
   - Check frontend-backend communication

2. **Data Preparation:**
   - Prepare sample documents
   - Test document processing pipeline
   - Verify embedding generation

3. **Development Workflow:**
   - Set up Git repository
   - Configure CI/CD pipeline
   - Set up development branches

4. **Production Preparation:**
   - Configure production environment
   - Set up monitoring and logging
   - Prepare deployment scripts

---

## 🎓 **Cách học hiệu quả**

### **Bước 1: Đọc công thức → tra ký hiệu → hiểu trực giác**
- Đọc setup instructions
- Tra cứu bảng ký hiệu để hiểu từng component
- Tìm hiểu ý nghĩa của từng bước setup

### **Bước 2: Điền "Thẻ thuật toán" cho từng mô hình**
- Hoàn thành 12 mục trong thẻ thuật toán cho mỗi setup step
- Viết pseudocode và commands
- Kiểm tra checklist

### **Bước 3: Làm Lab nhỏ → Mini-project → Case study**
- Bắt đầu với lab đơn giản (Python environment)
- Tiến tới mini-project phức tạp hơn (full stack setup)
- Áp dụng vào case study thực tế (production deployment)

### **Bước 4: Đánh giá bằng metric phù hợp**
- Chọn metric đánh giá phù hợp (setup success, performance)
- So sánh với baseline
- Phân tích kết quả và optimize

---

*Chúc bạn setup thành công! 🚀*

## 🐍 **Python Environment Setup**

### **1. Virtual Environment**

```bash
# Tạo virtual environment
python -m venv rag_env

# Activate environment
# Linux/macOS:
source rag_env/bin/activate
# Windows:
rag_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### **2. Install Dependencies**

```bash
# Core dependencies
pip install fastapi uvicorn pydantic

# ML/AI libraries
pip install torch torchvision torchaudio
pip install transformers sentence-transformers
pip install openai anthropic

# Vector database
pip install faiss-cpu  # hoặc faiss-gpu nếu có GPU
pip install chromadb

# Data processing
pip install pandas numpy scipy
pip install scikit-learn

# Web scraping (optional)
pip install beautifulsoup4 requests

# Evaluation
pip install nltk rouge-score

# Development tools
pip install pytest black isort
pip install jupyter notebook
```

### **3. Environment Variables**

Tạo file `.env`:
```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Anthropic API (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Vector database
VECTOR_DB_PATH=./vector_db
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Server settings
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

## 🗄️ **Vector Database Setup**

### **1. FAISS Setup**

```python
# test_faiss.py
import faiss
import numpy as np

def test_faiss_installation():
    """Test FAISS installation"""
    # Create sample data
    dimension = 384
    num_vectors = 1000
    
    # Generate random vectors
    vectors = np.random.randn(num_vectors, dimension).astype('float32')
    
    # Normalize vectors
    faiss.normalize_L2(vectors)
    
    # Create index
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    
    # Test search
    query = np.random.randn(1, dimension).astype('float32')
    faiss.normalize_L2(query)
    
    scores, indices = index.search(query, 5)
    
    print("FAISS test successful!")
    print(f"Found {len(indices[0])} similar vectors")
    
    return True

if __name__ == "__main__":
    test_faiss_installation()
```

### **2. ChromaDB Setup**

```python
# test_chroma.py
import chromadb
from chromadb.config import Settings

def test_chroma_installation():
    """Test ChromaDB installation"""
    # Create client
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"
    ))
    
    # Create collection
    collection = client.create_collection("test_collection")
    
    # Add documents
    documents = [
        "This is a test document about AI.",
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks."
    ]
    
    collection.add(
        documents=documents,
        metadatas=[{"source": "test"} for _ in documents],
        ids=["1", "2", "3"]
    )
    
    # Test query
    results = collection.query(
        query_texts=["What is AI?"],
        n_results=2
    )
    
    print("ChromaDB test successful!")
    print(f"Found {len(results['documents'][0])} relevant documents")
    
    return True

if __name__ == "__main__":
    test_chroma_installation()
```

## 🌐 **Frontend Setup**

### **1. Node.js Environment**

```bash
# Install Node.js (nếu chưa có)
# Ubuntu/Debian:
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS:
brew install node

# Windows: Download từ https://nodejs.org/

# Verify installation
node --version
npm --version
```

### **2. React Frontend**

```bash
# Tạo React app
npx create-react-app rag-frontend
cd rag-frontend

# Install dependencies
npm install axios react-router-dom
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material

# Development dependencies
npm install --save-dev @types/react @types/react-dom
```

### **3. Frontend Configuration**

Tạo file `src/config/api.js`:
```javascript
const API_CONFIG = {
    baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json',
    }
};

export default API_CONFIG;
```

## 🔧 **Development Tools Setup**

### **1. Code Quality Tools**

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
EOF
```

### **2. Testing Setup**

```bash
# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov
pip install httpx  # for FastAPI testing

# Create pytest.ini
cat > pytest.ini << EOF
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=app --cov-report=html
EOF
```

### **3. Docker Setup**

```bash
# Install Docker
# Ubuntu:
sudo apt-get update
sudo apt-get install docker.io docker-compose

# macOS: Download Docker Desktop
# Windows: Download Docker Desktop

# Verify installation
docker --version
docker-compose --version
```

## 📁 **Project Structure**

```bash
# Tạo cấu trúc thư mục
mkdir -p rag_chatbot/{backend,frontend,tests,docs,data}

# Backend structure
mkdir -p rag_chatbot/backend/{app,models,services,utils}
mkdir -p rag_chatbot/backend/app/{api,core,db}

# Frontend structure
mkdir -p rag_chatbot/frontend/src/{components,pages,services,utils}

# Data directories
mkdir -p rag_chatbot/data/{documents,embeddings,vector_db}

# Create necessary files
touch rag_chatbot/backend/requirements.txt
touch rag_chatbot/backend/main.py
touch rag_chatbot/frontend/package.json
touch rag_chatbot/docker-compose.yml
touch rag_chatbot/README.md
```

## 🚀 **Quick Start**

### **1. Backend Setup**

```bash
cd rag_chatbot/backend

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_key_here"

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **2. Frontend Setup**

```bash
cd rag_chatbot/frontend

# Install dependencies
npm install

# Start development server
npm start
```

### **3. Test Installation**

```bash
# Test backend
cd backend
python -m pytest tests/ -v

# Test frontend
cd frontend
npm test

# Test API
curl http://localhost:8000/health
```

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **FAISS Installation Error:**
```bash
# Ubuntu/Debian
sudo apt-get install libblas-dev liblapack-dev
pip install faiss-cpu --no-cache-dir

# macOS
brew install openblas
export LDFLAGS="-L/usr/local/opt/openblas/lib"
export CPPFLAGS="-I/usr/local/opt/openblas/include"
pip install faiss-cpu
```

2. **CUDA Issues:**
```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
# https://pytorch.org/get-started/locally/
```

3. **Memory Issues:**
```bash
# Reduce batch size in config
CHUNK_SIZE=256
BATCH_SIZE=32
```

4. **API Key Issues:**
```bash
# Verify API key
python -c "import openai; openai.api_key='your_key'; print('Valid')"
```

## 📊 **Performance Monitoring**

### **1. System Monitoring**

```bash
# Install monitoring tools
pip install psutil memory-profiler

# Monitor resource usage
python -m memory_profiler your_script.py
```

### **2. API Monitoring**

```python
# Add to FastAPI app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {process_time:.2f}s")
    return response
```

## 🔒 **Security Setup**

### **1. API Security**

```python
# Add to main.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Implement your token verification logic
    if not is_valid_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return token
```

### **2. Environment Security**

```bash
# Create .env.example (không chứa sensitive data)
cp .env .env.example
# Edit .env.example để remove sensitive data

# Add .env to .gitignore
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
echo "vector_db/" >> .gitignore
```

## 📚 **Next Steps**

1. **Complete Setup Verification:**
   - Test all components individually
   - Verify API connectivity
   - Check frontend-backend communication

2. **Data Preparation:**
   - Prepare sample documents
   - Test document processing pipeline
   - Verify embedding generation

3. **Development Workflow:**
   - Set up Git repository
   - Configure CI/CD pipeline
   - Set up development branches

4. **Production Preparation:**
   - Configure production environment
   - Set up monitoring and logging
   - Prepare deployment scripts

---

*Chúc bạn setup thành công! 🚀*

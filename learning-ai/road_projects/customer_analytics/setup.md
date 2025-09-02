# 🔧 Customer Analytics Dashboard - Setup Instructions

> **Mục tiêu**: Hướng dẫn cài đặt môi trường development và production cho Customer Analytics Dashboard

## 🐳 **Docker Environment Setup**

### **1. Prerequisites**

**System Requirements:**
- Docker & Docker Compose
- Python 3.9+
- Node.js 16+
- PostgreSQL 13+
- Redis 6+

**Install Docker:**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# macOS
brew install --cask docker

# Windows
# Download Docker Desktop from https://www.docker.com/products/docker-desktop
```

### **2. Project Structure**

```bash
# Clone repository
git clone https://github.com/nhatnamdeveloper/docs.git
cd docs/road_projects/customer_analytics

# Create project structure
mkdir -p {backend,frontend,data,scripts,config}
mkdir -p backend/{app,tests,models,utils}
mkdir -p frontend/{src,public,components}
mkdir -p data/{raw,processed,models}
```

### **3. Docker Compose Setup**

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  # Backend API
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/customer_analytics
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./backend:/app
      - ./data:/app/data

  # Frontend Dashboard
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend
    volumes:
      - ./frontend:/app

  # PostgreSQL Database
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=customer_analytics
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./data/init:/docker-entrypoint-initdb.d

  # Redis Cache
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # MLflow for Experiment Tracking
  mlflow:
    image: python:3.9
    command: mlflow server --host 0.0.0.0 --port 5000
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000
    volumes:
      - ./mlflow:/mlflow
      - mlflow_data:/mlflow

  # Grafana for Monitoring
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  mlflow_data:
  grafana_data:
```

## 🐍 **Python Backend Setup**

### **1. Backend Dependencies**

**requirements.txt:**
```txt
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Database
sqlalchemy==2.0.23
alembic==1.13.1
psycopg2-binary==2.9.9
redis==5.0.1

# Data Processing
pandas==2.1.4
numpy==1.25.2
scikit-learn==1.3.2
scipy==1.11.4

# ML/Analytics
mlflow==2.8.1
xgboost==2.0.3
lightgbm==4.1.0

# Visualization
plotly==5.17.0
matplotlib==3.8.2
seaborn==0.13.0

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Development
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1
```

### **2. Backend Dockerfile**

**backend/Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### **3. Database Setup**

**data/init/01_init.sql:**
```sql
-- Create database schema
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    age INTEGER,
    gender VARCHAR(10),
    city VARCHAR(100),
    country VARCHAR(100),
    income_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    product_category VARCHAR(100),
    product_name VARCHAR(255),
    transaction_date TIMESTAMP NOT NULL,
    payment_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customer_behavior (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    session_id VARCHAR(100),
    page_url VARCHAR(500),
    time_spent INTEGER, -- seconds
    event_type VARCHAR(50), -- page_view, click, purchase, etc.
    event_data JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customer_segments (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    segment_name VARCHAR(100),
    rfm_score INTEGER,
    clv_prediction DECIMAL(10,2),
    churn_probability DECIMAL(5,4),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_behavior_customer_id ON customer_behavior(customer_id);
CREATE INDEX idx_behavior_timestamp ON customer_behavior(timestamp);
CREATE INDEX idx_segments_customer_id ON customer_segments(customer_id);
```

## ⚛️ **React Frontend Setup**

### **1. Frontend Dependencies**

**frontend/package.json:**
```json
{
  "name": "customer-analytics-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.1",
    "axios": "^1.6.2",
    "recharts": "^2.8.0",
    "antd": "^5.12.8",
    "@ant-design/icons": "^5.2.6",
    "dayjs": "^1.11.10",
    "lodash": "^4.17.21"
  },
  "devDependencies": {
    "@types/react": "^18.2.45",
    "@types/react-dom": "^18.2.18",
    "@types/lodash": "^4.14.202",
    "typescript": "^5.3.3",
    "vite": "^5.0.10",
    "@vitejs/plugin-react": "^4.2.1",
    "eslint": "^8.56.0",
    "prettier": "^3.1.1"
  },
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "format": "prettier --write src"
  }
}
```

### **2. Frontend Dockerfile**

**frontend/Dockerfile:**
```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

## 🚀 **Quick Start**

### **1. Development Environment**

```bash
# Clone and setup
git clone https://github.com/nhatnamdeveloper/docs.git
cd docs/road_projects/customer_analytics

# Start all services
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f backend
```

### **2. Access Services**

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3001 (admin/admin)

### **3. Database Connection**

```bash
# Connect to PostgreSQL
docker-compose exec db psql -U user -d customer_analytics

# Connect to Redis
docker-compose exec redis redis-cli
```

## 🔧 **Configuration**

### **Environment Variables**

**backend/.env:**
```env
# Database
DATABASE_URL=postgresql://user:password@db:5432/customer_analytics
REDIS_URL=redis://redis:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# External APIs
GOOGLE_ANALYTICS_API_KEY=your-ga-api-key
SALESFORCE_API_KEY=your-sf-api-key

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_URL=http://localhost:3001
```

## 📊 **Data Import**

### **Sample Data Generation**

```bash
# Generate sample data
python scripts/generate_sample_data.py

# Import data to database
python scripts/import_data.py

# Run initial analytics
python scripts/run_analytics.py
```

---

**🎯 Next Steps:**
1. Follow the implementation guide in `implementation/`
2. Set up monitoring and alerting
3. Configure production deployment
4. Run performance tests
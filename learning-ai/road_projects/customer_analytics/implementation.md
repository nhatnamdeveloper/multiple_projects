# 💻 Customer Analytics Dashboard - Implementation

> **Mục tiêu**: Implement hệ thống Customer Analytics Dashboard hoàn chỉnh với code thực tế

## 🏗️ **Backend Implementation**

### **1. FastAPI Application Structure**

**backend/app/main.py:**
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import uvicorn
import structlog
from prometheus_client import Counter, Histogram
import time

from app.database import engine, Base
from app.routers import customers, analytics, segments, dashboard
from app.core.config import settings
from app.core.monitoring import setup_monitoring

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Customer Analytics Dashboard")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Setup monitoring
    setup_monitoring()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Customer Analytics Dashboard")

# Create FastAPI app
app = FastAPI(
    title="Customer Analytics Dashboard API",
    description="Real-time customer analytics and insights API",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    REQUEST_LATENCY.observe(process_time)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routers
app.include_router(customers.router, prefix="/api/v1/customers", tags=["customers"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(segments.router, prefix="/api/v1/segments", tags=["segments"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Analytics Dashboard API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

### **2. Database Models**

**backend/app/models/customer.py:**
```python
from sqlalchemy import Column, Integer, String, DateTime, Numeric, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

Base = declarative_base()

class Customer(Base):
    """Customer database model"""
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True)
    first_name = Column(String(100))
    last_name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(10))
    city = Column(String(100))
    country = Column(String(100))
    income_level = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Transaction(Base):
    """Transaction database model"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(50), index=True, nullable=False)
    transaction_id = Column(String(100), unique=True, index=True, nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default="USD")
    product_category = Column(String(100))
    product_name = Column(String(255))
    transaction_date = Column(DateTime(timezone=True), nullable=False)
    payment_method = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CustomerBehavior(Base):
    """Customer behavior database model"""
    __tablename__ = "customer_behavior"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(50), index=True, nullable=False)
    session_id = Column(String(100))
    page_url = Column(String(500))
    time_spent = Column(Integer)  # seconds
    event_type = Column(String(50))  # page_view, click, purchase, etc.
    event_data = Column(JSON)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class CustomerSegment(Base):
    """Customer segment database model"""
    __tablename__ = "customer_segments"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(50), index=True, nullable=False)
    segment_name = Column(String(100))
    rfm_score = Column(Integer)
    clv_prediction = Column(Numeric(10, 2))
    churn_probability = Column(Numeric(5, 4))
    last_updated = Column(DateTime(timezone=True), server_default=func.now())

# Pydantic models for API
class CustomerBase(BaseModel):
    customer_id: str
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    income_level: Optional[str] = None

class CustomerCreate(CustomerBase):
    pass

class CustomerUpdate(BaseModel):
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    income_level: Optional[str] = None

class CustomerResponse(CustomerBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class TransactionBase(BaseModel):
    customer_id: str
    transaction_id: str
    amount: float
    currency: str = "USD"
    product_category: Optional[str] = None
    product_name: Optional[str] = None
    transaction_date: datetime
    payment_method: Optional[str] = None

class TransactionCreate(TransactionBase):
    pass

class TransactionResponse(TransactionBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class CustomerSegmentResponse(BaseModel):
    customer_id: str
    segment_name: str
    rfm_score: Optional[int] = None
    clv_prediction: Optional[float] = None
    churn_probability: Optional[float] = None
    last_updated: datetime
    
    class Config:
        from_attributes = True
```

### **3. Analytics Engine**

**backend/app/analytics/rfm_analyzer.py:**
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sqlalchemy.orm import Session
from app.models.customer import Customer, Transaction
import structlog

logger = structlog.get_logger()

class RFMAnalyzer:
    """RFM (Recency, Frequency, Monetary) Analysis Engine"""
    
    def __init__(self, db: Session):
        self.db = db
        self.analysis_date = datetime.now()
    
    def calculate_rfm_scores(self, customer_id: str) -> Dict[str, int]:
        """Calculate RFM scores for a specific customer"""
        try:
            # Get customer transactions
            transactions = self.db.query(Transaction).filter(
                Transaction.customer_id == customer_id
            ).all()
            
            if not transactions:
                return {"recency": 0, "frequency": 0, "monetary": 0}
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'transaction_date': t.transaction_date,
                    'amount': float(t.amount)
                }
                for t in transactions
            ])
            
            # Calculate RFM metrics
            recency = (self.analysis_date - df['transaction_date'].max()).days
            frequency = len(df)
            monetary = df['amount'].sum()
            
            # Calculate RFM scores (1-5 scale)
            rfm_scores = {
                'recency': self._score_recency(recency),
                'frequency': self._score_frequency(frequency),
                'monetary': self._score_monetary(monetary)
            }
            
            logger.info("RFM scores calculated", 
                       customer_id=customer_id, 
                       rfm_scores=rfm_scores)
            
            return rfm_scores
            
        except Exception as e:
            logger.error("Error calculating RFM scores", 
                        customer_id=customer_id, 
                        error=str(e))
            return {"recency": 0, "frequency": 0, "monetary": 0}
    
    def _score_recency(self, recency_days: int) -> int:
        """Score recency (lower days = higher score)"""
        if recency_days <= 30:
            return 5
        elif recency_days <= 60:
            return 4
        elif recency_days <= 90:
            return 3
        elif recency_days <= 180:
            return 2
        else:
            return 1
    
    def _score_frequency(self, frequency: int) -> int:
        """Score frequency (higher frequency = higher score)"""
        if frequency >= 20:
            return 5
        elif frequency >= 10:
            return 4
        elif frequency >= 5:
            return 3
        elif frequency >= 2:
            return 2
        else:
            return 1
    
    def _score_monetary(self, monetary: float) -> int:
        """Score monetary (higher amount = higher score)"""
        if monetary >= 10000:
            return 5
        elif monetary >= 5000:
            return 4
        elif monetary >= 1000:
            return 3
        elif monetary >= 500:
            return 2
        else:
            return 1
    
    def segment_customers(self, rfm_scores: Dict[str, int]) -> str:
        """Segment customers based on RFM scores"""
        r, f, m = rfm_scores['recency'], rfm_scores['frequency'], rfm_scores['monetary']
        
        # Define segments
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        elif r >= 3 and f >= 3 and m >= 2:
            return "At Risk"
        elif r >= 2 and f >= 2 and m >= 2:
            return "Can't Lose"
        elif r >= 3 and f >= 2 and m >= 2:
            return "About to Sleep"
        elif r >= 2 and f >= 2 and m >= 3:
            return "Need Attention"
        elif r >= 2 and f >= 3 and m >= 2:
            return "New Customers"
        else:
            return "Lost"
    
    def analyze_all_customers(self) -> Dict[str, Any]:
        """Analyze RFM for all customers"""
        try:
            # Get all customers with transactions
            customers = self.db.query(Customer).all()
            
            results = {
                'total_customers': len(customers),
                'segments': {},
                'rfm_distribution': {'recency': {}, 'frequency': {}, 'monetary': {}}
            }
            
            for customer in customers:
                rfm_scores = self.calculate_rfm_scores(customer.customer_id)
                segment = self.segment_customers(rfm_scores)
                
                # Count segments
                results['segments'][segment] = results['segments'].get(segment, 0) + 1
                
                # Count RFM distributions
                for metric, score in rfm_scores.items():
                    results['rfm_distribution'][metric][score] = \
                        results['rfm_distribution'][metric].get(score, 0) + 1
            
            logger.info("RFM analysis completed", 
                       total_customers=len(customers),
                       segments_count=len(results['segments']))
            
            return results
            
        except Exception as e:
            logger.error("Error in RFM analysis", error=str(e))
            return {}
```

### **4. API Endpoints**

**backend/app/routers/analytics.py:**
```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from app.database import get_db
from app.analytics.rfm_analyzer import RFMAnalyzer
from app.analytics.clv_predictor import CLVPredictor
from app.models.customer import Customer, Transaction
from app.schemas.analytics import (
    RFMAnalysisResponse,
    CLVPredictionResponse,
    CustomerMetricsResponse,
    DashboardMetricsResponse
)

router = APIRouter()

@router.get("/rfm/{customer_id}", response_model=RFMAnalysisResponse)
async def get_rfm_analysis(
    customer_id: str,
    db: Session = Depends(get_db)
):
    """Get RFM analysis for a specific customer"""
    try:
        analyzer = RFMAnalyzer(db)
        rfm_scores = analyzer.calculate_rfm_scores(customer_id)
        segment = analyzer.segment_customers(rfm_scores)
        
        return RFMAnalysisResponse(
            customer_id=customer_id,
            rfm_scores=rfm_scores,
            segment=segment,
            analysis_date=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rfm/overview", response_model=Dict[str, Any])
async def get_rfm_overview(db: Session = Depends(get_db)):
    """Get RFM analysis overview for all customers"""
    try:
        analyzer = RFMAnalyzer(db)
        results = analyzer.analyze_all_customers()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clv/{customer_id}", response_model=CLVPredictionResponse)
async def get_clv_prediction(
    customer_id: str,
    db: Session = Depends(get_db)
):
    """Get CLV prediction for a specific customer"""
    try:
        predictor = CLVPredictor(db)
        clv_prediction = predictor.predict_clv(customer_id)
        
        return CLVPredictionResponse(
            customer_id=customer_id,
            clv_prediction=clv_prediction,
            prediction_date=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/customer/{customer_id}", response_model=CustomerMetricsResponse)
async def get_customer_metrics(
    customer_id: str,
    db: Session = Depends(get_db)
):
    """Get comprehensive metrics for a customer"""
    try:
        # Get customer data
        customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Get transaction data
        transactions = db.query(Transaction).filter(
            Transaction.customer_id == customer_id
        ).all()
        
        if not transactions:
            return CustomerMetricsResponse(
                customer_id=customer_id,
                total_transactions=0,
                total_spent=0.0,
                average_order_value=0.0,
                first_purchase_date=None,
                last_purchase_date=None,
                purchase_frequency=0.0
            )
        
        # Calculate metrics
        df = pd.DataFrame([
            {
                'amount': float(t.amount),
                'transaction_date': t.transaction_date
            }
            for t in transactions
        ])
        
        total_transactions = len(transactions)
        total_spent = df['amount'].sum()
        average_order_value = df['amount'].mean()
        first_purchase_date = df['transaction_date'].min()
        last_purchase_date = df['transaction_date'].max()
        
        # Calculate purchase frequency (days between purchases)
        df_sorted = df.sort_values('transaction_date')
        if len(df_sorted) > 1:
            purchase_intervals = (df_sorted['transaction_date'].diff().dt.days).dropna()
            purchase_frequency = purchase_intervals.mean()
        else:
            purchase_frequency = 0.0
        
        return CustomerMetricsResponse(
            customer_id=customer_id,
            total_transactions=total_transactions,
            total_spent=total_spent,
            average_order_value=average_order_value,
            first_purchase_date=first_purchase_date,
            last_purchase_date=last_purchase_date,
            purchase_frequency=purchase_frequency
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/overview", response_model=DashboardMetricsResponse)
async def get_dashboard_overview(
    db: Session = Depends(get_db),
    days: int = Query(30, description="Number of days to analyze")
):
    """Get dashboard overview metrics"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get recent transactions
        recent_transactions = db.query(Transaction).filter(
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        ).all()
        
        if not recent_transactions:
            return DashboardMetricsResponse(
                total_revenue=0.0,
                total_transactions=0,
                average_order_value=0.0,
                unique_customers=0,
                top_products=[],
                revenue_trend=[],
                customer_segments={}
            )
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'amount': float(t.amount),
                'transaction_date': t.transaction_date,
                'customer_id': t.customer_id,
                'product_category': t.product_category,
                'product_name': t.product_name
            }
            for t in recent_transactions
        ])
        
        # Calculate metrics
        total_revenue = df['amount'].sum()
        total_transactions = len(df)
        average_order_value = df['amount'].mean()
        unique_customers = df['customer_id'].nunique()
        
        # Top products
        top_products = df.groupby('product_name')['amount'].sum().sort_values(
            ascending=False
        ).head(10).to_dict()
        
        # Revenue trend (daily)
        revenue_trend = df.groupby(df['transaction_date'].dt.date)['amount'].sum().to_dict()
        
        # Customer segments (using RFM)
        analyzer = RFMAnalyzer(db)
        segments_analysis = analyzer.analyze_all_customers()
        customer_segments = segments_analysis.get('segments', {})
        
        return DashboardMetricsResponse(
            total_revenue=total_revenue,
            total_transactions=total_transactions,
            average_order_value=average_order_value,
            unique_customers=unique_customers,
            top_products=list(top_products.items()),
            revenue_trend=list(revenue_trend.items()),
            customer_segments=customer_segments
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## ⚛️ **Frontend Implementation**

### **1. React Dashboard Component**

**frontend/src/components/Dashboard.tsx:**
```typescript
import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Table, DatePicker, Select } from 'antd';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { 
    DollarOutlined, 
    ShoppingCartOutlined, 
    UserOutlined, 
    TrendingUpOutlined 
} from '@ant-design/icons';
import axios from 'axios';
import dayjs from 'dayjs';

const { RangePicker } = DatePicker;
const { Option } = Select;

interface DashboardMetrics {
    total_revenue: number;
    total_transactions: number;
    average_order_value: number;
    unique_customers: number;
    top_products: [string, number][];
    revenue_trend: [string, number][];
    customer_segments: Record<string, number>;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const Dashboard: React.FC = () => {
    const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
    const [loading, setLoading] = useState(true);
    const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
        dayjs().subtract(30, 'days'),
        dayjs()
    ]);

    useEffect(() => {
        fetchDashboardData();
    }, [dateRange]);

    const fetchDashboardData = async () => {
        try {
            setLoading(true);
            const [startDate, endDate] = dateRange;
            const days = endDate.diff(startDate, 'days');
            
            const response = await axios.get(`/api/v1/analytics/dashboard/overview?days=${days}`);
            setMetrics(response.data);
        } catch (error) {
            console.error('Error fetching dashboard data:', error);
        } finally {
            setLoading(false);
        }
    };

    const formatCurrency = (value: number) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(value);
    };

    const formatNumber = (value: number) => {
        return new Intl.NumberFormat('en-US').format(value);
    };

    if (loading) {
        return <div>Loading dashboard...</div>;
    }

    if (!metrics) {
        return <div>No data available</div>;
    }

    return (
        <div style={{ padding: '24px' }}>
            <h1>Customer Analytics Dashboard</h1>
            
            {/* Date Range Selector */}
            <div style={{ marginBottom: '24px' }}>
                <RangePicker
                    value={dateRange}
                    onChange={(dates) => dates && setDateRange(dates)}
                    style={{ width: 300 }}
                />
            </div>

            {/* Key Metrics */}
            <Row gutter={16} style={{ marginBottom: '24px' }}>
                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Total Revenue"
                            value={metrics.total_revenue}
                            prefix={<DollarOutlined />}
                            formatter={(value) => formatCurrency(value as number)}
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Total Transactions"
                            value={metrics.total_transactions}
                            prefix={<ShoppingCartOutlined />}
                            formatter={(value) => formatNumber(value as number)}
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Average Order Value"
                            value={metrics.average_order_value}
                            prefix={<DollarOutlined />}
                            formatter={(value) => formatCurrency(value as number)}
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Unique Customers"
                            value={metrics.unique_customers}
                            prefix={<UserOutlined />}
                            formatter={(value) => formatNumber(value as number)}
                        />
                    </Card>
                </Col>
            </Row>

            {/* Charts */}
            <Row gutter={16}>
                <Col span={12}>
                    <Card title="Revenue Trend" style={{ marginBottom: '16px' }}>
                        <LineChart width={500} height={300} data={metrics.revenue_trend.map(([date, value]) => ({
                            date,
                            revenue: value
                        }))}>
                            <Line 
                                type="monotone" 
                                dataKey="revenue" 
                                stroke="#8884d8" 
                                strokeWidth={2}
                            />
                        </LineChart>
                    </Card>
                </Col>
                <Col span={12}>
                    <Card title="Customer Segments" style={{ marginBottom: '16px' }}>
                        <PieChart width={500} height={300}>
                            <Pie
                                data={Object.entries(metrics.customer_segments).map(([name, value], index) => ({
                                    name,
                                    value
                                }))}
                                cx={250}
                                cy={150}
                                outerRadius={100}
                                fill="#8884d8"
                                dataKey="value"
                                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            >
                                {Object.entries(metrics.customer_segments).map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                        </PieChart>
                    </Card>
                </Col>
            </Row>

            {/* Top Products Table */}
            <Card title="Top Products by Revenue" style={{ marginTop: '16px' }}>
                <Table
                    dataSource={metrics.top_products.map(([name, revenue], index) => ({
                        key: index,
                        product: name,
                        revenue: revenue,
                        formattedRevenue: formatCurrency(revenue)
                    }))}
                    columns={[
                        {
                            title: 'Product',
                            dataIndex: 'product',
                            key: 'product',
                        },
                        {
                            title: 'Revenue',
                            dataIndex: 'formattedRevenue',
                            key: 'revenue',
                            sorter: (a, b) => a.revenue - b.revenue,
                        }
                    ]}
                    pagination={false}
                />
            </Card>
        </div>
    );
};

export default Dashboard;
```

## 🚀 **Quick Start Implementation**

### **1. Start the Application**

```bash
# Clone and setup
cd road_projects/customer_analytics

# Start all services
docker-compose up -d

# Check if all services are running
docker-compose ps

# View logs
docker-compose logs -f backend
```

### **2. Generate Sample Data**

```bash
# Generate sample customers and transactions
python scripts/generate_sample_data.py

# The script will create:
# - 1000 customers with demographic data
# - 5000 transactions over the last 12 months
# - Customer behavior data
```

### **3. Access the Dashboard**

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3001

---

**🎯 Next Steps:**
1. Implement additional analytics features
2. Add real-time data streaming
3. Set up monitoring and alerting
4. Deploy to production environment
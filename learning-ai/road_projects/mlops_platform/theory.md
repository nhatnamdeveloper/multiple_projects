# 🚀 MLOps Platform - Lý thuyết

> **Mục tiêu**: Xây dựng nền tảng MLOps hoàn chỉnh với experiment tracking, model registry, deployment, monitoring

## 🧠 **Lý thuyết cơ bản**

### **1. MLOps Overview**

**Khái niệm cốt lõi:**
- **MLOps (Machine Learning Operations)**: Quy trình và công cụ để quản lý ML lifecycle
- **ML Lifecycle**: Development → Training → Deployment → Monitoring → Retraining
- **CI/CD for ML**: Continuous Integration/Deployment cho machine learning
- **Model Governance**: Quản lý, kiểm soát và tuân thủ cho ML models

### **2. MLOps Components**

**A. Experiment Tracking:**
- **MLflow**: Open-source platform cho ML lifecycle
- **Weights & Biases**: Experiment tracking và collaboration
- **TensorBoard**: Visualization cho TensorFlow models
- **Custom Solutions**: In-house experiment tracking

**B. Model Registry:**
- **Model Versioning**: Quản lý phiên bản model
- **Model Lineage**: Tracking model origins và dependencies
- **Model Metadata**: Metadata management cho models
- **Model Approval Workflow**: Approval process cho deployment

**C. Model Deployment:**
- **Model Serving**: REST APIs, gRPC, batch inference
- **Containerization**: Docker, Kubernetes deployment
- **A/B Testing**: Model comparison và experimentation
- **Canary Deployment**: Gradual rollout strategies

**D. Monitoring & Observability:**
- **Model Performance**: Accuracy, latency, throughput
- **Data Drift**: Detection of data distribution changes
- **Model Drift**: Detection of model performance degradation
- **Infrastructure Monitoring**: Resource utilization, errors

### **3. MLOps Best Practices**

**A. Reproducibility:**
- **Code Versioning**: Git-based version control
- **Data Versioning**: DVC, LakeFS, data lineage
- **Environment Management**: Conda, Docker, virtual environments
- **Experiment Tracking**: Parameters, metrics, artifacts

**B. Scalability:**
- **Distributed Training**: Multi-GPU, multi-node training
- **Model Optimization**: Quantization, pruning, distillation
- **Auto-scaling**: Kubernetes HPA, cloud auto-scaling
- **Load Balancing**: Traffic distribution across model instances

**C. Security:**
- **Model Security**: Adversarial attacks, model stealing
- **Data Security**: Encryption, access control, GDPR compliance
- **Infrastructure Security**: Network security, authentication
- **Audit Trail**: Complete logging và tracking

## 🔧 **Technical Architecture**

### **1. MLOps Platform Architecture**

```python
class MLOpsArchitecture:
    """Architecture cho MLOps Platform"""
    
    def __init__(self):
        self.components = {
            'development': ['Code Management', 'Data Versioning', 'Experiment Tracking'],
            'training': ['Distributed Training', 'Hyperparameter Tuning', 'Model Validation'],
            'deployment': ['Model Registry', 'Container Orchestration', 'Serving Infrastructure'],
            'monitoring': ['Performance Monitoring', 'Data Drift Detection', 'Alerting'],
            'governance': ['Model Approval', 'Access Control', 'Compliance Tracking']
        }
    
    def explain_data_flow(self):
        """Explain data flow trong hệ thống"""
        print("""
        **MLOps Platform Data Flow:**
        
        1. **Development Phase:**
           - Code development với version control
           - Data versioning và lineage tracking
           - Experiment tracking (parameters, metrics, artifacts)
           - Model prototyping và validation
        
        2. **Training Phase:**
           - Distributed training orchestration
           - Hyperparameter optimization
           - Model validation và testing
           - Artifact storage và versioning
        
        3. **Deployment Phase:**
           - Model packaging và containerization
           - Model registry management
           - Deployment orchestration (Kubernetes)
           - A/B testing và canary deployments
        
        4. **Monitoring Phase:**
           - Real-time model performance monitoring
           - Data drift detection
           - Infrastructure monitoring
           - Alerting và incident response
        
        5. **Governance Phase:**
           - Model approval workflows
           - Access control và security
           - Compliance tracking
           - Audit trail maintenance
        """)
```

### **2. Experiment Tracking Implementation**

**MLflow Integration:**
```python
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn

class ExperimentTracker:
    """MLflow-based experiment tracking"""
    
    def __init__(self, tracking_uri="http://localhost:5000"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def start_experiment(self, experiment_name: str, run_name: str = None):
        """Start a new experiment run"""
        mlflow.set_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name)
    
    def log_parameters(self, params: dict):
        """Log experiment parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log experiment metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, model_name: str, model_type: str = "sklearn"):
        """Log trained model"""
        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(model, model_name)
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifacts (files, plots, etc.)"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_dataframe(self, df: pd.DataFrame, name: str):
        """Log pandas DataFrame as artifact"""
        df.to_csv(f"{name}.csv", index=False)
        mlflow.log_artifact(f"{name}.csv")
    
    def end_run(self):
        """End current experiment run"""
        mlflow.end_run()
    
    def search_runs(self, experiment_name: str, filter_string: str = None):
        """Search experiment runs"""
        experiment = self.client.get_experiment_by_name(experiment_name)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string
        )
        return runs
    
    def compare_runs(self, run_ids: list):
        """Compare multiple runs"""
        runs_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            run_data = {
                'run_id': run_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'params': run.data.params,
                'metrics': run.data.metrics
            }
            runs_data.append(run_data)
        
        return pd.DataFrame(runs_data)

class MLExperiment:
    """Complete ML experiment with tracking"""
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
    
    def run_classification_experiment(self, data: pd.DataFrame, target_col: str, experiment_name: str):
        """Run a complete classification experiment"""
        
        with self.tracker.start_experiment(experiment_name, "classification_experiment"):
            
            # Log data info
            self.tracker.log_parameters({
                'dataset_size': len(data),
                'features': len(data.columns) - 1,
                'target_column': target_col
            })
            
            # Data preprocessing
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Log split info
            self.tracker.log_parameters({
                'train_size': len(X_train),
                'test_size': len(X_test),
                'random_state': 42
            })
            
            # Model training
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Log metrics
            self.tracker.log_metrics({
                'accuracy': accuracy,
                'precision_macro': report['macro avg']['precision'],
                'recall_macro': report['macro avg']['recall'],
                'f1_macro': report['macro avg']['f1-score']
            })
            
            # Log model
            self.tracker.log_model(model, "random_forest_classifier", "sklearn")
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.tracker.log_dataframe(feature_importance, "feature_importance")
            
            # Log predictions
            predictions_df = pd.DataFrame({
                'true': y_test,
                'predicted': y_pred,
                'probability': y_pred_proba.max(axis=1)
            })
            
            self.tracker.log_dataframe(predictions_df, "predictions")
            
            return model, accuracy
```

### **3. Model Registry Implementation**

**Model Registry Management:**
```python
import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    model_name: str
    version: str
    model_type: str
    framework: str
    created_at: str
    created_by: str
    description: str
    tags: List[str]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    status: str  # 'development', 'staging', 'production', 'archived'
    approval_status: str  # 'pending', 'approved', 'rejected'
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None

class ModelRegistry:
    """Model registry for versioning and management"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = registry_path
        self.metadata_path = os.path.join(registry_path, "metadata")
        self.models_path = os.path.join(registry_path, "models")
        
        # Create directories
        os.makedirs(self.metadata_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
    
    def register_model(self, model, model_name: str, model_type: str, 
                      framework: str, description: str = "", 
                      tags: List[str] = None, parameters: Dict[str, Any] = None,
                      metrics: Dict[str, float] = None, created_by: str = "unknown") -> str:
        """Register a new model"""
        
        # Generate model ID and version
        model_id = self._generate_model_id(model_name)
        version = self._get_next_version(model_name)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            model_type=model_type,
            framework=framework,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            description=description,
            tags=tags or [],
            parameters=parameters or {},
            metrics=metrics or {},
            artifacts=[],
            status="development",
            approval_status="pending"
        )
        
        # Save model
        model_path = os.path.join(self.models_path, f"{model_id}_{version}")
        self._save_model(model, model_path, framework)
        
        # Save metadata
        metadata.artifacts.append(model_path)
        self._save_metadata(metadata)
        
        return model_id
    
    def _generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{timestamp}"
    
    def _get_next_version(self, model_name: str) -> str:
        """Get next version number for model"""
        existing_versions = self.list_model_versions(model_name)
        if not existing_versions:
            return "v1.0.0"
        
        # Simple versioning - in practice, use semantic versioning
        return f"v{len(existing_versions) + 1}.0.0"
    
    def _save_model(self, model, model_path: str, framework: str):
        """Save model to disk"""
        os.makedirs(model_path, exist_ok=True)
        
        if framework == "sklearn":
            with open(os.path.join(model_path, "model.pkl"), "wb") as f:
                pickle.dump(model, f)
        elif framework == "pytorch":
            torch.save(model.state_dict(), os.path.join(model_path, "model.pth"))
        elif framework == "tensorflow":
            model.save(os.path.join(model_path, "model"))
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save model metadata"""
        metadata_file = os.path.join(self.metadata_path, f"{metadata.model_id}_{metadata.version}.json")
        with open(metadata_file, "w") as f:
            json.dump(asdict(metadata), f, indent=2)
    
    def load_model(self, model_id: str, version: str = None):
        """Load model from registry"""
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_path = metadata.artifacts[0]  # Assuming first artifact is the model
        
        if metadata.framework == "sklearn":
            with open(os.path.join(model_path, "model.pkl"), "rb") as f:
                return pickle.load(f)
        elif metadata.framework == "pytorch":
            # You'd need to know the model architecture
            raise NotImplementedError("PyTorch model loading requires architecture definition")
        elif metadata.framework == "tensorflow":
            import tensorflow as tf
            return tf.keras.models.load_model(os.path.join(model_path, "model"))
        else:
            raise ValueError(f"Unsupported framework: {metadata.framework}")
    
    def get_model_metadata(self, model_id: str, version: str = None) -> Optional[ModelMetadata]:
        """Get model metadata"""
        if version:
            metadata_file = os.path.join(self.metadata_path, f"{model_id}_{version}.json")
        else:
            # Get latest version
            metadata_files = [f for f in os.listdir(self.metadata_path) 
                           if f.startswith(model_id)]
            if not metadata_files:
                return None
            metadata_file = os.path.join(self.metadata_path, sorted(metadata_files)[-1])
        
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                data = json.load(f)
                return ModelMetadata(**data)
        return None
    
    def list_models(self) -> List[str]:
        """List all models in registry"""
        metadata_files = os.listdir(self.metadata_path)
        model_ids = set()
        
        for file in metadata_files:
            if file.endswith(".json"):
                model_id = file.split("_")[0]
                model_ids.add(model_id)
        
        return list(model_ids)
    
    def list_model_versions(self, model_name: str) -> List[str]:
        """List all versions of a model"""
        metadata_files = [f for f in os.listdir(self.metadata_path) 
                         if f.startswith(model_name) and f.endswith(".json")]
        
        versions = []
        for file in metadata_files:
            version = file.split("_")[-1].replace(".json", "")
            versions.append(version)
        
        return sorted(versions)
    
    def update_model_status(self, model_id: str, version: str, status: str, 
                          approved_by: str = None):
        """Update model status"""
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Model {model_id} version {version} not found")
        
        metadata.status = status
        if approved_by:
            metadata.approved_by = approved_by
            metadata.approved_at = datetime.now().isoformat()
            metadata.approval_status = "approved"
        
        self._save_metadata(metadata)
    
    def delete_model(self, model_id: str, version: str):
        """Delete model from registry"""
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Model {model_id} version {version} not found")
        
        # Delete model files
        for artifact in metadata.artifacts:
            if os.path.exists(artifact):
                import shutil
                shutil.rmtree(artifact)
        
        # Delete metadata
        metadata_file = os.path.join(self.metadata_path, f"{model_id}_{version}.json")
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
```

### **4. Model Deployment Implementation**

**Model Serving with FastAPI:**
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, Any, List
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class ModelServer:
    """Model serving server with FastAPI"""
    
    def __init__(self, model_registry: ModelRegistry, model_id: str, version: str = None):
        self.model_registry = model_registry
        self.model_id = model_id
        self.version = version
        self.model = None
        self.metadata = None
        self.app = FastAPI(title=f"Model Server - {model_id}")
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Load model
        self._load_model()
        
        # Setup routes
        self._setup_routes()
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.latency_sum = 0
    
    def _load_model(self):
        """Load model from registry"""
        try:
            self.model = self.model_registry.load_model(self.model_id, self.version)
            self.metadata = self.model_registry.get_model_metadata(self.model_id, self.version)
            logging.info(f"Loaded model {self.model_id} version {self.metadata.version}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "model_id": self.model_id,
                "version": self.metadata.version if self.metadata else "unknown",
                "status": "running",
                "model_type": self.metadata.model_type if self.metadata else "unknown"
            }
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "model_loaded": self.model is not None}
        
        @self.app.get("/model-info")
        async def model_info():
            if not self.metadata:
                raise HTTPException(status_code=404, detail="Model metadata not found")
            
            return {
                "model_id": self.metadata.model_id,
                "model_name": self.metadata.model_name,
                "version": self.metadata.version,
                "model_type": self.metadata.model_type,
                "framework": self.metadata.framework,
                "created_at": self.metadata.created_at,
                "description": self.metadata.description,
                "tags": self.metadata.tags,
                "parameters": self.metadata.parameters,
                "metrics": self.metadata.metrics,
                "status": self.metadata.status
            }
        
        @self.app.post("/predict")
        async def predict(data: Dict[str, Any]):
            start_time = time.time()
            
            try:
                # Increment request count
                self.request_count += 1
                
                # Validate input
                if "features" not in data:
                    raise HTTPException(status_code=400, detail="Features not provided")
                
                features = data["features"]
                
                # Convert to appropriate format
                if isinstance(features, list):
                    features = np.array(features)
                elif isinstance(features, dict):
                    features = pd.DataFrame([features])
                
                # Make prediction
                if self.metadata.framework == "sklearn":
                    if len(features.shape) == 1:
                        features = features.reshape(1, -1)
                    prediction = self.model.predict(features)
                    prediction_proba = self.model.predict_proba(features)
                    
                    result = {
                        "prediction": prediction.tolist(),
                        "probability": prediction_proba.tolist(),
                        "model_id": self.model_id,
                        "version": self.metadata.version
                    }
                else:
                    # Handle other frameworks
                    result = {"prediction": "Not implemented for this framework"}
                
                # Calculate latency
                latency = time.time() - start_time
                self.latency_sum += latency
                
                return result
                
            except Exception as e:
                self.error_count += 1
                logging.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict-batch")
        async def predict_batch(data: Dict[str, Any]):
            start_time = time.time()
            
            try:
                self.request_count += 1
                
                if "features" not in data:
                    raise HTTPException(status_code=400, detail="Features not provided")
                
                features_list = data["features"]
                
                if not isinstance(features_list, list):
                    raise HTTPException(status_code=400, detail="Features must be a list")
                
                # Convert to DataFrame
                features_df = pd.DataFrame(features_list)
                
                # Make batch prediction
                if self.metadata.framework == "sklearn":
                    predictions = self.model.predict(features_df)
                    predictions_proba = self.model.predict_proba(features_df)
                    
                    result = {
                        "predictions": predictions.tolist(),
                        "probabilities": predictions_proba.tolist(),
                        "model_id": self.model_id,
                        "version": self.metadata.version,
                        "batch_size": len(features_list)
                    }
                else:
                    result = {"predictions": "Not implemented for this framework"}
                
                latency = time.time() - start_time
                self.latency_sum += latency
                
                return result
                
            except Exception as e:
                self.error_count += 1
                logging.error(f"Batch prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def get_metrics():
            avg_latency = self.latency_sum / self.request_count if self.request_count > 0 else 0
            error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
            
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": error_rate,
                "average_latency": avg_latency,
                "model_id": self.model_id,
                "version": self.metadata.version if self.metadata else "unknown"
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the model server"""
        uvicorn.run(self.app, host=host, port=port)

class ModelDeploymentManager:
    """Manage model deployments"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.active_deployments = {}
    
    def deploy_model(self, model_id: str, version: str, host: str = "0.0.0.0", 
                    port: int = 8000) -> str:
        """Deploy a model"""
        deployment_id = f"{model_id}_{version}_{port}"
        
        if deployment_id in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} already exists")
        
        # Create model server
        server = ModelServer(self.model_registry, model_id, version)
        
        # Start server in background
        import threading
        server_thread = threading.Thread(
            target=server.run,
            kwargs={"host": host, "port": port}
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Store deployment info
        self.active_deployments[deployment_id] = {
            "server": server,
            "thread": server_thread,
            "host": host,
            "port": port,
            "status": "running"
        }
        
        # Update model status
        self.model_registry.update_model_status(model_id, version, "production")
        
        return deployment_id
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all active deployments"""
        deployments = []
        
        for deployment_id, info in self.active_deployments.items():
            deployments.append({
                "deployment_id": deployment_id,
                "host": info["host"],
                "port": info["port"],
                "status": info["status"]
            })
        
        return deployments
    
    def stop_deployment(self, deployment_id: str):
        """Stop a deployment"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        # In a real implementation, you'd properly stop the server
        # For now, just remove from active deployments
        del self.active_deployments[deployment_id]
```

## 📊 **Monitoring Implementation**

### **1. Model Performance Monitoring**

**Performance Monitor:**
```python
import time
import threading
from collections import deque
from typing import Dict, List, Any
import numpy as np
import requests
import json

class ModelPerformanceMonitor:
    """Monitor model performance in real-time"""
    
    def __init__(self, model_endpoint: str, window_size: int = 1000):
        self.model_endpoint = model_endpoint
        self.window_size = window_size
        
        # Performance metrics
        self.latency_history = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)
        self.error_history = deque(maxlen=window_size)
        self.accuracy_history = deque(maxlen=window_size)
        
        # Current metrics
        self.current_requests = 0
        self.current_errors = 0
        self.start_time = time.time()
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: int = 30):
        """Start monitoring thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get current metrics
                metrics = self._get_current_metrics()
                
                # Update history
                self.latency_history.append(metrics['latency'])
                self.throughput_history.append(metrics['throughput'])
                self.error_history.append(metrics['error_rate'])
                
                # Log metrics
                self._log_metrics(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Monitoring error: {str(e)}")
                time.sleep(interval)
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        try:
            response = requests.get(f"{self.model_endpoint}/metrics", timeout=5)
            data = response.json()
            
            return {
                'latency': data.get('average_latency', 0),
                'throughput': data.get('request_count', 0),
                'error_rate': data.get('error_rate', 0),
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'latency': 0,
                'throughput': 0,
                'error_rate': 1.0,
                'timestamp': time.time()
            }
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to storage"""
        # In practice, log to database or monitoring system
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check for performance alerts"""
        # Latency alert
        if metrics['latency'] > 1.0:  # 1 second threshold
            self._send_alert("High Latency", f"Latency: {metrics['latency']:.2f}s")
        
        # Error rate alert
        if metrics['error_rate'] > 0.05:  # 5% error rate threshold
            self._send_alert("High Error Rate", f"Error Rate: {metrics['error_rate']:.2%}")
        
        # Throughput alert
        if metrics['throughput'] < 10:  # Low throughput threshold
            self._send_alert("Low Throughput", f"Throughput: {metrics['throughput']} req/s")
    
    def _send_alert(self, alert_type: str, message: str):
        """Send alert notification"""
        # In practice, send to alerting system (Slack, email, etc.)
        print(f"ALERT [{alert_type}]: {message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.latency_history:
            return {}
        
        return {
            'avg_latency': np.mean(self.latency_history),
            'p95_latency': np.percentile(self.latency_history, 95),
            'p99_latency': np.percentile(self.latency_history, 99),
            'avg_throughput': np.mean(self.throughput_history),
            'avg_error_rate': np.mean(self.error_history),
            'total_requests': len(self.latency_history),
            'monitoring_duration': time.time() - self.start_time
        }
```

## 🎯 **Business Impact**

### **Expected Outcomes:**
- **Faster Deployment**: 80% reduction in time-to-production
- **Better Model Performance**: Continuous monitoring và improvement
- **Reduced Risk**: Automated testing và validation
- **Cost Optimization**: Efficient resource utilization
- **Compliance**: Automated governance và audit trails

---

**📚 References:**
- "MLOps: Continuous Delivery and Automation Pipelines in Machine Learning" by Treveil et al.
- "Designing Machine Learning Systems" by Chip Huyen
- "Building Machine Learning Powered Applications" by Emmanuel Ameisen
- "MLflow: A Platform for ML Development and Production" by Chen et al.
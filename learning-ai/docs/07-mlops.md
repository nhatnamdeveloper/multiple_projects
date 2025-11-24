# 🚀 MLOps - Machine Learning Operations

> **Mục tiêu**: Trở thành chuyên gia MLOps, có khả năng xây dựng và vận hành hệ thống ML production end-to-end một cách đáng tin cậy, có thể mở rộng và tái lập.

Nếu Data Science là quá trình tạo ra một công thức nấu ăn ngon trong một căn bếp tại nhà (Jupyter Notebook), thì **MLOps** là nghệ thuật xây dựng và vận hành một chuỗi nhà hàng chuyên nghiệp, đảm bảo món ăn (mô hình) được phục vụ đến hàng triệu thực khách (người dùng) với chất lượng đồng nhất, nhanh chóng và an toàn.

MLOps là sự kết hợp của **Machine Learning**, **Development** và **Operations**. Nó áp dụng các nguyên tắc của DevOps (như CI/CD, tự động hóa, giám sát) vào vòng đời của một dự án machine learning.

## 📋 Tổng quan nội dung

```mermaid
graph TD
    A[🚀 MLOps] --> B[🔧 Model Development Lifecycle]
    A --> C[📊 Model Serving & Deployment]
    A --> D[🔄 CI/CD/CT & Pipelines]
    A --> E[📈 Monitoring & Observability]
    A --> F[🛡️ Security & Governance]
    
    B --> B1[Experiment Tracking]
    B --> B2[Model Registry]
    B --> B3[Data & Feature Versioning]
    B --> B4[Feature Stores]
    
    C --> C1[Online vs. Batch Serving]
    C --> C2[Containerization (Docker)]
    C --> C3[Orchestration (Kubernetes)]
    
    D --> D1[CI - Tích hợp liên tục]
    D --> D2[CD - Triển khai liên tục]
    D --> D3[CT - Huấn luyện liên tục]
    
    E --> E1[Model Performance Monitoring]
    E --> E2[Data Drift & Concept Drift]
    E --> E3[Infrastructure Monitoring]
    
    F --> F1[Access Control (IAM)]
    F --> F2[Data Privacy (PII)]
    F --> F3[Model Governance]
    
```

![MLOps Overview](assets/mlops-overview.svg)

![MLOps Overview PNG](assets/mlops-overview.png)

**📁 [Xem file PNG trực tiếp](assets/mlops-overview.png)**

**📁 [Xem file PNG trực tiếp](assets/mlops-overview.png)**

**📁 [Xem file PNG trực tiếp](assets/mlops-overview.png)**

## 🧩 Chương trình 50/50 (Lý thuyết : Thực hành)

- Mục tiêu: 50% lý thuyết (kiến trúc hệ thống, tiêu chuẩn vận hành/safety, chiến lược triển khai), 50% thực hành (triển khai pipeline/serving/monitoring có kiểm thử)

| Mô-đun | Lý thuyết (50%) | Thực hành (50%) |
|---|---|---|
| Experiment & Registry | Nguyên tắc tracking/versioning | Thiết lập MLflow + registry flow |
| Serving & Deployment | Kiến trúc REST/batch/stream | FastAPI + container + autoscale demo |
| CI/CD & Pipelines | GitOps, tests, rollback | GH Actions pipeline + smoke tests |
| Monitoring & Drift | Metrics, drift, alerting | Evidently + Grafana dashboards |
| Security & Cost | AuthZ, PII, cost control | Policy checks + cost report |

Rubric (100đ/module): Lý thuyết 30 | Code 30 | Kết quả 30 | Báo cáo 10

---

## 🔧 1. Vòng đời phát triển mô hình (Model Development Lifecycle)

### 1.1 Theo dõi thí nghiệm (Experiment Tracking)

> **Tại sao cần thiết?** Machine Learning là một bộ môn khoa học thực nghiệm. Một nguyên tắc vàng của khoa học là **khả năng tái lập (reproducibility)**. Nếu bạn không thể tái lập lại kết quả của chính mình, bạn không đang làm khoa học, bạn chỉ đang "chơi đùa". Experiment tracking là quy trình ghi lại một cách có hệ thống tất cả mọi thứ liên quan đến một lần chạy mô hình để đảm bảo tính tái lập.

**Những gì cần được theo dõi?**

1.  **Code Version**: Mã Git commit hash nào đã được sử dụng để chạy thí nghiệm này?
2.  **Data Version**: Mô hình được huấn luyện trên phiên bản dữ liệu nào? (Thường dùng các công cụ như DVC - Data Version Control).
3.  **Hyperparameters**: Tất cả các tham số đầu vào của mô hình (learning rate, batch size, số layer, v.v.).
4.  **Environment**: Phiên bản của các thư viện (ví dụ: `requirements.txt` hoặc `poetry.lock`), phiên bản Python, HĐH.
5.  **Metrics**: Các chỉ số hiệu suất của mô hình trên tập train/validation/test (loss, accuracy, F1-score, v.v.).
6.  **Artifacts**: Các "hiện vật" được tạo ra, quan trọng nhất là file trọng số của mô hình đã huấn luyện, ngoài ra còn có các biểu đồ, file log, ví dụ dự đoán...

#### Tích hợp MLflow

MLflow là một công cụ mã nguồn mở phổ biến giúp thực hiện tất cả những điều trên.

-   **MLflow Tracking**: Cung cấp API để ghi lại (log) các tham số, metrics và artifacts.
-   **MLflow Projects**: Định dạng để đóng gói code ML.
-   **MLflow Models**: Định dạng chung để đóng gói mô hình.
-   **Model Registry**: Một kho lưu trữ tập trung để quản lý vòng đời của các mô hình.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class MLExperimentTracker:
    """
    Quản lý và theo dõi các thí nghiệm ML với MLflow.
    """
    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        print(f"🔬 MLflow Experiment '{experiment_name}' được thiết lập tại: {tracking_uri}")

    def run_experiment(self, X_train, y_train, X_val, y_val, model_params: dict, run_name: str = None):
        """
        Chạy một thí nghiệm và ghi lại mọi thứ với MLflow.
        """
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"🚀 Bắt đầu run: {run_id}")

            # 1. Log Hyperparameters
            mlflow.log_params(model_params)
            print(f"📝 Đã log tham số: {model_params}")

            # 2. Huấn luyện mô hình
            model = RandomForestRegressor(**model_params, random_state=42)
            model.fit(X_train, y_train)

            # 3. Đánh giá và log Metrics
            y_val_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            mlflow.log_metric("validation_rmse", rmse)
            print(f"📊 Đã log metric: validation_rmse = {rmse:.4f}")

            # 4. Log Artifacts (ví dụ: feature importance)
            # (Thêm code để tạo biểu đồ feature importance và lưu lại)
            # mlflow.log_artifact("feature_importance.png")

            # 5. Log Model
            # "signature" giúp MLflow hiểu input/output của mô hình
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="random-forest-model",
                signature=signature
            )
            print("📦 Đã log mô hình.")

            return run_id, model

# Ví dụ sử dụng
def demonstrate_experiment_tracking():
    tracker = MLExperimentTracker("House Price Prediction")
    
    # Tạo dữ liệu giả
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)])
    y = pd.Series(np.random.rand(100) * 100)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Thí nghiệm 1
    params1 = {"n_estimators": 100, "max_depth": 10}
    tracker.run_experiment(X_train, y_train, X_val, y_val, params1, "n_100_depth_10")
    
    # Thí nghiệm 2
    params2 = {"n_estimators": 200, "max_depth": 5}
    tracker.run_experiment(X_train, y_train, X_val, y_val, params2, "n_200_depth_5")

# Để chạy:
# 1. pip install mlflow
# 2. Chạy `mlflow ui` trong terminal tại thư mục dự án.
# 3. Chạy file Python này.
# 4. Mở trình duyệt và truy cập http://127.0.0.1:5000 để xem kết quả.
```

### 1.2 Đăng ký mô hình (Model Registry)

> **Model Registry** là một kho lưu trữ tập trung, đóng vai trò là **"nguồn chân lý duy nhất" (single source of truth)** cho tất cả các mô hình đã được huấn luyện và sẵn sàng để triển khai. Nó giúp quản lý vòng đời của mô hình một cách có hệ thống.

**Tại sao cần thiết?**
-   **Quản lý phiên bản**: Theo dõi chính xác phiên bản mô hình nào (`v1.2`, `v2.0`) đang chạy ở môi trường nào (`staging`, `production`).
-   **Quản trị (Governance)**: Thiết lập quy trình phê duyệt. Ai có quyền đẩy một mô hình từ `staging` lên `production`? Mô hình cần phải vượt qua những bài kiểm tra nào?
-   **Tái lập và Rollback**: Dễ dàng quay lại một phiên bản cũ hơn nếu phiên bản mới gặp lỗi.

#### Vòng đời mô hình trong Registry
1.  **Development/None**: Mô hình mới được một data scientist huấn luyện xong và đăng ký vào registry. Nó chưa được kiểm duyệt và chưa sẵn sàng cho bất cứ đâu.
2.  **Staging**: Mô hình đã cho thấy kết quả tốt trong thí nghiệm và được "thăng hạng" lên Staging. Tại đây, các kỹ sư sẽ thực hiện các bài kiểm tra tích hợp, kiểm tra hiệu năng (latency, throughput), và đảm bảo nó hoạt động tốt trong một môi trường gần giống production.
3.  **Production**: Sau khi vượt qua tất cả các bài kiểm tra ở Staging, mô hình được phê duyệt và chuyển sang Production. Nó bắt đầu phục vụ traffic thực tế từ người dùng.
4.  **Archived**: Khi một mô hình mới hơn được đưa lên Production, phiên bản cũ sẽ được chuyển sang trạng thái Archived. Nó không còn phục vụ traffic nhưng vẫn được lưu trữ để có thể rollback khi cần hoặc để phân tích lại trong tương lai.

```python
import mlflow
from mlflow.tracking import MlflowClient

class ModelRegistryManager:
    """Quản lý vòng đời mô hình với MLflow Model Registry."""
    def __init__(self, tracking_uri="sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        print(f"📦 Model Registry Manager kết nối tới: {mlflow.get_tracking_uri()}")

    def register_new_version(self, model_name: str, run_id: str):
        """Đăng ký một mô hình mới từ một MLflow run."""
        model_uri = f"runs:/{run_id}/random-forest-model"
        try:
            model_version_details = mlflow.register_model(model_uri, model_name)
            print(f"✅ Đã đăng ký mô hình '{model_name}', phiên bản: {model_version_details.version}")
            return model_version_details
        except Exception as e:
            print(f"❌ Lỗi khi đăng ký mô hình: {e}")
            return None

    def transition_stage(self, model_name: str, version: str, stage: str):
        """Chuyển một phiên bản mô hình sang stage mới."""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True # Tự động đưa các version cũ trong stage này về Archived
            )
            print(f"✅ Đã chuyển mô hình '{model_name}' v{version} sang stage '{stage}'.")
        except Exception as e:
            print(f"❌ Lỗi khi chuyển stage: {e}")

    def get_production_model(self, model_name: str):
        """Tải mô hình đang ở stage Production."""
        try:
            model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
            print(f"✅ Đã tải mô hình '{model_name}' từ stage Production.")
            return model
        except Exception as e:
            print(f"❌ Không tìm thấy mô hình production cho '{model_name}': {e}")
            return None

# Ví dụ sử dụng
# (run_id phải được lấy từ hàm run_experiment ở trên)
# registry_manager = ModelRegistryManager()
# registered_model = registry_manager.register_new_version("house-price-predictor", run_id_cua_ban)
# if registered_model:
#     # Chuyển sang Staging để test
#     registry_manager.transition_stage("house-price-predictor", registered_model.version, "Staging")
#     # Sau khi test xong...
#     registry_manager.transition_stage("house-price-predictor", registered_model.version, "Production")
```

## 📊 2. Phục vụ và Triển khai mô hình (Model Serving & Deployment)

> **Model Serving** là quá trình đưa một mô hình đã huấn luyện vào một môi trường production để nó có thể nhận dữ liệu đầu vào và trả về dự đoán.

### 2.1 Các kiến trúc phục vụ mô hình

Việc lựa chọn kiến trúc phụ thuộc vào yêu cầu của bài toán về độ trễ (latency) và thông lượng (throughput).

1.  **Online Serving (Real-time Serving)**:
    -   **Kịch bản**: Cần dự đoán ngay lập tức cho một yêu cầu đơn lẻ.
    -   **Ví dụ**: Phát hiện gian lận thẻ tín dụng ngay khi giao dịch diễn ra; gợi ý sản phẩm cho người dùng khi họ đang duyệt web.
    -   **Đặc điểm**: Yêu cầu độ trễ rất thấp (low latency).
    -   **Kiến trúc phổ biến**: Triển khai mô hình như một **REST API** (sử dụng FastAPI, Flask) hoặc **gRPC service**.

2.  **Batch Serving (Offline Serving)**:
    -   **Kịch bản**: Cần dự đoán cho một lượng lớn dữ liệu mà không cần kết quả ngay lập tức.
    -   **Ví dụ**: Phân loại email spam cho toàn bộ hòm thư vào ban đêm; dự báo doanh số cho tất cả các cửa hàng vào cuối ngày.
    -   **Đặc điểm**: Ưu tiên thông lượng cao (high throughput) hơn là độ trễ thấp.
    -   **Kiến trúc phổ biến**: Một **job được lập lịch** (scheduled job) chạy định kỳ (ví dụ: dùng Cron, Airflow), đọc dữ liệu từ một kho dữ liệu (data warehouse), thực hiện dự đoán, và lưu kết quả trở lại kho.

3.  **Streaming Serving (Near Real-time)**:
    -   **Kịch bản**: Cần dự đoán trên một dòng dữ liệu (stream) đang chảy liên tục.
    -   **Ví dụ**: Phân tích cảm xúc (sentiment analysis) của các tweet về một chủ đề đang nóng; gợi ý video tiếp theo trên TikTok.
    -   **Đặc điểm**: Cân bằng giữa độ trễ và thông lượng.
    -   **Kiến trúc phổ biến**: Tích hợp mô hình với các nền tảng xử lý stream như **Apache Kafka**, **Spark Streaming**, hoặc **Apache Flink**.

### 2.2 REST API với FastAPI

FastAPI là một lựa chọn hiện đại và hiệu quả để xây dựng API cho mô hình ML nhờ hiệu năng cao (dựa trên Starlette và Pydantic) và khả năng tự động tạo tài liệu API (Swagger UI).

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models để validate input/output
class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str

# Khởi tạo app FastAPI
app = FastAPI(title="ML Prediction Service")

# Tải mô hình khi ứng dụng khởi động
try:
    model_data = joblib.load("path/to/your/model.pkl")
    model = model_data['model']
    MODEL_VERSION = model_data.get('version', '1.0.0')
    logger.info(f"Mô hình phiên bản {MODEL_VERSION} đã được tải.")
except FileNotFoundError:
    model = None
    MODEL_VERSION = "N/A"
    logger.error("File mô hình không tìm thấy!")

@app.get("/health")
def health_check():
    """Kiểm tra sức khỏe của dịch vụ"""
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng.")
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Thực hiện dự đoán từ các feature đầu vào"""
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng.")
    
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        return PredictionResponse(
            prediction=prediction,
            model_version=MODEL_VERSION
        )
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {e}")
        raise HTTPException(status_code=400, detail=f"Dữ liệu đầu vào không hợp lệ: {e}")
```

## 🔄 3. CI/CD/CT cho Machine Learning

Đây là nơi MLOps thực sự tỏa sáng, tự động hóa vòng đời ML.

-   **CI (Continuous Integration - Tích hợp liên tục)**: Giống như trong phát triển phần mềm truyền thống. Mỗi khi có một thay đổi trong code (ví dụ: một pull request), hệ thống sẽ tự động chạy linting, unit test, và kiểm tra chất lượng code.
-   **CD (Continuous Delivery - Giao hàng liên tục)**: Sau khi CI thành công, hệ thống sẽ tự động build các "hiện vật" (ví dụ: Docker image) và triển khai chúng lên môi trường Staging. Sau khi các bài test trên Staging thành công, việc đẩy lên Production có thể cần một bước phê duyệt thủ công.
-   **CT (Continuous Training - Huấn luyện liên tục)**: Đây là điểm độc đáo của MLOps.
    -   **Trigger**: Một quy trình CT có thể được kích hoạt bởi nhiều yếu tố:
        1.  **Có dữ liệu mới**: Hệ thống giám sát phát hiện có một lượng lớn dữ liệu mới.
        2.  **Hiệu suất mô hình giảm sút (Model Decay)**: Mô hình production hoạt động kém đi theo thời gian.
        3.  **Theo lịch trình**: Huấn luyện lại mô hình hàng tuần hoặc hàng tháng.
    -   **Quy trình**: Hệ thống tự động khởi chạy một pipeline để huấn luyện lại mô hình trên dữ liệu mới. Mô hình mới sau đó sẽ được đánh giá. Nếu nó tốt hơn mô hình hiện tại, nó sẽ được đăng ký vào Model Registry và trở thành một "ứng cử viên" cho việc triển khai ra production (thông qua pipeline CD).



## 📈 4. Giám sát và Khả năng quan sát (Monitoring & Observability)

> **Mục tiêu**: Đảm bảo mô hình ML hoạt động đúng như mong đợi trong môi trường production, phát hiện sớm các vấn đề để kịp thời khắc phục. Giám sát không chỉ là theo dõi hiệu suất, mà còn là hiểu được "tại sao" hiệu suất thay đổi.

**Tại sao cần thiết?**
-   Mô hình ML không giống phần mềm truyền thống. Hiệu suất của chúng có thể suy giảm theo thời gian do sự thay đổi của dữ liệu và môi trường.
-   Giám sát giúp phát hiện sớm các vấn đề như **Data Drift** (dữ liệu thay đổi) và **Concept Drift** (mối quan hệ thay đổi).

### 4.1 Giám sát hiệu suất mô hình (Model Performance Monitoring)

-   **Mục tiêu**: Theo dõi các chỉ số hiệu suất kỹ thuật của mô hình trong production.
-   **Cách làm**:
    1.  **Thu thập dự đoán**: Lưu lại tất cả các dự đoán của mô hình trong production.
    2.  **Thu thập nhãn thật (Ground Truth)**: Khi có nhãn thật (thường có độ trễ), so sánh với dự đoán của mô hình.
    3.  **Tính toán Metrics**: Tính toán các metrics phù hợp (Accuracy, F1-score cho phân loại; RMSE, MAE cho hồi quy) trên dữ liệu production.
-   **Thách thức**: Nhãn thật thường không có sẵn ngay lập tức, đòi hỏi chiến lược giám sát có độ trễ.
-   **Trực quan hóa**: Sử dụng các dashboard (Grafana, Kibana) để hiển thị xu hướng của các metric theo thời gian.

### 4.2 Data Drift (Trôi dạt dữ liệu)

-   **Khái niệm**: Xảy ra khi **phân phối của dữ liệu đầu vào (input features)** trong production thay đổi đáng kể so với phân phối dữ liệu mà mô hình đã được huấn luyện.
-   **Ví dụ**: Mô hình dự đoán giá nhà được huấn luyện trên dữ liệu giá nhà ở thành phố lớn, nhưng sau đó lại được dùng để dự đoán giá nhà ở nông thôn.
-   **Tác động**: Có thể làm giảm hiệu suất của mô hình, vì mô hình "chưa bao giờ thấy" loại dữ liệu mới này.
-   **Phát hiện**:
    -   **Thống kê**: Dùng các kiểm định thống kê để so sánh phân phối của từng feature giữa dữ liệu training và dữ liệu production (ví dụ: KS-statistic, Population Stability Index - PSI).
    -   **Machine Learning**: Huấn luyện một mô hình nhỏ để phân loại xem một mẫu dữ liệu đến từ tập training hay tập production. Nếu mô hình này có độ chính xác cao, có nghĩa là đã có Data Drift.

### 4.3 Concept Drift (Trôi dạt khái niệm)

-   **Khái niệm**: Xảy ra khi **mối quan hệ giữa dữ liệu đầu vào và biến mục tiêu (target variable)** thay đổi theo thời gian.
-   **Ví dụ**: Mô hình dự đoán sự hài lòng của khách hàng dựa trên hành vi mua sắm. Sau một chiến dịch marketing lớn, khách hàng có thể mua sắm nhiều hơn nhưng lại ít hài lòng hơn, hoặc các yếu tố trước đây dẫn đến hài lòng nay không còn đúng nữa.
-   **Tác động**: Trực tiếp làm giảm độ chính xác của mô hình, vì "luật chơi" đã thay đổi.
-   **Phát hiện**: Đây là loại drift khó phát hiện hơn Data Drift vì nó đòi hỏi nhãn thật.
    -   Theo dõi hiệu suất mô hình trên dữ liệu production.
    -   Phân tích sai số của mô hình để tìm ra các mẫu lỗi mới.

### 4.4 Cảnh báo và Hành động (Alerting & Action)

-   **Cảnh báo**: Thiết lập ngưỡng cảnh báo cho các metric hiệu suất và các chỉ số Data/Concept Drift. Khi một ngưỡng bị vượt quá, hệ thống sẽ gửi cảnh báo đến đội ngũ MLOps.
-   **Hành động tự động**: Khi có drift đáng kể, có thể kích hoạt các hành động tự động như:
    -   **Huấn luyện lại mô hình (Retraining)**: Sử dụng dữ liệu mới để huấn luyện lại mô hình.
    -   **Switch sang mô hình dự phòng**: Tạm thời chuyển sang một mô hình cũ hơn, đã biết là ổn định.
    -   **Chuyển sang chế độ thủ công**: Nếu độ tin cậy của mô hình quá thấp.

## 📚 Tài liệu tham khảo

### MLOps Fundamentals
- [MLOps: Continuous Delivery for Machine Learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) - Google Cloud
- [The MLOps Community](https://mlops.community/) - Community resources

### Tools và Frameworks
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) - MLflow official docs
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - FastAPI official docs
- [Redis Documentation](https://redis.io/documentation) - Redis official docs
- [Evidently AI](https://evidentlyai.com/) - Mở rộng các công cụ giám sát mô hình

### Best Practices
- [MLOps Best Practices](https://www.databricks.com/blog/2020/12/22/mlops-best-practices.html) - Databricks
- [Production ML Systems](https://www.oreilly.com/library/view/production-machine-learning/9781098106668/) - O'Reilly

## 🎯 Bài tập thực hành

1.  **Experiment Tracking**: Setup MLflow và track multiple experiments.
2.  **Model Registry**: Implement model versioning và stage management.
3.  **Model Serving**: Tạo REST API với FastAPI cho model deployment.
4.  **Data Drift Detection**: Sử dụng thư viện như `Evidently AI` để phát hiện Data Drift trên một bộ dữ liệu giả.
5.  **Performance Monitoring**: Thiết lập một pipeline để theo dõi hiệu suất mô hình và các chỉ số drift, đồng thời cấu hình cảnh báo.

## 🚀 Bước tiếp theo

Sau khi hoàn thành MLOps cơ bản, bạn sẽ:
-   Hiểu sâu về ML lifecycle management.
-   Có khả năng triển khai model serving systems.
-   Biết cách implement CI/CD cho ML.
-   Sẵn sàng học advanced MLOps như Kubernetes deployment và distributed training.

---

*Chúc bạn trở thành MLOps Engineer xuất sắc! 🎉*
# 🧠 Explainable AI (XAI) - Diễn giải và Tin cậy trong AI

> **Mục tiêu**: Hiểu tầm quan trọng của việc diễn giải mô hình, nắm vững các kỹ thuật phổ biến như LIME và SHAP, và nhận thức được các khía cạnh đạo đức liên quan đến AI.

## 📋 Tổng quan nội dung

```mermaid
graph TD
    A[🧠 Explainable AI] --> B[🤔 Tại sao cần XAI?]
    A --> C[⚙️ Các phương pháp diễn giải]
    A --> D[🛠️ Công cụ Model-Agnostic]
    A --> E[⚖️ Đạo đức và Sự công bằng]
    
    B --> B1[Xây dựng lòng tin]
    B --> B2[Debug và cải thiện mô hình]
    B --> B3[Tuân thủ pháp lý (GDPR)]
    B --> B4[Phát hiện và giảm thiểu thiên vị (bias)]
    
    C --> C1[Global vs. Local Explanations]
    C --> C2[Model-Specific vs. Model-Agnostic]
    C --> C3[Intrinsic vs. Post-hoc]
    
    D --> D1[LIME (Local Interpretable Model-agnostic Explanations)]
    D --> D2[SHAP (SHapley Additive exPlanations)]
    D --> D3[Feature Importance]
    
    E --> E1[Fairness (Công bằng)]
    E --> E2[Accountability (Trách nhiệm giải trình)]
    E --> E3[Transparency (Minh bạch)]
```

## 📖 1. Glossary (Định nghĩa cốt lõi)

-   **Interpretability (Khả năng diễn giải nội tại)**: Mức độ mà một người có thể hiểu nguyên nhân dẫn đến một quyết định của mô hình. Các mô hình đơn giản như Linear Regression hay Decision Tree có tính diễn giải nội tại cao.
-   **Explainability (Khả năng giải thích sau hoc)**: Khả năng giải thích các hoạt động bên trong của một mô hình phức tạp (thường là "hộp đen" - black box) bằng một mô hình khác, đơn giản hơn.
-   **Global Explanation**: Giải thích hành vi tổng thể của mô hình. *Ví dụ: "Nhìn chung, diện tích và vị trí là hai yếutoos quan trọng nhất ảnh hưởng đến giá nhà."*
-   **Local Explanation**: Giải thích lý do cho một dự đoán **cụ thể**. *Ví dụ: "Giá của ngôi nhà này được dự đoán là cao vì nó có diện tích lớn, mặc dù vị trí của nó không phải là tốt nhất."*
-   **Model-Agnostic**: Phương pháp có thể áp dụng cho bất kỳ loại mô hình nào (Linear Regression, Random Forest, Neural Network, ...).
-   **Model-Specific**: Phương pháp chỉ hoạt động với một loại mô hình cụ thể (ví dụ: xem xét trọng số của Linear Regression).

---

## 🤔 2. Tại sao XAI lại quan trọng?

Khi các mô hình AI ngày càng phức tạp (như Deep Learning) và được áp dụng vào các lĩnh vực có ảnh hưởng lớn (y tế, tài chính, pháp luật), câu hỏi "Tại sao mô hình lại đưa ra quyết định này?" trở nên cực kỳ quan trọng.

-   **Xây dựng lòng tin**: Người dùng (bác sĩ, nhân viên ngân hàng, khách hàng) sẽ không tin tưởng một hệ thống "hộp đen" nếu không hiểu lý do đằng sau các quyết định của nó.
-   **Debug và Cải thiện**: Khi mô hình dự đoán sai, XAI giúp ta tìm ra *tại sao* nó sai. Có phải do dữ liệu nhiễu, do feature sai, hay do logic của mô hình có vấn đề?
-   **Phát hiện Thiên vị (Bias)**: XAI có thể phơi bày việc mô hình đang dựa vào các thuộc tính nhạy cảm (như giới tính, chủng tộc) để đưa ra quyết định, giúp ta xây dựng các hệ thống công bằng hơn.
-   **Tuân thủ pháp lý**: Nhiều quy định (như GDPR của Châu Âu) yêu cầu "quyền được giải thích" (right to explanation) cho các quyết định tự động.

---

## ⚙️ 3. Thẻ thuật toán - LIME (Local Interpretable Model-agnostic Explanations)

### 1. Bài toán & dữ liệu
- **Bài toán**: Giải thích dự đoán của bất kỳ mô hình "hộp đen" nào (bộ phân loại hoặc hồi quy) bằng cách xấp xỉ cục bộ nó bằng một mô hình có thể diễn giải được (linear model, decision tree).
- **Dữ liệu**: Một mẫu dữ liệu đơn lẻ mà bạn muốn giải thích dự đoán của nó, cùng với mô hình "hộp đen" đã huấn luyện.
- **Ứng dụng**: Giải thích các dự đoán cho ảnh, văn bản, dữ liệu bảng.

### 2. Mô hình & công thức
- **Ý tưởng cốt lõi**: Mô hình phức tạp có thể được xấp xỉ bằng một mô hình đơn giản hơn (như Linear Regression) trong một vùng lân cận cục bộ của điểm dữ liệu cần giải thích.
- **Công thức (tổng quát)**: LIME tối thiểu hóa hàm mất mát:
  $$ \xi(x) = \operatorname*{argmin}_{g \in \mathcal{G}} \mathcal{L}(f, g, \pi_x) + \Omega(g) $$
  Trong đó:
  -   $f$: Mô hình "hộp đen" gốc.
  -   $g$: Mô hình có thể diễn giải được (linear, tree).
  -   $\pi_x$: Hàm trọng số thể hiện khoảng cách của mẫu được nhiễu đến mẫu gốc $x$.
  -   $\mathcal{L}(f, g, \pi_x)$: Hàm đo lường mức độ $g$ xấp xỉ $f$ trong vùng lân cận của $x$.
  -   $\Omega(g)$: Hàm độ phức tạp của mô hình $g$ (ví dụ: số feature trong linear model).

### 3. Loss & mục tiêu
- **Mục tiêu**: Tìm một mô hình $g$ đơn giản, có thể diễn giải được, xấp xỉ tốt mô hình $f$ phức tạp trong vùng lân cận của mẫu $x$ cần giải thích.

### 4. Tối ưu hoá & cập nhật
- **Algorithm**:
  1.  Tạo các mẫu dữ liệu nhiễu xung quanh mẫu gốc $x$.
  2.  Thu thập dự đoán của mô hình $f$ cho các mẫu nhiễu này.
  3.  Tính trọng số cho các mẫu nhiễu dựa trên khoảng cách của chúng đến $x$.
  4.  Huấn luyện một mô hình $g$ đơn giản (ví dụ: Linear Regression) trên các mẫu nhiễu và dự đoán của $f$, có tính đến trọng số.
  5.  Các hệ số của $g$ chính là lời giải thích.

### 5. Hyperparams
- **Số mẫu nhiễu**: Số lượng mẫu được tạo ra xung quanh mẫu gốc.
- **Kernel width**: Phạm vi của hàm trọng số $\pi_x$.
- **Số feature trong mô hình diễn giải**: Giúp kiểm soát độ phức tạp của $g$.

### 6. Độ phức tạp
- **Time**: Phụ thuộc vào số mẫu nhiễu và thời gian dự đoán của mô hình "hộp đen".
- **Space**: Không đáng kể.

### 7. Metrics đánh giá
- **Độ tin cậy của giải thích**: LIME không có metric nội tại, cần đánh giá qua trực quan hóa và kiểm tra.
- **Local fidelity**: Kiểm tra xem mô hình $g$ có thực sự xấp xỉ tốt $f$ trong vùng cục bộ không.

### 8. Ưu / Nhược điểm
**Ưu điểm**:
-   **Model-Agnostic**: Áp dụng được cho mọi mô hình.
-   **Local Explanation**: Cung cấp giải thích cho từng dự đoán cụ thể.
-   Dễ hiểu và trực quan hóa.

**Nhược điểm**:
-   **Tính không ổn định (Instability)**: Các lời giải thích có thể thay đổi đáng kể nếu các mẫu nhiễu được tạo ra khác nhau một chút.
-   **Phạm vi cục bộ**: Giải thích chỉ có giá trị trong một vùng nhỏ, không thể khái quát hóa toàn cục.
-   Cần xác định các hyperparameter (số mẫu nhiễu, kernel width).

### 9. Bẫy & mẹo
- **Bẫy**: Chọn số mẫu nhiễu quá ít hoặc kernel width quá lớn có thể dẫn đến giải thích sai lệch.
- **Mẹo**: Kết hợp với Domain Knowledge để kiểm tra tính hợp lý của giải thích.
- **Mẹo**: Luôn trực quan hóa kết quả để hiểu rõ hơn.

### 10. Pseudocode:
```python
def LIME_explanation(model, instance, num_perturbations, feature_names, kernel_width):
    # 1. Generate perturbed samples
    perturbed_samples, distances = generate_samples_around_instance(instance, num_perturbations, kernel_width)
    
    # 2. Get predictions from black-box model
    predictions = model.predict(perturbed_samples)
    
    # 3. Compute weights based on distance
    weights = calculate_weights(distances)
    
    # 4. Train an interpretable model (e.g., Weighted Linear Regression)
    interpretable_model = train_weighted_linear_model(perturbed_samples, predictions, weights)
    
    # 5. Extract explanation (e.g., coefficients)
    explanation = get_coefficients(interpretable_model, feature_names)
    
    return explanation
```

### 11. Code mẫu (LIME cho mô hình phân loại văn bản)
```python
import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# 1. Load data và train mô hình đơn giản
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

vectorizer = TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(train_vectors, newsgroups_train.target)

# 2. Chọn một instance để giải thích
idx = 8
text_instance = newsgroups_test.data[idx]
true_class = newsgroups_test.target_names[newsgroups_test.target[idx]]
predicted_class = newsgroups_test.target_names[rf_model.predict(test_vectors[idx])[0]]

print(f"Mẫu văn bản:\n{text_instance}")
print(f"Lớp thật: {true_class}, Lớp dự đoán: {predicted_class}")

# 3. Khởi tạo LIME Explainer
# Hàm predict_proba của mô hình cần được cung cấp cho LIME
c = make_pipeline(vectorizer, rf_model)
explainer = lime.lime_text.LimeTextExplainer(
    class_names=newsgroups_train.target_names,
    split_expression=r'\W+', # Tách từ theo khoảng trắng/ký tự không phải từ
    random_state=42
)

# 4. Giải thích dự đoán
num_features = 10 # Số lượng từ quan trọng muốn hiển thị
explanation = explainer.explain_instance(
    text_instance, 
    c.predict_proba, 
    num_features=num_features, 
    labels=(rf_model.predict(test_vectors[idx])[0],) # Chỉ giải thích cho lớp dự đoán
)

print("\n--- Giải thích LIME ---")
# Các từ có trọng số dương đẩy dự đoán về lớp mục tiêu, âm đẩy ra xa
for word, weight in explanation.as_list():
    print(f"'{word}': {weight:.4f}")

# explanation.show_in_notebook(text=True) # Dùng trong Jupyter Notebook để trực quan hóa
```

### 12. Checklist kiểm tra nhanh:
- [ ] LIME có được áp dụng cho một mẫu cụ thể không?
- [ ] Số feature được hiển thị có phù hợp không?
- [ ] Giải thích có phù hợp với kiến thức nghiệp vụ không?
- [ ] Các tham số (num_features, kernel_width) có được điều chỉnh để giải thích tốt nhất không?

Đây là các công cụ mạnh mẽ nhất vì chúng có thể được áp dụng cho bất kỳ mô hình nào sau khi đã huấn luyện xong.

### 4.1 LIME (Local Interpretable Model-agnostic Explanations)

-   **Tư tưởng cốt lõi**: "Mặc dù một mô hình phức tạp có thể có ranh giới quyết định rất ngoằn ngoèo trên toàn cục, nhưng ở một khu vực **cục bộ (local)** rất nhỏ xung quanh một điểm dữ liệu, ranh giới đó có thể được xấp xỉ bằng một mô hình đơn giản (như một đường thẳng)."
-   **Quy trình hoạt động (để giải thích một dự đoán)**:
    1.  **Chọn một mẫu dữ liệu** bạn muốn giải thích (ví dụ: một khách hàng cụ thể được dự đoán là sẽ churn).
    2.  **Tạo dữ liệu giả (Perturbation)**: LIME tạo ra hàng trăm/nghìn mẫu dữ liệu mới bằng cách thay đổi một chút các feature của mẫu gốc (ví dụ: thay đổi `monthly_charges` một chút, hoặc xóa một vài từ trong một câu văn bản).
    3.  **Lấy dự đoán từ mô hình hộp đen**: Đưa tất cả các mẫu dữ liệu giả này qua mô hình phức tạp của bạn để lấy dự đoán của nó.
    4.  **Huấn luyện một mô hình đơn giản**: Bây giờ, LIME huấn luyện một mô hình tuyến tính đơn giản (có thể diễn giải được) để học cách ánh xạ từ các mẫu dữ liệu giả đến dự đoán của mô hình hộp đen. Các mẫu giả ở gần mẫu gốc sẽ được gán trọng số cao hơn.
    5.  **Diễn giải mô hình đơn giản**: Các trọng số của mô hình tuyến tính này chính là lời giải thích. Một trọng số dương lớn cho một feature có nghĩa là feature đó đã "đẩy" dự đoán lên cao, và ngược lại.

-   **Kết quả**: "Dự đoán cho khách hàng này là 'Churn' **bởi vì** `contract_type` là 'Month-to-month' (đóng góp +0.4) và `tenure` thấp (đóng góp +0.3), mặc dù `monthly_charges` không cao (đóng góp -0.1)."

#### Ví dụ Code: Giải thích phân loại ảnh với LIME

Đây là ví dụ sử dụng LIME để giải thích dự đoán của một mô hình phân loại ảnh (ví dụ: InceptionV3 trên ImageNet). LIME sẽ chỉ ra những vùng pixel nào trong ảnh đã đóng góp nhiều nhất vào dự đoán của mô hình.

```python
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

# Thư viện LIME
import lime
from lime import lime_image

# PyTorch
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# 1. Tải mô hình và tiền xử lý ảnh
# Tải mô hình InceptionV3 đã được huấn luyện trước trên ImageNet
model = models.inception_v3(pretrained=True, aux_logits=True)
model.eval() # Chuyển sang chế độ đánh giá

# Tiền xử lý ảnh cho InceptionV3
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. Tải và xử lý nhãn ImageNet
LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
response = requests.get(LABELS_URL)
labels_map = response.json()
idx2label = [labels_map[str(k)][1] for k in range(len(labels_map))]

# 3. Định nghĩa hàm dự đoán của mô hình cho LIME
# LIME yêu cầu hàm predict_proba trả về xác suất cho tất cả các lớp
def predict_fn(images):
    # images là một mảng numpy (num_samples, H, W, C)
    # Cần chuyển đổi về tensor và định dạng (num_samples, C, H, W)
    images_tensor = torch.stack([preprocess(Image.fromarray((img * 255).astype(np.uint8))) for img in images])
    with torch.no_grad():
        logits = model(images_tensor)
    if isinstance(logits, tuple): # InceptionV3 có aux_logits
        logits = logits[0]
    return F.softmax(logits, dim=1).numpy()

# 4. Tải ảnh mẫu
img_url = "https://raw.githubusercontent.com/marcotcr/lime/master/doc/notebooks/7_5.png"
response = requests.get(img_url)
img_original = Image.open(BytesIO(response.content)).convert('RGB')
img_np = np.array(img_original) / 255.0 # Chuyển về float [0, 1]

# Dự đoán ban đầu của mô hình
logits_orig = model(preprocess(img_original).unsqueeze(0))
if isinstance(logits_orig, tuple):
    logits_orig = logits_orig[0]
pred_class_idx = torch.argmax(logits_orig).item()
pred_class_name = idx2label[pred_class_idx]
print(f"Mô hình dự đoán: {pred_class_name} (Class ID: {pred_class_idx})")

# 5. Khởi tạo LIMEImageExplainer
explainer = lime_image.LimeImageExplainer(random_state=42)

# 6. Giải thích dự đoán của mô hình
# num_samples: số lượng ảnh nhiễu để tạo ra
# batch_size: số lượng ảnh đưa vào predict_fn cùng lúc
explanation = explainer.explain_instance(
    img_np, 
    predict_fn, 
    top_labels=5, 
    hide_color=0, 
    num_samples=1000, 
    batch_size=50
)

# 7. Trực quan hóa kết quả
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=False, 
    num_features=10, 
    hide_rest=False
)

# Vẽ ảnh gốc và giải thích
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(img_np)
ax1.set_title("Ảnh gốc")
ax1.axis('off')

# LIME explanation (vùng ảnh quan trọng được highlight)
ax2.imshow(mark_boundaries(temp / 2 + 0.5, mask))
ax2.set_title(f"LIME giải thích cho: {idx2label[explanation.top_labels[0]]}")
ax2.axis('off')
plt.tight_layout()
plt.show()

# Hiển thị các giải thích chi tiết hơn cho các lớp khác
# for label in explanation.top_labels:
#     print(f"\nGiải thích cho lớp '{idx2label[label]}':")
#     image, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=5, hide_rest=True)
#     plt.imshow(image / 2 + 0.5)
#     plt.title(f"LIME cho lớp: {idx2label[label]}")
#     plt.axis('off')
#     plt.show()
```

### 4.2 SHAP (SHapley Additive exPlanations)

-   **Tư tưởng cốt lõi**: Dựa trên **Giá trị Shapley** từ lý thuyết trò chơi hợp tác. SHAP tính toán sự "đóng góp" công bằng của mỗi feature vào việc tạo ra dự đoán cuối cùng của mô hình. Giá trị Shapley cho mỗi feature là mức đóng góp trung bình mà feature đó mang lại cho dự đoán trên tất cả các kết hợp (coalitions) có thể có của các feature.
-   **Câu hỏi nó trả lời**: "Giá trị của feature X đã làm thay đổi dự đoán cuối cùng bao nhiêu so với dự đoán trung bình (baseline) của mô hình?"
-   **Cách hoạt động (trực quan)**:
    -   Để tính đóng góp của feature `tuổi` cho một dự đoán cụ thể, SHAP xem xét tất cả các tập con feature có thể có.
    -   Nó so sánh dự đoán của mô hình khi có feature `tuổi` và khi không có feature `tuổi` (thường được thay thế bằng giá trị trung bình hoặc ngẫu nhiên từ dữ liệu khác) trong mọi bối cảnh kết hợp feature khác nhau.
    -   Đóng góp của `tuổi` được tính là sự thay đổi trung bình trong dự đoán trên tất cả các bối cảnh này.
-   **Ưu điểm so với LIME**:
    -   **Nền tảng lý thuyết vững chắc**: Dựa trên giá trị Shapley, có các thuộc tính toán học tốt (local accuracy, missingness, consistency) đảm bảo sự công bằng và nhất quán của giải thích.
    -   **Giải thích toàn cục và cục bộ**: SHAP có thể cung cấp cả giải thích cho từng dự đoán riêng lẻ (local) và tóm tắt tầm quan trọng của feature trên toàn bộ tập dữ liệu (global) một cách nhất quán.
-   **Các loại biểu đồ phổ biến**:
    -   **Force Plot**: Trực quan hóa các feature "đẩy" dự đoán lên hoặc xuống so với baseline cho một mẫu cụ thể.
    -   **Summary Plot**: Tổng hợp tầm quan trọng và hướng ảnh hưởng của tất cả các feature trên nhiều mẫu, cho thấy bức tranh toàn cảnh về cách mô hình hoạt động.
    -   **Dependence Plot**: Hiển thị tác động của một feature lên dự đoán của mô hình, và cách tác động này thay đổi khi một feature khác thay đổi.

#### Ví dụ Code: Giải thích mô hình bảng với SHAP

Đây là ví dụ sử dụng SHAP để giải thích dự đoán của một mô hình `RandomForestClassifier` trên bộ dữ liệu `Iris`.

```python
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 1. Load dữ liệu và huấn luyện mô hình
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Chọn một instance để giải thích
instance_idx = 5 # Chọn mẫu thứ 5 trong tập test
instance_to_explain = X_test[instance_idx]
true_class = target_names[y_test[instance_idx]]
predicted_class_idx = model.predict(instance_to_explain.reshape(1, -1))[0]
predicted_class_name = target_names[predicted_class_idx]

print(f"Mẫu dữ liệu cần giải thích: {instance_to_explain}")
print(f"Lớp thật: {true_class}, Lớp dự đoán: {predicted_class_name}")

# 3. Khởi tạo SHAP Explainer
# Đối với tree-based models, TreeExplainer hiệu quả hơn
explainer = shap.TreeExplainer(model)

# Tính toán SHAP values cho mẫu cần giải thích
# shap_values là một list, mỗi phần tử là SHAP values cho một lớp
shap_values = explainer.shap_values(instance_to_explain)

# 4. Trực quan hóa kết quả (Force Plot)
# Force plot cho thấy cách các feature đẩy dự đoán từ baseline (expected value) đến giá trị cuối cùng.
print("\n--- SHAP Force Plot cho mẫu cụ thể ---")
# Index của lớp dự đoán
class_id_to_explain = predicted_class_idx
shap.initjs() # Cần thiết để hiển thị interactive plot trong Jupyter
shap.force_plot(
    explainer.expected_value[class_id_to_explain], 
    shap_values[class_id_to_explain], 
    instance_to_explain, 
    feature_names=feature_names,
    matplotlib=True # Force plot to render as static Matplotlib figure
)
plt.title(f"SHAP Force Plot cho lớp: {predicted_class_name}")
plt.tight_layout()
plt.show()

# 5. SHAP Summary Plot (Global Explanation)
# Hiển thị tầm quan trọng và hướng ảnh hưởng của các feature trên toàn bộ tập dữ liệu
print("\n--- SHAP Summary Plot (Global) ---")
shap_values_test = explainer.shap_values(X_test)
# Đối với multi-class, thường chọn shap_values cho lớp dương (hoặc lớp dự đoán)
shap.summary_plot(
    shap_values_test, 
    X_test, 
    feature_names=feature_names, 
    class_names=target_names,
    show=False # Don't show immediately for better control
)
plt.title("SHAP Summary Plot (Iris Dataset)")
plt.tight_layout()
plt.show()

# 6. SHAP Dependence Plot (Feature Interaction)
# Hiển thị mối quan hệ giữa một feature và dự đoán, có thể thấy tương tác với feature khác
print("\n--- SHAP Dependence Plot (Feature Interaction) ---")
# Ví dụ: Tác động của 'petal length (cm)' lên dự đoán, tương tác với 'petal width (cm)'
shap.dependence_plot(
    "petal length (cm)", 
    shap_values_test[predicted_class_idx], 
    X_test, 
    feature_names=feature_names,
    interaction_index="petal width (cm)",
    show=False
)
plt.title(f"SHAP Dependence Plot: petal length (cm) vs petal width (cm) for {predicted_class_name}")
plt.tight_layout()
plt.show()
```
---

## ⚖️ 4. Đạo đức và Sự công bằng trong AI (AI Ethics & Fairness)

XAI không chỉ là một công cụ kỹ thuật mà còn là nền tảng cho việc xây dựng các hệ thống AI có trách nhiệm.

-   **Fairness (Công bằng)**: XAI giúp phát hiện xem mô hình có đang đưa ra các quyết định bất lợi một cách có hệ thống cho một nhóm người cụ thể nào đó hay không (ví dụ: từ chối cho vay đối với một giới tính hoặc chủng tộc nhất định).
-   **Accountability (Trách nhiệm giải trình)**: Khi một hệ thống AI gây ra lỗi (ví dụ: xe tự lái gây tai nạn), XAI giúp truy vết và xác định thành phần nào trong mô hình đã gây ra quyết định sai lầm đó.
-   **Transparency (Minh bạch)**: Cung cấp sự minh bạch về cách các quyết định được đưa ra, giúp xây dựng lòng tin và cho phép sự giám sát từ bên ngoài.

## 🎯 5. Bài tập và Tham khảo

### 5.1 Bài tập thực hành
1.  **Phân tích Feature Importance**: Huấn luyện một mô hình RandomForest và sử dụng thuộc tính `feature_importances_` để tìm ra các feature quan trọng nhất. So sánh kết quả này với kết quả từ SHAP.
2.  **Giải thích dự đoán cục bộ**: Chọn một vài dự đoán đúng và một vài dự đoán sai từ mô hình của bạn. Sử dụng LIME và SHAP (Force Plot) để giải thích tại sao mô hình lại đưa ra các quyết định đó. Phân tích xem lời giải thích có hợp lý không.
3.  **Phân tích Bias**: Sử dụng SHAP Summary Plot để xem một feature nhạy cảm (ví dụ: `Sex` trong bộ dữ liệu Titanic) có ảnh hưởng như thế nào đến đầu ra của mô hình trên toàn bộ tập dữ liệu.

### 5.2 Tài liệu tham khảo
-   **Thư viện**: `lime`, `shap`, `eli5`, `interpret-community`.
-   **Sách**: "Interpretable Machine Learning" của Christoph Molnar (một nguồn tài liệu tuyệt vời và miễn phí).
-   **Bài báo quan trọng**:
    -   "Why Should I Trust You?": Explaining the Predictions of Any Classifier" (LIME paper).
    -   "A Unified Approach to Interpreting Model Predictions" (SHAP paper).

---
*Chúc bạn học tập hiệu quả! 🚀*

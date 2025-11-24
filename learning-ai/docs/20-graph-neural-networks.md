# 🕸️ Graph Neural Networks (GNNs) - Mạng nơ-ron trên đồ thị

> **Mục tiêu**: Hiểu các khái niệm cơ bản và kiến trúc cốt lõi của Mạng nơ-ron đồ thị (GNNs), một lĩnh vực đang phát triển nhanh chóng cho việc học trên dữ liệu có cấu trúc quan hệ.

## 📋 Tổng quan nội dung

```mermaid
graph TD
    A[🕸️ Graph Neural Networks] --> B[🧠 Nền tảng đồ thị]
    A --> C[🔧 Cơ chế Message Passing]
    A --> D[🏛️ Các kiến trúc GNN]
    A --> E[🎯 Các tác vụ trên đồ thị]
    
    B --> B1[Nodes, Edges, Adjacency Matrix]
    B --> B2[Node Features, Edge Features]
    B --> B3[Graph Isomorphism Problem]
    
    C --> C1[Aggregate Function]
    C --> C2[Update Function]
    C --> C3[Permutation Invariance/Equivariance]
    
    D --> D1[Graph Convolutional Networks (GCN)]
    D --> D2[GraphSAGE]
    D --> D3[Graph Attention Networks (GAT)]
    
    E --> E1[Node Classification]
    E --> E2[Link Prediction]
    E --> E3[Graph Classification]
```

## 📚 1. Bảng ký hiệu (Notation)

- **Graph (\$\mathcal{G}\$)**: Một đồ thị được định nghĩa bởi tập hợp các đỉnh và cạnh, \$\mathcal{G} = (\mathcal{V}, \mathcal{E})
- **Node/Vertex ($v \in \mathcal{V}$)**: Một đỉnh trong đồ thị.
- **Edge ($e_{ij} \in \mathcal{E}$)**: Một cạnh nối giữa đỉnh $i$ và đỉnh $j$.
- **Adjacency Matrix (\$\mathbf{A}\$)**: Ma trận kề, \$\mathbf{A}_{ij} = 1\$ nếu có cạnh nối giữa $i$ và $j$, ngược lại bằng 0.
- **Node Feature Matrix (\$\mathbf{X}\$)**: Ma trận đặc trưng của các đỉnh, mỗi hàng là một vector đặc trưng của một đỉnh.
- **Node Embedding/Hidden State (\$\mathbf{h}_v^{(k)}\$)**: Biểu diễn vector của đỉnh $v$ tại layer thứ $k$.

## 📖 2. Glossary (Định nghĩa cốt lõi)

-   **Graph**: Một cấu trúc dữ liệu bao gồm các **đỉnh (nodes)** và các **cạnh (edges)** nối giữa chúng. Dùng để mô hình hóa các mối quan hệ.
-   **Node Classification**: Bài toán dự đoán nhãn cho từng đỉnh trong đồ thị. *Ví dụ: Phân loại một người dùng trong mạng xã hội là bot hay người thật.*
-   **Link Prediction**: Bài toán dự đoán xem liệu có một cạnh tồn tại giữa hai đỉnh hay không. *Ví dụ: Gợi ý kết bạn trong mạng xã hội.*
-   **Graph Classification**: Bài toán dự đoán nhãn cho toàn bộ đồ thị. *Ví dụ: Phân loại một phân tử hóa học là độc hại hay không.*
-   **Permutation Invariance**: Tính chất của một hàm mà output không thay đổi khi thứ tự của các input bị hoán vị. Các hàm `sum`, `mean`, `max` có tính chất này, rất quan trọng cho các hàm `Aggregate` trong GNN.

---

## 🧠 3. Nền tảng: Tại sao cần GNNs?

Các mạng nơ-ron truyền thống như CNN hay RNN được thiết kế cho dữ liệu có cấu trúc dạng lưới (grid-like) như ảnh (2D grid) hoặc văn bản (1D sequence). Tuy nhiên, rất nhiều dữ liệu trong thế giới thực không có cấu trúc này, mà tồn tại dưới dạng đồ thị với các mối quan hệ phức tạp:
-   **Mạng xã hội**: Người dùng là các đỉnh, mối quan hệ bạn bè là các cạnh.
-   **Hóa học**: Các nguyên tử là đỉnh, liên kết hóa học là cạnh.
-   **Hệ thống gợi ý**: Người dùng và sản phẩm là các đỉnh, hành vi mua hàng/đánh giá là các cạnh.

GNNs ra đời để học trực tiếp trên cấu trúc đồ thị này, cho phép mô hình hóa các mối quan hệ và tương tác một cách tự nhiên.

## 🔧 4. Cơ chế cốt lõi: Message Passing (Lan truyền thông điệp)

Hầu hết các kiến trúc GNN hiện đại đều tuân theo một khuôn khổ chung gọi là **Message Passing** hoặc **Neighborhood Aggregation**. Ý tưởng rất trực quan: "Một đỉnh được định nghĩa bởi những hàng xóm của nó."

Một layer GNN thực hiện việc cập nhật biểu diễn (embedding) của mỗi đỉnh thông qua 3 bước:

1.  **Gather (Thu thập)**: Với mỗi đỉnh, nó "nhìn" sang các đỉnh hàng xóm và thu thập các vector đặc trưng của chúng.
2.  **Aggregate (Tổng hợp)**: Đỉnh đó tổng hợp tất cả thông tin từ hàng xóm thành một "thông điệp" duy nhất. Phép tổng hợp này phải có tính **hoán vị bất biến (permutation invariant)**, vì các hàng xóm không có thứ tự cố định. Các hàm phổ biến là `sum`, `mean`, hoặc `max`.
3.  **Update (Cập nhật)**: Đỉnh đó sử dụng thông điệp tổng hợp từ hàng xóm và vector đặc trưng hiện tại của chính nó để tính toán ra vector đặc trưng mới cho layer tiếp theo. Bước này thường bao gồm một phép biến đổi tuyến tính và một hàm kích hoạt phi tuyến.

Khi xếp chồng nhiều layer GNN lên nhau, một đỉnh có thể tổng hợp thông tin từ các hàng xóm ngày càng xa hơn (hàng xóm của hàng xóm, v.v.), cho phép nó học được các đặc trưng cấu trúc phức tạp hơn.

---

## ⚙️ 5. Thẻ thuật toán - GCN (Graph Convolutional Network)

### 1. Bài toán & dữ liệu
- **Bài toán**: Học biểu diễn (embedding) cho các đỉnh trong đồ thị để thực hiện các tác vụ như phân loại đỉnh (Node Classification), phân loại đồ thị (Graph Classification).
- **Dữ liệu**: Đồ thị $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, với $\mathbf{X}$ là ma trận đặc trưng đỉnh, $\mathbf{A}$ là ma trận kề.
- **Ứng dụng**: Phân loại bài báo khoa học, phân loại người dùng trong mạng xã hội, phân tích mạng lưới.

### 2. Mô hình & công thức
- **Ý tưởng cốt lõi**: Tổng hợp thông tin từ các đỉnh hàng xóm và thông tin của chính đỉnh đó, sau đó biến đổi tuyến tính và áp dụng hàm kích hoạt phi tuyến.
- **Công thức một layer GCN**:
  $$ \mathbf{H}^{(l+1)} = \sigma \left( \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right) $$
  Trong đó:
  -   $\mathbf{H}^{(l)}$: Ma trận biểu diễn đỉnh (embeddings) của layer $l$. $\mathbf{H}^{(0)} = \mathbf{X}$.
  -   $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$: Ma trận kề với các vòng lặp tự nối (self-loops).
  -   $\tilde{\mathbf{D}}$: Ma trận bậc (degree matrix) của $\tilde{\mathbf{A}}$.
  -   $\mathbf{W}^{(l)}$: Ma trận trọng số học được của layer $l$.
  -   $\sigma$: Hàm kích hoạt phi tuyến (ví dụ: ReLU).

### 3. Loss & mục tiêu
- **Mục tiêu**: Tối thiểu hóa hàm mất mát trên các nhãn đã biết (ví dụ: Cross-entropy cho phân loại đỉnh).
- **Loss**: Phụ thuộc vào tác vụ. Đối với Node Classification, thường là cross-entropy chỉ tính trên các đỉnh đã được gán nhãn.

### 4. Tối ưu hoá & cập nhật
- **Algorithm**: Lan truyền ngược (Backpropagation) để cập nhật các ma trận trọng số $\mathbf{W}^{(l)}$.
- **Optimizer**: Thường là Adam hoặc SGD.

### 5. Hyperparams
- **Số layer GCN**: Thường ít (2-3 layer) do vấn đề over-smoothing.
- **Learning Rate**: 0.01-0.001.
- **Hidden dimension**: Kích thước vector biểu diễn ẩn cho mỗi đỉnh.

### 6. Độ phức tạp
- **Time**: $O(|\mathcal{E}| D L)$ với $|\mathcal{E}|$ là số cạnh, $D$ là số chiều embedding, $L$ là số layer. Có thể tốn kém với đồ thị lớn.
- **Space**: $O(|\mathcal{V}| D + |\mathcal{E}|)$ với $|\mathcal{V}|$ là số đỉnh.

### 7. Metrics đánh giá
- **Node Classification**: Accuracy, F1-score.
- **Graph Classification**: Accuracy.

### 8. Ưu / Nhược điểm
**Ưu điểm**:
-   Học biểu diễn mạnh mẽ cho dữ liệu đồ thị.
-   Tận dụng được cấu trúc cục bộ của đồ thị.
-   Dễ hiểu và triển khai (so với các phương pháp dựa trên spectral graph theory).

**Nhược điểm**:
-   **Transductive**: Thường chỉ hoạt động tốt trên các đồ thị đã biết trong quá trình training (không khả năng quy nạp trên các đỉnh mới hoặc đồ thị mới).
-   **Over-smoothing**: Khi stacking quá nhiều layer GCN, các biểu diễn đỉnh có xu hướng trở nên giống nhau.
-   **Chưa giải quyết tốt bài toán đồ thị lớn**: Yêu cầu toàn bộ ma trận kề, tốn bộ nhớ.

### 9. Bẫy & mẹo
- **Bẫy**: Over-smoothing khi dùng quá nhiều layer.
- **Bẫy**: Khó khăn với đồ thị lớn do yêu cầu ma trận kề.
- **Mẹo**: Sử dụng Dropout để chống overfitting.
- **Mẹo**: Sử dụng Early Stopping.
- **Mẹo**: Feature Engineering cho đỉnh và cạnh.

### 10. Pseudocode (một layer GCN):
```python
# H(l) là ma trận đặc trưng của các đỉnh ở layer l
# A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2) (ma trận kề được chuẩn hóa)
# W(l) là ma trận trọng số học được

H_next = ReLU(A_hat @ H_current @ W)
# (Trong PyTorch Geometric, A_hat được xử lý hiệu quả hơn)
```

### 11. Code mẫu (GCN Layer với PyTorch Geometric)
```python
import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super().__init__()
        # GCNConv là một lớp GCN
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # x: ma trận đặc trưng của các đỉnh (num_nodes, num_node_features)
        # edge_index: ma trận cạnh (2, num_edges)
        
        x = self.conv1(x, edge_index)
        x = self.relu(x) # Áp dụng hàm kích hoạt ReLU
        x = self.conv2(x, edge_index)
        return x

# Ví dụ khởi tạo và sử dụng
# from torch_geometric.datasets import Planetoid
# from torch_geometric.data import Data
#
# # Tải một dataset đồ thị (ví dụ: Cora)
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data = dataset[0] # Lấy đồ thị đầu tiên
#
# # Khởi tạo mô hình GCN
# model = GCN(num_node_features=dataset.num_node_features, 
#             hidden_channels=16, 
#             num_classes=dataset.num_classes)
#
# # Forward pass
# # output = model(data.x, data.edge_index)
# # print(output.shape) # (num_nodes, num_classes)
```

### 12. Checklist kiểm tra nhanh:
- [ ] Dữ liệu đồ thị có được biểu diễn đúng (feature đỉnh, ma trận kề)?
- [ ] Số layer GCN có phù hợp (tránh over-smoothing)?
- [ ] Hàm mất mát và optimizer có được chọn đúng cho tác vụ không?
- [ ] Có thể trực quan hóa embeddings để hiểu những gì GNN học được không?

---

## 🏛️ 6. Các kiến trúc GNN phổ biến

### 6.1 Graph Convolutional Networks (GCN)

-   **Tư tưởng**: GCN là một trong những kiến trúc GNN tiên phong, mở rộng ý tưởng của phép tích chập (convolution) từ dữ liệu dạng lưới (ảnh) sang dữ liệu đồ thị. Nó đơn giản hóa các phương pháp dựa trên miền tần số (spectral graph theory) thành một cách tiếp cận dựa trên lan truyền thông điệp trong miền không gian (spatial domain).
-   **Cách hoạt động**: Mỗi layer GCN tính toán biểu diễn mới cho một đỉnh bằng cách tổng hợp thông tin từ chính nó và các đỉnh hàng xóm. Phép tổng hợp này thường là một trung bình có trọng số, nơi các đỉnh có bậc cao hơn được chuẩn hóa để tránh ảnh hưởng quá mức.
-   **Công thức một layer GCN (Spectral perspective simplified to Spatial)**:
    $$ \mathbf{H}^{(l+1)} = \sigma \left( \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right) $$
    Trong đó:
    -   $\mathbf{H}^{(l)}$: Ma trận biểu diễn đỉnh (embeddings) của layer $l$. $\mathbf{H}^{(0)} = \mathbf{X}$ (ma trận đặc trưng ban đầu).
    -   $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$: Ma trận kề $\mathbf{A}$ được thêm các vòng lặp tự nối (self-loops) $\mathbf{I}$ để đỉnh $v$ cũng tổng hợp thông tin từ chính nó.
    -   $\tilde{\mathbf{D}}$: Ma trận bậc (degree matrix) của $\tilde{\mathbf{A}}$. Việc chuẩn hóa $\tilde{\mathbf{D}}^{-\frac{1}{2}}$ giúp ổn định quá trình huấn luyện và tránh các giá trị embedding bị thổi phồng.
    -   $\mathbf{W}^{(l)}$: Ma trận trọng số học được của layer $l$.
    -   $\sigma$: Hàm kích hoạt phi tuyến (ví dụ: ReLU).
-   **Ưu điểm**:
    -   Đơn giản, dễ hiểu và dễ triển khai.
    -   Hiệu quả cho nhiều tác vụ trên đồ thị bán giám sát (semi-supervised).
-   **Nhược điểm**:
    -   **Transductive**: Chủ yếu hoạt động trên các đồ thị đã biết trong quá trình training.
    -   **Over-smoothing**: Khi xếp chồng nhiều layer, biểu diễn của các đỉnh có xu hướng trở nên rất giống nhau, làm mất đi khả năng phân biệt.
    -   **Không mở rộng (Not scalable)**: Yêu cầu toàn bộ ma trận kề, tốn bộ nhớ và tính toán cho đồ thị lớn.

### 6.2 GraphSAGE (Graph SAmple and aggreGatE)

-   **Tư tưởng**: GraphSAGE giải quyết một hạn chế lớn của GCN là khả năng mở rộng (scalability) và khả năng quy nạp (inductive capability). Thay vì yêu cầu toàn bộ đồ thị và hoạt động theo kiểu transductive, GraphSAGE được thiết kế để học các hàm tổng hợp có thể áp dụng cho các đỉnh mới hoặc đồ thị mới chưa từng thấy trong quá trình huấn luyện.
-   **Cách hoạt động**: GraphSAGE học một hàm để tạo ra các biểu diễn đỉnh bằng cách lấy mẫu (sample) và tổng hợp (aggregate) các đặc trưng từ các hàng xóm của mỗi đỉnh. Quá trình này được thực hiện lặp đi lặp lại qua nhiều layer.
-   **Quy trình chính**:
    1.  **Lấy mẫu hàng xóm (Neighbor Sampling)**: Đối với mỗi đỉnh, GraphSAGE lấy mẫu một tập con cố định các hàng xóm, thay vì sử dụng tất cả hàng xóm. Điều này giúp kiểm soát bộ nhớ và thời gian tính toán.
    2.  **Tổng hợp thông tin (Information Aggregation)**: Thông tin từ các hàng xóm đã được lấy mẫu sau đó được tổng hợp bằng một hàm tổng hợp (aggregator function) thành một vector duy nhất. Hàm tổng hợp phải có tính chất hoán vị bất biến (permutation invariant) vì thứ tự hàng xóm không quan trọng.
-   **Các hàm Aggregator phổ biến**:
    -   **Mean Aggregator**: Lấy trung bình embedding của các hàng xóm (tương tự với GCN).
    -   **Pooling Aggregator**: Áp dụng một mạng nơ-ron nhỏ (MLP) lên embedding của mỗi hàng xóm, sau đó áp dụng Max Pooling hoặc Mean Pooling.
    -   **LSTM Aggregator**: Áp dụng một mạng LSTM lên một hoán vị ngẫu nhiên của các hàng xóm. Điều này cho phép mạng nắm bắt các mẫu phức tạp hơn, nhưng không còn hoán vị bất biến hoàn toàn.
-   **Công thức (cho Mean Aggregator)**:
    -   Aggregate: $\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{MEAN} \left( \{ \mathbf{h}_u^{(k)} \mid u \in \mathcal{N}(v) \} \right)$
    -   Update: $\mathbf{h}_v^{(k+1)} = \sigma \left( \mathbf{W}^{(k)} \cdot \text{CONCAT}(\mathbf{h}_v^{(k)}, \mathbf{h}_{\mathcal{N}(v)}^{(k)}) \right)$
-   **Lợi ích**:
    -   **Inductive Capability**: Có khả năng tạo embeddings cho các đỉnh mới hoặc toàn bộ đồ thị mới chưa thấy trong training.
    -   **Scalability**: Nhờ chiến lược lấy mẫu, có thể áp dụng cho các đồ thị rất lớn (hàng tỷ đỉnh và cạnh).
    -   **Linh hoạt**: Cho phép tùy chỉnh các hàm tổng hợp.

### 6.3 Graph Attention Networks (GAT)

-   **Tư tưởng**: GAT giải quyết một hạn chế của các GNN trước đó (như GCN và GraphSAGE) là việc gán tầm quan trọng như nhau cho tất cả các hàng xóm. Thay vào đó, GAT giới thiệu cơ chế **attention** để học một cách linh hoạt về mức độ quan trọng của mỗi đỉnh hàng xóm đối với một đỉnh trung tâm.
-   **Cách hoạt động**:
    1.  **Tính toán hệ số Attention**: Đối với mỗi cặp đỉnh $(i, j)$ trong đó $j$ là hàng xóm của $i$, GAT tính toán một hệ số attention $e_{ij}$. Hệ số này thể hiện mức độ liên quan hoặc tầm quan trọng của đỉnh $j$ đối với đỉnh $i$. Việc này thường được thực hiện bằng cách áp dụng một phép biến đổi tuyến tính và hàm kích hoạt LeakyReLU cho sự kết hợp của các embedding của $i$ và $j$.
    2.  **Chuẩn hóa Attention**: Các hệ số attention thô $e_{ij}$ được chuẩn hóa bằng hàm `softmax` trên tất cả các hàng xóm của đỉnh $i$ để có được các trọng số attention $\alpha_{ij}$ có tổng bằng 1.
    3.  **Tổng hợp thông tin**: Biểu diễn mới của đỉnh $i$ ($h_i'$) được tính bằng tổng có trọng số của các biểu diễn (embedding) của các đỉnh hàng xóm, với trọng số chính là các hệ số attention $\alpha_{ij}$ đã học.
-   **Công thức (cho một head attention)**:
    -   Tính attention scores: $e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \, \Vert \, \mathbf{W}\mathbf{h}_j])$
    -   Chuẩn hóa attention: $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp(e_{ik})}$
    -   Update: $\mathbf{h}_i' = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij} \mathbf{W}\mathbf{h}_j \right)$
    -   (Lưu ý: $\Vert$ là phép nối vector, $\mathbf{a}$ là vector trọng số attention học được).
-   **Multi-head Attention**: Để tăng cường khả năng biểu diễn và ổn định quá trình huấn luyện, GAT thường sử dụng Multi-head Attention (tương tự Transformer), trong đó nhiều "head" attention độc lập được tính toán và kết quả của chúng được nối (concatenate) hoặc lấy trung bình.
-   **Lợi ích**:
    -   **Khả năng biểu diễn mạnh mẽ**: Học được các mối quan hệ phức tạp hơn bằng cách gán trọng số khác nhau cho các hàng xóm.
    -   **Giải thích được (Interpretability)**: Các hệ số attention có thể cung cấp cái nhìn về mức độ quan trọng của các hàng xóm.
    -   **Inductive**: Có khả năng áp dụng cho các cấu trúc đồ thị chưa thấy (tương tự GraphSAGE).
    -   **Không yêu cầu ma trận kề**: Thích hợp cho các đồ thị động (dynamic graphs) hoặc các đồ thị không rõ ràng.
-   **Nhược điểm**:
    -   Phức tạp hơn về mặt tính toán so với GCN.
    -   Đôi khi có thể gặp vấn đề về bộ nhớ với đồ thị lớn do việc tính toán các cặp attention.
## 🎯 6. Bài tập và Tham khảo

### 6.1 Bài tập thực hành
1.  **Node Classification trên Cora**: Sử dụng bộ dữ liệu Cora (mạng lưới trích dẫn khoa học), xây dựng một mô hình GCN để phân loại các bài báo khoa học vào các lĩnh vực khác nhau.
2.  **Link Prediction**: Xây dựng một mô hình GNN để dự đoán các mối quan hệ bạn bè chưa được thiết lập trong một mạng xã hội.
3.  **GraphSAGE vs. GAT**: Implement cả hai mô hình trên cùng một bộ dữ liệu và so sánh hiệu suất, thời gian huấn luyện.

### 6.2 Tài liệu tham khảo
-   **Thư viện**: `PyTorch Geometric (PyG)`, `Deep Graph Library (DGL)`.
-   **Khóa học**: Stanford CS224W: Machine Learning with Graphs.
-   **Bài báo quan trọng**:
    -   "Semi-Supervised Classification with Graph Convolutional Networks" (GCN paper).
    -   "Inductive Representation Learning on Large Graphs" (GraphSAGE paper).
    -   "Graph Attention Networks" (GAT paper).

---
*Chúc bạn học tập hiệu quả! 🚀*

# 🌐 Federated Learning - Học liên hợp

> **Mục tiêu**: Hiểu các nguyên tắc cơ bản, kiến trúc và thách thức của Học liên hợp, một phương pháp huấn luyện mô hình machine learning trên dữ liệu phân tán mà không cần thu thập dữ liệu về một nơi.

## 📋 Tổng quan nội dung

```mermaid
graph TD
    A[🌐 Federated Learning] --> B[🤔 Tại sao cần FL?]
    A --> C[⚙️ Thuật toán FedAvg]
    A --> D[🌋 Các thách thức chính]
    A --> E[🏛️ Các kiến trúc FL]
    A --> F[🌍 Ứng dụng]
    
    B --> B1[Bảo vệ quyền riêng tư (Privacy)]
    B --> B2[Tuân thủ pháp lý (GDPR, HIPAA)]
    B --> B3[Giảm chi phí truyền dữ liệu]
    B --> B4[Tận dụng dữ liệu tại chỗ (on-edge)]
    
    C --> C1[Vòng lặp huấn luyện]
    C --> C2[Lựa chọn Clients]
    C --> C3[Huấn luyện cục bộ (Local Training)]
    C --> C4[Tổng hợp mô hình (Aggregation)]
    
    D --> D1[Data Heterogeneity (Non-IID)]
    D --> D2[Communication Bottleneck]
    D --> D3[Bảo mật và Quyền riêng tư nâng cao]
    D --> D4[Quản lý hệ thống]
    
    E --> E1[Cross-Device]
    E --> E2[Cross-Silo]
    
    F --> F1[Bàn phím thông minh (Gboard)]
    F --> F2[Y tế]
    F --> F3[Tài chính]
```

## 📖 1. Glossary (Định nghĩa cốt lõi)

-   **Federated Learning (FL)**: Một kỹ thuật huấn luyện ML trong đó nhiều thiết bị (clients) hợp tác để huấn luyện một mô hình chung mà không cần chia sẻ dữ liệu gốc của họ.
-   **Client**: Một thiết bị hoặc một trung tâm dữ liệu cục bộ tham gia vào quá trình huấn luyện (ví dụ: điện thoại di động, bệnh viện).
-   **Server**: Một máy chủ trung tâm điều phối quá trình huấn luyện, có nhiệm vụ gửi mô hình toàn cục và tổng hợp các cập nhật.
-   **Global Model**: Mô hình dùng chung được lưu trữ trên server.
-   **Local Model**: Một bản sao của mô hình toàn cục được huấn luyện trên dữ liệu cục bộ của mỗi client.
-   **Communication Round**: Một chu kỳ hoàn chỉnh bao gồm: server gửi mô hình, client huấn luyện, và server tổng hợp cập nhật.
-   **Non-IID Data**: Dữ liệu không được phân phối một cách độc lập và đồng nhất (Independently and Identically Distributed) trên các client. Đây là đặc điểm và cũng là thách thức lớn nhất của FL. Ví dụ: mỗi người dùng điện thoại có thói quen gõ phím rất khác nhau.

---

## 🤔 2. Tại sao cần Học liên hợp? Vấn đề cốt lõi

Machine learning truyền thống hoạt động theo mô hình tập trung:
1.  Thu thập tất cả dữ liệu từ người dùng/thiết bị về một máy chủ trung tâm.
2.  Huấn luyện một mô hình lớn trên toàn bộ dữ liệu này.

Mô hình này ngày càng gặp nhiều vấn đề:
-   **Quyền riêng tư (Privacy)**: Người dùng ngày càng lo ngại về việc dữ liệu cá nhân (tin nhắn, hình ảnh, thông tin sức khỏe) của họ bị thu thập và lưu trữ ở một nơi.
-   **Pháp lý (Regulation)**: Các luật như GDPR (Châu Âu) hay HIPAA (Y tế Mỹ) đặt ra các quy định rất nghiêm ngặt về việc di chuyển và xử lý dữ liệu cá nhân.
-   **Chi phí và Độ trễ**: Gửi một lượng lớn dữ liệu (ví dụ: video) từ hàng triệu thiết bị về máy chủ là rất tốn kém về băng thông và có độ trễ cao.

**Federated Learning đảo ngược quy trình này**:
> Thay vì mang dữ liệu đến mô hình, FL mang mô hình đến dữ liệu.

Mô hình được huấn luyện ngay trên thiết bị của người dùng, và chỉ có các **cập nhật của mô hình** (model updates - các con số toán học) được gửi về máy chủ. Dữ liệu gốc không bao giờ rời khỏi thiết bị.

---

## ⚙️ 3. Thuật toán kinh điển: Federated Averaging (FedAvg)

FedAvg là thuật toán nền tảng và phổ biến nhất trong FL.

**Quy trình hoạt động trong một Communication Round:**

1.  **Selection (Lựa chọn)**: Server trung tâm chọn ra một tập con các client (ví dụ: 100 trong số 10,000 client có sẵn) để tham gia vào vòng huấn luyện. Việc lựa chọn thường ưu tiên các client đang sạc, có kết nối Wi-Fi và không được sử dụng.
2.  **Distribution (Phân phối)**: Server gửi phiên bản hiện tại của mô hình toàn cục (global model) đến các client đã được chọn.
3.  **Local Training (Huấn luyện cục bộ)**:
    -   Mỗi client nhận mô hình toàn cục và huấn luyện nó trên dữ liệu của **chính mình** trong một vài epoch.
    -   Quá trình này tạo ra một "phiên bản cải tiến" của mô hình, đã học được từ dữ liệu cục bộ.
4.  **Aggregation (Tổng hợp)**:
    -   Mỗi client **không gửi dữ liệu** về server. Thay vào đó, nó chỉ gửi về **sự thay đổi** của các trọng số (model updates hoặc model weights).
    -   Server chờ nhận đủ các cập nhật từ các client.
5.  **Update (Cập nhật)**: Server tổng hợp tất cả các cập nhật nhận được, thường bằng cách lấy **trung bình có trọng số** (weighted average) dựa trên số lượng mẫu dữ liệu của mỗi client. Kết quả của phép tổng hợp này trở thành mô hình toàn cục mới cho vòng tiếp theo.

Quá trình này được lặp lại hàng trăm, hàng nghìn vòng cho đến khi mô hình toàn cục hội tụ.

---

## 🌋 4. Các thách thức chính trong Federated Learning

FL không phải là một giải pháp hoàn hảo và đi kèm với nhiều thách thức kỹ thuật độc đáo:

-   **Data Heterogeneity (Tính không đồng nhất của dữ liệu - Non-IID)**:
    -   **Vấn đề**: Dữ liệu trên mỗi client rất khác nhau (ví dụ: người A gõ nhiều về công nghệ, người B gõ nhiều về nấu ăn). Khi server lấy trung bình các cập nhật, các cập nhật từ các client khác nhau có thể "xung đột" với nhau, làm cho mô hình toàn cục hội tụ chậm hoặc không chính xác.
    -   **Giải pháp**: Các thuật toán như **FedProx** thêm một thành phần vào hàm mất mát cục bộ để "kéo" mô hình cục bộ không đi quá xa so với mô hình toàn cục. Điều này giúp các mô hình cục bộ không bị "lệch" quá mức do dữ liệu Non-IID, cải thiện sự ổn định và hội tụ của mô hình toàn cục.

-   **Communication Bottleneck (Nút cổ chai giao tiếp)**:
    -   **Vấn đề**: Mặc dù không gửi dữ liệu, việc gửi toàn bộ trọng số của một mô hình lớn (hàng triệu tham số) từ hàng nghìn client vẫn rất tốn băng thông.
    -   **Giải pháp**:
        -   **Quantization**: Giảm độ chính xác của các trọng số (ví dụ: từ float32 xuống int8, hoặc thậm chí nhị phân hóa).
        -   **Sparsification**: Chỉ gửi các cập nhật trọng số quan trọng nhất (những trọng số có thay đổi lớn nhất).
        -   **Federated Dropout**: Một biến thể của dropout được áp dụng cho việc truyền thông, nơi chỉ một phần các trọng số được cập nhật và gửi đi.

-   **Privacy Concerns (Lo ngại về quyền riêng tư nâng cao)**:
    -   **Vấn đề**: Mặc dù không gửi dữ liệu gốc, các cập nhật mô hình vẫn có thể bị tấn công để suy ngược ra thông tin nhạy cảm về dữ liệu training (inference attacks, reconstruction attacks).
    -   **Giải pháp**:
        -   **Differential Privacy (Quyền riêng tư vi phân)**: Thêm một lượng nhiễu (noise) có kiểm soát vào các cập nhật (trọng số hoặc gradient) trước khi gửi về server. Điều này đảm bảo rằng sự hiện diện hay vắng mặt của bất kỳ một mẫu dữ liệu cá nhân nào trong tập training sẽ không ảnh hưởng đáng kể đến output của mô hình.
        -   **Secure Aggregation (Tổng hợp bảo mật)**: Sử dụng các kỹ thuật mã hóa đồng cấu (Homomorphic Encryption) hoặc tính toán đa bên an toàn (Secure Multi-Party Computation - SMPC) để server chỉ có thể giải mã được tổng của các cập nhật, chứ không thể xem được cập nhật của từng client riêng lẻ. Điều này ngăn chặn server nhìn thấy thông tin từ từng client.
        -   **Homomorphic Encryption (Mã hóa đồng cấu)**: Một kỹ thuật mã hóa cho phép thực hiện các phép toán trên dữ liệu đã mã hóa mà không cần giải mã. Điều này có thể được sử dụng để tổng hợp các cập nhật mô hình mà server không cần nhìn thấy các giá trị cập nhật riêng lẻ.

---

## 🏛️ 5. Các kiến trúc Federated Learning

1.  **Cross-Device FL**:
    -   **Mô tả**: Áp dụng trên một số lượng rất lớn các thiết bị di động hoặc thiết bị IoT.
    -   **Đặc điểm**: Số lượng client khổng lồ (hàng triệu), không đáng tin cậy (có thể mất kết nối bất cứ lúc nào), dữ liệu Non-IID cao, tài nguyên tính toán hạn chế.
    -   **Ví dụ**: Huấn luyện mô hình gợi ý từ khóa trên bàn phím Gboard của Google.
2.  **Cross-Silo FL**:
    -   **Mô tả**: Áp dụng trên một số lượng nhỏ các client, nhưng mỗi client là một tổ chức lớn (silo) có nhiều dữ liệu và tài nguyên.
    -   **Đặc điểm**: Số lượng client nhỏ (2-100), đáng tin cậy, luôn sẵn sàng, tài nguyên tính toán mạnh.
    -   **Ví dụ**: Nhiều bệnh viện hợp tác để huấn luyện một mô hình chẩn đoán ung thư mà không cần chia sẻ dữ liệu bệnh nhân.

## 🎯 6. Bài tập và Tham khảo

### 6.1 Bài tập thực hành
1.  **Mô phỏng FedAvg**: Sử dụng thư viện `flower` hoặc `PySyft`, mô phỏng một kịch bản FL đơn giản. Chia bộ dữ liệu MNIST thành 10 phần cho 10 client, mỗi client chỉ có dữ liệu của một chữ số. Quan sát xem mô hình toàn cục có học được cách nhận dạng tất cả 10 chữ số không.
2.  **Nghiên cứu Non-IID**: Thử nghiệm các cách chia dữ liệu Non-IID khác nhau và xem ảnh hưởng của nó đến tốc độ hội tụ của FedAvg.
3.  **So sánh với Centralized**: So sánh hiệu suất của mô hình được huấn luyện bằng FL với mô hình được huấn luyện theo cách truyền thống (tập trung) trên cùng một bộ dữ liệu.

### 6.2 Tài liệu tham khảo
-   **Thư viện**: `Flower` (phổ biến, dễ sử dụng), `PySyft` (tập trung vào privacy), `TensorFlow Federated (TFF)`.
-   **Bài báo gốc**: "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg paper).
-   **Khóa học**:
    -   "Federated Learning: One-World" của OpenMined trên YouTube.
    -   Các tutorial của thư viện Flower.

---
*Chúc bạn học tập hiệu quả! 🚀*

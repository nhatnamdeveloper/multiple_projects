# 🎨 Generative Models - Mô hình sinh

> **Mục tiêu**: Hiểu các kiến trúc và nguyên lý đằng sau các mô hình có khả năng *tạo ra* dữ liệu mới (ảnh, văn bản, âm thanh), tập trung vào GANs, VAEs và Diffusion Models.

## 📋 Tổng quan nội dung

```mermaid
graph TD
    A[🎨 Generative Models] --> B[🧠 Nền tảng lý thuyết]
    A --> C[⚔️ Generative Adversarial Networks (GANs)]
    A --> D[🧬 Variational Autoencoders (VAEs)]
    A --> E[🌫️ Diffusion Models]
    A --> F[📊 Đánh giá mô hình sinh]
    
    B --> B1[Latent Space (Không gian ẩn)]
    B --> B2[Likelihood-based vs. Implicit Models]
    B --> B3[Generative Learning Trilemma]
    
    C --> C1[Kiến trúc Generator & Discriminator]
    C --> C2[Hàm mất mát Min-Max]
    C --> C3[Các vấn đề (Mode Collapse, Instability)]
    C --> C4[Các biến thể (DCGAN, WGAN, StyleGAN)]
    
    D --> D1[Kiến trúc Encoder & Decoder]
    D --> D2[Reparameterization Trick]
    D --> D3[Hàm mất mát (Reconstruction + KL Divergence)]
    
    E --> E1[Forward & Reverse Process]
    E --> E2[Noise Schedule]
    E --> E3[U-Net Denoising Model]
    
    F --> F1[Inception Score (IS)]
    F --> F2[Fréchet Inception Distance (FID)]
    F --> F3[Precision & Recall]
```

## 📚 1. Bảng ký hiệu (Notation)

- **Real Data ($x$)**: Dữ liệu thật từ tập huấn luyện.
- **Latent Vector ($z$)**: Vector ngẫu nhiên trong không gian ẩn, dùng làm "hạt giống" để sinh dữ liệu.
- **Generator ($G$)**: Mạng nơ-ron sinh ra dữ liệu giả, $G(z) = \hat{x}$.
- **Discriminator ($D$)**: Mạng nơ-ron phân biệt dữ liệu thật/giả, $D(x)$ trả về xác suất $x$ là thật.
- **Encoder ($E$)**: Mạng nơ-ron mã hóa dữ liệu thật vào không gian ẩn, $E(x) = z$.
- **Decoder ($Dec$)**: Mạng nơ-ron giải mã từ không gian ẩn ra dữ liệu, $Dec(z) = \hat{x}$.

## 📖 2. Glossary (Định nghĩa cốt lõi)

-   **Generative Model**: Mô hình học phân phối xác suất $P(x)$ của dữ liệu và có thể sinh ra các mẫu mới từ phân phối đó.
-   **Discriminative Model**: Mô hình học ranh giới quyết định giữa các lớp, hay học xác suất có điều kiện $P(y|x)$.
-   **Latent Space**: Không gian biểu diễn có số chiều thấp hơn, nơi các đặc trưng cốt lõi, trừu tượng của dữ liệu được mã hóa.
-   **Mode Collapse**: Một lỗi phổ biến của GAN khi Generator chỉ học cách tạo ra một vài mẫu dữ liệu giả rất thuyết phục thay vì toàn bộ sự đa dạng của dữ liệu thật.
-   **Reparameterization Trick**: Một kỹ thuật toán học cho phép backpropagation có thể "chảy" qua một node lấy mẫu ngẫu nhiên (stochastic node), là chìa khóa để huấn luyện VAE.
-   **Diffusion Process**: Quá trình thêm nhiễu (noise) dần dần vào dữ liệu (forward) và sau đó học cách khử nhiễu để tái tạo lại dữ liệu gốc (reverse).

---

## 🧠 3. Nền tảng lý thuyết: Không gian ẩn (Latent Space)

> **Tư tưởng cốt lõi**: Hầu hết dữ liệu trong thế giới thực (như ảnh chân dung) đều có một cấu trúc tiềm ẩn. Thay vì nằm ngẫu nhiên trong không gian pixel (ví dụ: ảnh nhiễu), chúng nằm trên một **manifold** có số chiều thấp hơn nhiều. Ví dụ, tất cả các ảnh chân dung người đều có chung các đặc điểm như "có mắt", "có mũi", "miệng ở dưới mũi".

-   **Latent Space (Không gian ẩn)** là một không gian có số chiều thấp hơn dùng để biểu diễn các đặc trưng trừu tượng này.
-   **Ví dụ**: Một không gian ẩn 2 chiều cho ảnh chân dung có thể có một trục là "độ tuổi" và trục còn lại là "góc nhìn khuôn mặt".
-   **Mục tiêu của mô hình sinh**:
    1.  Học một **Encoder** để ánh xạ một bức ảnh thật $x$ vào một điểm $z$ trong không gian ẩn.
    2.  Học một **Decoder (hay Generator)** để ánh xạ một điểm $z$ bất kỳ trong không gian ẩn trở lại một bức ảnh thực tế $\hat{x}$.

Nếu học thành công, ta có thể lấy một điểm $z$ ngẫu nhiên trong không gian này và dùng Decoder để sinh ra một bức ảnh chân dung hoàn toàn mới chưa từng tồn tại.

---

## ⚔️ 4. Generative Adversarial Networks (GANs)

GAN được giới thiệu bởi Ian Goodfellow và cộng sự vào năm 2014, dựa trên một ý tưởng độc đáo về một "trò chơi" giữa hai mạng nơ-ron.

### 4.1 Tư duy trực quan: Trò chơi Mèo vờn Chuột
Hãy tưởng tượng một cuộc đối đầu giữa hai nhân vật:
1.  **Generator (G)**: Một **họa sĩ chuyên làm tranh giả**. Mục tiêu của G là vẽ ra những bức tranh giả trông y như thật để lừa người khác.
2.  **Discriminator (D)**: Một **nhà phê bình nghệ thuật**. Mục tiêu của D là phân biệt đâu là tranh thật (từ bộ sưu tập gốc) và đâu là tranh giả do G vẽ ra.

**Quá trình huấn luyện:**
-   **Vòng 1 (Huấn luyện Discriminator)**:
    -   Đưa cho D một nửa là tranh thật và một nửa là tranh giả do G vẽ.
    -   Dự đoán của D được so sánh với nhãn thật/giả.
    -   Cập nhật trọng số của D để nó ngày càng giỏi hơn trong việc phân biệt.
-   **Vòng 2 (Huấn luyện Generator)**:
    -   G vẽ ra một bức tranh giả và đưa cho D.
    -   G muốn D phải tin rằng đây là tranh thật (tức $D(G(z))$ phải tiến về 1).
    -   Lỗi của D được lan truyền ngược lại để cập nhật trọng số của **chỉ G**, giúp G học cách vẽ ra những bức tranh ngày càng thuyết phục hơn.

Quá trình này lặp đi lặp lại. D ngày càng tinh vi, buộc G cũng phải ngày càng tiến bộ. Cuối cùng, G sẽ tạo ra những sản phẩm giả mà D không thể phân biệt được nữa.

### 4.2 Hàm mất mát Min-Max
Trò chơi này được mô tả bằng hàm mất mát min-max:
$$ \min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log(1 - D(G(z)))] $$

-   **Phần $\max_{D}$**: Discriminator `D` muốn tối đa hóa hàm này. Nó muốn $D(x)$ (xác suất ảnh thật là thật) tiến về 1 và $D(G(z))$ (xác suất ảnh giả là thật) tiến về 0.
-   **Phần $\min_{G}$**: Generator `G` muốn tối thiểu hóa hàm này. Nó muốn $D(G(z))$ tiến về 1, làm cho vế thứ hai của công thức trở nên âm vô cùng.

### 4.3 Các vấn đề thường gặp khi huấn luyện GAN
-   **Training Instability**: Quá trình huấn luyện GAN rất "mong manh". Nếu D quá giỏi, G sẽ không học được gì (gradient bằng 0). Nếu G quá giỏi, nó sẽ dễ dàng lừa D và không cải thiện thêm.
-   **Mode Collapse**: G chỉ tìm ra một vài "chiêu" để lừa D (ví dụ: chỉ vẽ đúng một loại chó trông rất thật). Nó không học được toàn bộ phân phối của dữ liệu mà chỉ "sụp đổ" vào một vài mode.
-   **Vanishing Gradients**: Nếu D quá tự tin, gradient cho G sẽ trở nên rất nhỏ, khiến G học rất chậm hoặc không học được.

### 4.4 Các biến thể quan trọng
-   **DCGAN (Deep Convolutional GAN)**:
    -   **Mục tiêu**: Ổn định huấn luyện GAN bằng cách kết hợp các kiến trúc CNN.
    -   **Đề xuất kiến trúc**:
        -   Thay thế mọi lớp pooling bằng **strided convolutions** (trong Discriminator) và **fractional-strided convolutions** (Deconvolution/ConvTranspose2d trong Generator).
        -   Sử dụng **Batch Normalization** trong cả Generator và Discriminator (trừ layer output của G và layer input của D).
        -   Sử dụng **ReLU** cho các lớp Generator (trừ layer output dùng Tanh).
        -   Sử dụng **LeakyReLU** cho các lớp Discriminator.
-   **WGAN (Wasserstein GAN)**:
    -   **Vấn đề GAN truyền thống**: Hàm mất mát cross-entropy của GAN dựa trên khoảng cách Jensen-Shannon (JS divergence), có thể gây ra **vanishing gradients** khi hai phân phối (thật và giả) không chồng chập nhiều (common support), điều thường xuyên xảy ra khi huấn luyện.
    -   **Giải pháp**: Thay thế JS divergence bằng **Wasserstein distance** (Earth Mover's Distance).
    -   **Cách hoạt động**: Để đảm bảo Lipschitz continuity (một điều kiện cần để Wasserstein distance có đạo hàm tốt), WGAN sử dụng **weight clipping** (cắt giới hạn trọng số) hoặc **gradient penalty** (thêm penalty vào gradient của Discriminator).
    -   **Lợi ích**: Giúp huấn luyện ổn định hơn, ít bị mode collapse hơn, và Discriminator (trong WGAN gọi là Critic) trả về một giá trị có ý nghĩa hơn (ước tính khoảng cách Wasserstein).
-   **StyleGAN**: Một kiến trúc phức tạp cho phép kiểm soát các khía cạnh khác nhau của ảnh được sinh ra (ví dụ: tuổi, giới tính, kiểu tóc trong ảnh chân dung) thông qua không gian ẩn.

### 1. Bài toán & dữ liệu
- **Bài toán**: Tạo ra hình ảnh chất lượng cao từ nhiễu ngẫu nhiên, sử dụng kiến trúc mạng tích chập (CNNs).
- **Dữ liệu**: Tập hợp ảnh thật (ví dụ: MNIST, CelebA).
- **Ứng dụng**: Sinh ảnh chân dung, vật thể, tăng cường dữ liệu.

### 2. Mô hình & công thức
- **Generator ($G$)**: Mạng CNN chuyển đổi latent vector `z` thành ảnh $\hat{x}$. Sử dụng `ConvTranspose2d` (Deconvolution) để upsample.
- **Discriminator ($D$)**: Mạng CNN phân loại ảnh đầu vào $x$ là thật hay giả.
- **Kiến trúc đề xuất**:
  -   Thay thế Pooling layer bằng các bước tiến tích chập (strided convolutions) trong D và tích chập phân số (fractional-strided convolutions / deconvolution) trong G.
  -   Sử dụng Batch Normalization trong cả G và D (trừ layer output của G và layer input của D).
  -   Sử dụng ReLU trong G (trừ layer output dùng Tanh).
  -   Sử dụng LeakyReLU trong D.

### 3. Loss & mục tiêu
- **Mục tiêu**: Huấn luyện $G$ để sinh ảnh giả $G(z)$ mà $D$ không thể phân biệt được với ảnh thật, đồng thời huấn luyện $D$ để phân biệt ảnh thật và giả.
- **Hàm mất mát**: Hàm mất mát nhị phân cross-entropy (`nn.BCELoss`) cho cả G và D.

### 4. Tối ưu hoá & cập nhật
- **Algorithm**: Huấn luyện D và G xen kẽ.
  1.  **Huấn luyện D**: Tính loss của D trên ảnh thật và ảnh giả, thực hiện một bước tối ưu hóa để tối đa hóa $D(x)$ và tối thiểu hóa $D(G(z))$.
  2.  **Huấn luyện G**: Tính loss của G (bằng cách cố gắng làm cho $D(G(z))$ gần 1), thực hiện một bước tối ưu hóa để tối thiểu hóa $1 - D(G(z))$.
- **Optimizer**: Thường dùng Adam.

### 5. Hyperparams
- **Batch Size**: 64-128.
- **Learning Rate**: 0.0002.
- **Adam Betas**: (0.5, 0.999).
- **Latent Vector Size**: 100 chiều.

### 6. Độ phức tạp
- **Time**: Tốn kém, đặc biệt cho G (do upsampling) và D (do xử lý CNN).
- **Space**: Tốn bộ nhớ VRAM, cần GPU mạnh.

### 7. Metrics đánh giá
- **Inception Score (IS)**, **Fréchet Inception Distance (FID)**: Để đánh giá chất lượng và sự đa dạng của ảnh sinh ra.
- **Quan sát bằng mắt**: Rất quan trọng để đánh giá trực quan.

### 8. Ưu / Nhược điểm
**Ưu điểm**:
- Tạo ảnh có độ phân giải cao và chân thực.
- Kiến trúc CNN giúp học các đặc trưng không gian tốt.

**Nhược điểm**:
- Khó huấn luyện (training instability).
- Dễ bị Mode Collapse.
- Yêu cầu cân bằng cẩn thận giữa G và D.

### 9. Bẫy & mẹo
- **Bẫy**: Training Instability, Mode Collapse.
- **Mẹo**: Sử dụng kiến trúc DCGAN với các hướng dẫn đã được chứng minh.
- **Mẹo**: Giảm Learning Rate cho Discriminator.

### 10. Pseudocode:
```python
# Khởi tạo G và D
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Huấn luyện Discriminator
        D.zero_grad()
        real_images = batch
        batch_size = real_images.size(0)
        
        # Loss trên ảnh thật
        output_real = D(real_images)
        loss_D_real = criterion(output_real, labels_real) # labels_real = 1
        loss_D_real.backward()
        
        # Loss trên ảnh giả
        noise = sample_latent_vector()
        fake_images = G(noise)
        output_fake = D(fake_images.detach()) # .detach() để không cập nhật G
        loss_D_fake = criterion(output_fake, labels_fake) # labels_fake = 0
        loss_D_fake.backward()
        
        loss_D = loss_D_real + loss_D_fake
        optimizer_D.step()
        
        # 2. Huấn luyện Generator
        G.zero_grad()
        output_fake_from_G = D(fake_images) # Không .detach()
        loss_G = criterion(output_fake_from_G, labels_real) # G muốn D nghĩ ảnh giả là thật
        loss_G.backward()
        optimizer_G.step()
```

### 11. Code mẫu (kiến trúc DCGAN cơ bản)
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (256)x4x4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (128)x8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64)x16x16
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (img_channels)x32x32
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, img_channels, img_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (img_channels)x32x32
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64)x16x16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (128)x8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256)x4x4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)

# Ví dụ khởi tạo
# latent_dim = 100
# img_channels = 1 # For MNIST
# img_size = 32
# netG = Generator(latent_dim, img_channels, img_size)
# netD = Discriminator(img_channels, img_size)
# print(netG)
# print(netD)
```

### 12. Checklist kiểm tra nhanh:
- [ ] Hàm mất mát có được tính đúng cho G và D?
- [ ] Kiến trúc có tuân theo các hướng dẫn DCGAN không?
- [ ] Optimizer có được cấu hình đúng không (learning rate, betas)?
- [ ] Có cân bằng giữa khám phá và khai thác không (epsilon-greedy)?
- [ ] Ảnh sinh ra có chất lượng và đa dạng không?

---
5. Variational Autoencoders (VAEs)

VAE là một loại mô hình sinh khác, có nền tảng lý thuyết vững chắc hơn GAN và thường dễ huấn luyện hơn.

### 5.1 Kiến trúc Encoder-Decoder
VAE bao gồm hai phần:
1.  **Encoder**: Nhận một ảnh đầu vào $x$, và mã hóa nó thành một **phân phối xác suất** trong không gian ẩn. Thay vì mã hóa thành một điểm duy nhất $z$, nó mã hóa thành một phân phối hình chuông (Gaussian) với trung bình $\mu$ và độ lệch chuẩn $\sigma$.
2.  **Decoder**: Lấy một điểm $z$ được **lấy mẫu (sampled)** từ phân phối đó và cố gắng tái tạo lại ảnh gốc $\hat{x}$.

### 5.2 Reparameterization Trick
-   **Vấn đề**: Quá trình "lấy mẫu ngẫu nhiên" từ phân phối $(\mu, \sigma)$ là một phép toán ngẫu nhiên, không có đạo hàm, do đó không thể lan truyền ngược gradient qua nó.
-   **Giải pháp (Reparameterization Trick)**: Thay vì lấy mẫu trực tiếp, ta biến đổi nó:
    $$ z = \mu + \sigma \odot \epsilon $$
    Trong đó $\epsilon$ là một biến nhiễu ngẫu nhiên lấy từ phân phối chuẩn $N(0, 1)$.
-   **Tại sao hiệu quả?**: Bằng cách này, sự ngẫu nhiên được "tách" ra khỏi mạng. Mạng chỉ học cách tạo ra $\mu$ và $\sigma$, còn gradient có thể chảy ngược qua các phép nhân và cộng một cách bình thường.

### 5.3 Hàm mất mát kép (Dual Loss Function)
Loss của VAE bao gồm hai thành phần:
$$ L(\theta, \phi) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} - \underbrace{D_{KL}(q_\phi(z|x) || p(z))}_{\text{KL Divergence}} $$
1.  **Reconstruction Loss**: Đo lường mức độ giống nhau giữa ảnh tái tạo $\hat{x}$ và ảnh gốc $x$. Nó buộc mô hình phải học cách mã hóa tất cả các thông tin cần thiết vào không gian ẩn.
2.  **KL Divergence**: Đây là một thành phần **regularizer**. Nó đo lường sự khác biệt giữa phân phối $(\mu, \sigma)$ mà Encoder tạo ra và một phân phối chuẩn $N(0, 1)$. Nó buộc Encoder phải tạo ra các không gian ẩn "gọn gàng", có cấu trúc tốt, tránh việc các "cụm" dữ liệu nằm quá xa nhau.

### 5.4 So sánh GANs và VAEs
| Đặc điểm | GANs | VAEs |
| :--- | :--- | :--- |
| **Chất lượng ảnh** | Sắc nét, chân thực hơn. | Mờ hơn, "trung bình" hơn. |
| **Độ ổn định** | Khó huấn luyện, dễ sụp đổ. | Dễ huấn luyện hơn, ổn định hơn. |
| **Không gian ẩn** | Không có cấu trúc rõ ràng. | Có cấu trúc, liên tục. |
| **Nền tảng** | Lý thuyết trò chơi. | Lý thuyết xác suất (Bayesian). |
| **Mục tiêu** | Đánh lừa Discriminator. | Tối đa hóa lower bound của log-likelihood. |

## 🌫️ 6. Diffusion Models

Đây là kiến trúc mô hình sinh hiện đại và mạnh mẽ nhất hiện nay, đứng sau các mô hình nổi tiếng như DALL-E 2, Midjourney, và Stable Diffusion.

### 6.1 Tư duy trực quan: Thêm nhiễu và Khử nhiễu
Quá trình hoạt động của Diffusion Model bao gồm hai bước:

1.  **Forward Process (Quá trình xuôi - Cố định)**:
    -   Bắt đầu với một bức ảnh thật $x_0$.
    -   Thêm một chút nhiễu (noise) vào ảnh để tạo ra $x_1$.
    -   Thêm một chút nhiễu vào $x_1$ để tạo ra $x_2$.
    -   Lặp lại quá trình này `T` lần (ví dụ: `T=1000`) cho đến khi ảnh $x_T$ trở thành nhiễu hoàn toàn (pure noise).
    -   Quá trình này là cố định và không cần học.

2.  **Reverse Process (Quá trình ngược - Phải học)**:
    -   Đây là phần cốt lõi của mô hình.
    -   Mô hình (thường là một kiến trúc U-Net) được huấn luyện để làm một việc duy nhất: **dự đoán nhiễu đã được thêm vào ở một bước bất kỳ**.
    -   Nhiệm vụ của nó là: nhận vào một ảnh nhiễu $x_t$ và time step $t$, và dự đoán ra nhiễu $\epsilon$ đã được thêm vào $x_{t-1}$ để tạo ra $x_t$.
    -   Sau khi được huấn luyện, để sinh ảnh mới:
        -   Bắt đầu với một ảnh nhiễu hoàn toàn ngẫu nhiên $x_T$.
        -   Dùng mô hình để dự đoán nhiễu trong $x_T$, sau đó trừ nhiễu đó đi để tạo ra $x_{T-1}$.
        -   Lặp lại quá trình: từ $x_{T-1}$ tạo ra $x_{T-2}$,... cho đến khi ta có được $x_0$, một bức ảnh sạch và hoàn toàn mới.

### 6.2 Tại sao Diffusion Models hiệu quả?
-   **Bài toán đơn giản**: Thay vì học cách sinh ra một bức ảnh phức tạp từ đầu, mô hình chỉ cần học một nhiệm vụ đơn giản hơn nhiều là "khử nhiễu".
-   **Huấn luyện ổn định**: Quá trình training ổn định hơn nhiều so với GANs.
-   **Chất lượng và đa dạng**: Cho kết quả sinh ảnh vừa sắc nét, vừa đa dạng, kết hợp được ưu điểm của cả GANs và VAEs.

### 6.3 Điều khiển tạo ảnh: Conditional Diffusion (Diffusion có điều kiện)
-   **Mục tiêu**: Thay vì sinh ảnh ngẫu nhiên, ta muốn sinh ảnh theo một điều kiện nào đó (ví dụ: sinh ảnh chó, hoặc sinh ảnh từ text "con mèo đang bay").
-   **Cách hoạt động**: Trong quá trình khử nhiễu (reverse process), ta cung cấp thêm thông tin điều kiện (ví dụ: one-hot vector của nhãn lớp, hoặc embedding của text) cho mô hình khử nhiễu (thường là U-Net). Mô hình sẽ học cách kết hợp thông tin này để tạo ra ảnh phù hợp với điều kiện.

### 6.4 Tăng tốc độ và hiệu quả: Latent Diffusion Models (LDMs)
-   **Vấn đề**: Các Diffusion Models truyền thống hoạt động trực tiếp trên không gian pixel. Điều này rất tốn kém về mặt tính toán và bộ nhớ, đặc biệt với ảnh độ phân giải cao.
-   **Giải pháp**: Thay vì chạy quá trình diffusion trên ảnh gốc, Latent Diffusion Models (LDMs) nén ảnh vào một **không gian tiềm ẩn (latent space)** có số chiều thấp hơn bằng một Autoencoder đã được huấn luyện trước. Quá trình diffusion sau đó diễn ra hoàn toàn trong không gian tiềm ẩn này.
-   **Cách hoạt động**:
    1.  **Encoder**: Nén ảnh pixel $x$ thành biểu diễn tiềm ẩn $z$.
    2.  **Diffusion trong Latent Space**: Áp dụng quá trình thêm nhiễu và khử nhiễu trong không gian tiềm ẩn $z$.
    3.  **Decoder**: Giải mã biểu diễn tiềm ẩn đã được khử nhiễu trở lại không gian pixel để tạo ra ảnh cuối cùng.
-   **Lợi ích**: Giảm đáng kể chi phí tính toán, cho phép huấn luyện và sinh ảnh nhanh hơn mà vẫn giữ được chất lượng cao. **Stable Diffusion** là một ví dụ nổi bật của LDM.
## 📊 7. Đánh giá mô hình sinh (Evaluation of Generative Models)

Đánh giá mô hình sinh là một bài toán khó, vì không có một "đáp án đúng" duy nhất. Ta cần đo lường hai yếu tố: **chất lượng (quality)** và **tính đa dạng (diversity)** của các mẫu được sinh ra.

### 7.1 Inception Score (IS)
- **Tư tưởng**: Một mô hình tốt sẽ sinh ra những hình ảnh **rõ ràng** (dễ phân loại) và **đa dạng** (bao trùm nhiều lớp khác nhau).
- **Cách hoạt động**:
    1.  Dùng một mô hình phân loại ảnh (Inception Net) đã được pre-trained trên ImageNet.
    2.  Cho mô hình sinh tạo ra nhiều ảnh.
    3.  Với mỗi ảnh, lấy phân phối xác suất trên các lớp từ Inception Net ($P(y|x)$).
    4.  **Chất lượng**: Nếu ảnh rõ ràng, $P(y|x)$ sẽ có entropy thấp (ví dụ: rất chắc chắn đây là ảnh "chó").
    5.  **Đa dạng**: Phân phối xác suất trung bình của tất cả các ảnh ($P(y)$) phải có entropy cao (mô hình sinh ra nhiều loại ảnh khác nhau).
- **Công thức**: $IS = \exp(\mathbb{E}_x [D_{KL}(P(y|x) || P(y))])$
- **Nhược điểm**: Không so sánh với dữ liệu thật, có thể bị "lừa" bởi các mô hình chỉ sinh ra một ảnh đẹp cho mỗi lớp.

### 7.2 Fréchet Inception Distance (FID)
- **Tư tưởng**: So sánh phân phối của các ảnh thật và ảnh giả trong không gian đặc trưng của một mạng nơ-ron. FID càng thấp, hai phân phối càng gần nhau, mô hình càng tốt.
- **Cách hoạt động**:
    1.  Lấy một tập ảnh thật và một tập ảnh do mô hình sinh ra.
    2.  Đưa cả hai tập ảnh qua mạng Inception Net (đã bỏ lớp cuối) để lấy các vector đặc trưng (feature vectors).
    3.  Modelling hai tập vector đặc trưng này như hai phân phối Gaussian đa biến. Tính toán trung bình ($\mu$) và ma trận hiệp phương sai ($\Sigma$) cho mỗi tập.
    4.  Tính khoảng cách Fréchet giữa hai phân phối Gaussian này.
- **Công thức**: $FID(x, g) = ||\mu_x - \mu_g||^2 + \text{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2})$
- **Ưu điểm**: Robust hơn IS, có tương quan tốt hơn với chất lượng ảnh mà con người cảm nhận. Là metric tiêu chuẩn hiện nay để đánh giá GANs và các mô hình sinh ảnh khác.

## 🎯 8. Bài tập thực hành
1.  **DCGAN trên MNIST**: Implement và huấn luyện một mô hình Deep Convolutional GAN đơn giản trên bộ dữ liệu chữ số viết tay MNIST. Cố gắng tạo ra những chữ số trông như thật.
2.  **VAE trên Fashion-MNIST**: Huấn luyện một mô hình VAE trên bộ dữ liệu Fashion-MNIST. Trực quan hóa không gian ẩn 2D và thử sinh ra các sản phẩm thời trang mới bằng cách di chuyển trong không gian đó.
3.  **Text-to-Image với Diffusion**: Sử dụng thư viện `diffusers` của Hugging Face để chạy một mô hình Stable Diffusion đã được pre-trained. Thử nghiệm với các prompt khác nhau để tạo ra các bức ảnh độc đáo.

## 📚 9. Tài liệu tham khảo
-   **GANs**: "Generative Adversarial Nets" - Goodfellow et al. (2014)
-   **VAEs**: "Auto-Encoding Variational Bayes" - Kingma & Welling (2013)
-   **Diffusion Models**: "Denoising Diffusion Probabilistic Models" - Ho et al. (2020)
-   **Tutorials**:
    -   [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
    -   [Hugging Face Diffusers Library](https://huggingface.co/docs/diffusers/index)
    -   [Blog: "Intuitively Understanding Variational Autoencoders"](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

---
*Chúc bạn học tập hiệu quả! 🚀*
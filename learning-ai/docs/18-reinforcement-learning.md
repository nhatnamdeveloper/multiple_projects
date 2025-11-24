# 🤖 Reinforcement Learning (RL) - Học tăng cường

> **Mục tiêu**: Hiểu sâu về các khái niệm cốt lõi của Học tăng cường, từ nền tảng lý thuyết (MDPs, Bellman) đến các thuật toán kinh điển (Q-learning, Policy Gradients) và ứng dụng thực tế.

## 📋 Tổng quan nội dung

```mermaid
graph TD
    A[🤖 Reinforcement Learning] --> B[🧠 Nền tảng lý thuyết]
    A --> C[⚙️ Thuật toán Value-Based]
    A --> D[📈 Thuật toán Policy-Based]
    A --> E[🎭 Thuật toán Actor-Critic]
    A --> F[🌍 Ứng dụng]
    
    B --> B1[Markov Decision Process (MDP)]
    B --> B2[Bellman Equations]
    B --> B3[Value & Policy Iteration]
    B --> B4[Exploration vs. Exploitation]
    
    C --> C1[Q-Learning]
    C --> C2[Deep Q-Networks (DQN)]
    C --> C3[Double DQN & Dueling DQN]
    
    D --> D1[Policy Gradients]
    D --> D2[REINFORCE]
    
    E --> E1[Advantage Actor-Critic (A2C)]
    E --> E2[Asynchronous A3C]
    E --> E3[Proximal Policy Optimization (PPO)]
    
    F --> F1[Chơi game (Atari, AlphaGo)]
    F --> F2[Robotics]
    F --> F3[Tối ưu hóa tài nguyên]
    F --> F4[Hệ thống gợi ý]
```

## 📚 1. Bảng ký hiệu (Notation)

- **Agent**: Tác nhân, thực thể ra quyết định.
- **Environment**: Môi trường, nơi agent tương tác.
- **State ($s \in S$)**: Trạng thái của môi trường.
- **Action ($a \in A$)**: Hành động mà agent có thể thực hiện.
- **Reward ($r$)**: Phần thưởng (hoặc phạt) mà agent nhận được từ môi trường.
- **Policy ($\pi(a|s)$)**: Chính sách, chiến lược của agent. Đây là một hàm xác suất chọn hành động `a` khi đang ở trạng thái `s`.
- **Value Function ($V^\pi(s)$)**: Hàm giá trị, ước tính tổng phần thưởng kỳ vọng trong tương lai khi bắt đầu từ trạng thái `s` và đi theo chính sách $\pi$.
- **Q-Value Function ($Q^\pi(s, a)$)**: Hàm giá trị hành động, ước tính tổng phần thưởng kỳ vọng khi thực hiện hành động `a` tại trạng thái `s` rồi sau đó đi theo chính sách $\pi$.
- **Discount Factor ($\gamma$)**: Hệ số chiết khấu ($0 \le \gamma \le 1$), thể hiện tầm quan trọng của phần thưởng trong tương lai so với phần thưởng trước mắt.

## 📖 2. Glossary (Định nghĩa cốt lõi)

-   **Markov Decision Process (MDP)**: Một khuôn khổ toán học để mô hình hóa việc ra quyết định trong môi trường mà kết quả vừa ngẫu nhiên, vừa chịu sự kiểm soát của agent.
-   **Bellman Equations**: Hệ phương trình đệ quy mô tả mối quan hệ giữa giá trị của một trạng thái và giá trị của các trạng thái kế tiếp. Là nền tảng cho hầu hết các thuật toán RL.
-   **Exploration vs. Exploitation Tradeoff**: Sự đánh đổi kinh điển trong RL.
    -   **Exploitation (Khai thác)**: Chọn hành động tốt nhất dựa trên những gì đã biết.
    -   **Exploration (Khám phá)**: Thử các hành động mới để có thể tìm ra những lựa chọn tốt hơn trong tương lai.
-   **On-Policy vs. Off-Policy**:
    -   **On-Policy**: Agent học và hành động theo cùng một chính sách.
    -   **Off-Policy**: Agent học một chính sách tối ưu trong khi đang hành động theo một chính sách khác (thường là chính sách có tính khám phá cao hơn). Q-Learning là một ví dụ điển hình.

---
## 🧠 3. Nền tảng lý thuyết

### 3.1 Markov Decision Process (MDP)

MDP là "sân chơi" mà các agent RL hoạt động trong đó. Nó định nghĩa các quy tắc của trò chơi. Một MDP được xác định bởi một bộ 5 thành phần `(S, A, P, R, γ)`:

1.  **S (States)**: Một tập hợp tất cả các trạng thái có thể có của môi trường.
    -   *Ví dụ (Cờ vua)*: Toàn bộ các cách sắp xếp quân cờ trên bàn cờ.
2.  **A (Actions)**: Một tập hợp tất cả các hành động mà agent có thể thực hiện.
    -   *Ví dụ (Cờ vua)*: Tất cả các nước đi hợp lệ tại một trạng thái bàn cờ.
3.  **P (Transition Probability Function - Hàm xác suất chuyển đổi)**: $P(s'|s, a)$ là xác suất chuyển đến trạng thái mới $s'$ sau khi thực hiện hành động $a$ tại trạng thái $s$.
    -   *Ví dụ (Robot di chuyển)*: Nếu robot ra lệnh "đi thẳng", có 80% xác suất nó sẽ đi thẳng, 10% trượt sang trái, và 10% trượt sang phải.
4.  **R (Reward Function - Hàm phần thưởng)**: $R(s, a, s')$ là phần thưởng agent nhận được khi chuyển từ $s$ đến $s'$ bằng hành động $a$.
    -   *Ví dụ (Game Pac-Man)*: +10 điểm khi ăn một viên thức ăn, -500 điểm khi bị ma đuổi, +1 khi sống sót qua mỗi bước.
5.  **γ (Discount Factor)**: Hệ số chiết khấu.
    -   *Ví dụ*: Nếu $\gamma = 0.9$, phần thưởng nhận được ở 1 bước trong tương lai chỉ có giá trị bằng 90% so với phần thưởng nhận được ngay lập tức.

**Tính chất Markov ("Memoryless")**: Tương lai chỉ phụ thuộc vào hiện tại, không phụ thuộc vào quá khứ. $P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1}|s_t, a_t)$. Trạng thái $s_t$ đã chứa tất cả thông tin cần thiết.

### 3.2 Phương trình Bellman (Bellman Equations)

Phương trình Bellman là công thức đệ quy dùng để tính toán giá trị của một trạng thái hoặc một cặp trạng thái-hành động. Chúng kết nối giá trị của một trạng thái với giá trị của các trạng thái kế tiếp.

**Bellman Equation cho Value Function (V-function)**:
> "Giá trị của trạng thái hiện tại (`s`) bằng phần thưởng trước mắt cộng với giá trị (đã chiết khấu) của trạng thái tiếp theo mà bạn có khả năng sẽ đến."

$$ V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')] $$

**Bellman Equation cho Q-Value Function (Q-function)**:
> "Giá trị của việc thực hiện hành động `a` tại trạng thái `s` bằng phần thưởng trước mắt cộng với giá trị (đã chiết khấu) của cặp (trạng thái, hành động) tốt nhất ở bước tiếp theo."

$$ Q^\pi(s, a) = \sum_{s' \in S} P(s'|s, a) [R(s, a, s') + \gamma \sum_{a' \in A} \pi(a'|s') Q^\pi(s', a')] $$

**Phương trình Bellman tối ưu (Bellman Optimality Equations)**:
Đây là trường hợp đặc biệt khi chúng ta đi theo chính sách tối ưu (chọn hành động tốt nhất ở mỗi bước).

$$ V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s'|s, a) [R(s, a, s') + \gamma V^*(s')] $$
$$ Q^*(s, a) = \sum_{s' \in S} P(s'|s, a) [R(s, a, s') + \gamma \max_{a' \in A} Q^*(s', a')] $$

Đây chính là công thức nền tảng cho thuật toán **Q-Learning**. Nó cho phép chúng ta cập nhật giá trị Q của một cặp `(s, a)` dựa trên giá trị Q tối đa có thể đạt được ở trạng thái tiếp theo `s'`.

---

## ⚙️ 4. Thẻ thuật toán - Q-Learning

### 1. Bài toán & dữ liệu
- **Bài toán**: Tìm ra chính sách tối ưu $\pi^*$ trong một môi trường MDP mà không cần biết trước mô hình của môi trường (Transition-Probabilities `P` và Reward Function `R`).
- **Dữ liệu**: Các bộ `(state, action, reward, next_state)` mà agent thu thập được qua quá trình tương tác (thử và sai).
- **Ứng dụng**: Các bài toán điều khiển đơn giản, game, robot tìm đường.

### 2. Mô hình & công thức
- **Mô hình**: Một bảng (Q-table) lưu trữ giá trị $Q(s, a)$ cho mọi cặp (trạng thái, hành động).
- **Công thức cập nhật (dựa trên Bellman Optimality)**:
$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right] $$

### 3. Loss & mục tiêu
- **Mục tiêu**: Làm cho $Q(s, a)$ hội tụ về giá trị tối ưu $Q^*(s, a)$.
- **Loss (Temporal Difference Error)**:
  $$ \text{TD Error} = \underbrace{r_t + \gamma \max_{a} Q(s_{t+1}, a)}_{\text{TD Target}} - \underbrace{Q(s_t, a_t)}_{\text{Old Value}} $$
  Mục tiêu là giảm thiểu sai số này.

### 4. Tối ưu hoá & cập nhật
- **Algorithm**: Temporal Difference (TD) Learning. Cập nhật giá trị Q dựa trên ước tính hiện tại, không cần đợi đến cuối một episode.
- **Chính sách hành động**: Thường là **Epsilon-Greedy** để cân bằng giữa Exploration và Exploitation.
  - Với xác suất `1-ε`: Chọn hành động tốt nhất (khai thác).
  - Với xác suất `ε`: Chọn một hành động ngẫu nhiên (khám phá).

### 5. Hyperparams
- **Learning rate ($\alpha$)**: Tốc độ học (0.01-0.1).
- **Discount factor ($\gamma$)**: Tầm quan trọng của phần thưởng tương lai (0.9-0.99).
- **Epsilon ($\epsilon$)**: Tỷ lệ khám phá, thường giảm dần theo thời gian.

### 6. Độ phức tạp
- **Time**: $O(A)$ cho mỗi bước cập nhật (để tìm `max Q`).
- **Space**: $O(S \times A)$ để lưu Q-table. Đây là hạn chế lớn nhất.

### 7. Metrics đánh giá
- **Tổng phần thưởng mỗi episode**: Phải có xu hướng tăng lên.
- **Số bước để hoàn thành episode**: Phải có xu hướng giảm đi (cho các bài toán có mục tiêu).
- **Sự hội tụ của Q-table**: Các giá trị Q có ổn định sau một thời gian không.

### 8. Ưu / Nhược điểm
**Ưu điểm**:
- Đơn giản, dễ hiểu.
- **Off-policy**: Rất mạnh mẽ, cho phép học từ kinh nghiệm cũ hoặc từ các agent khác.
- Đảm bảo hội tụ nếu các cặp (s, a) được ghé thăm đủ nhiều.

**Nhược điểm**:
- Không thể hoạt động với không gian trạng thái/hành động lớn hoặc liên tục (do Q-table quá lớn).
- Gặp khó khăn trong môi trường có tính ngẫu nhiên cao.

### 9. Bẫy & mẹo
- **Bẫy**: Learning rate quá lớn có thể làm giá trị Q không ổn định.
- **Mẹo**: Giảm dần `epsilon` theo thời gian. Ban đầu khám phá nhiều, sau đó tập trung khai thác.
- **Mẹo**: Khởi tạo Q-table một cách lạc quan (với giá trị cao) để khuyến khích khám phá.

### 10. Pseudocode:
```python
initialize Q(s, a) arbitrarily
for each episode:
    initialize s
    for each step of episode:
        choose a from s using policy derived from Q (e.g., ε-greedy)
        take action a, observe r, s'
        Q(s, a) <- Q(s, a) + α[r + γ * max_a'(Q(s', a')) - Q(s, a)]
        s <- s'
    until s is terminal
```

## ⚙️ 5. Thuật toán dựa trên giá trị (Value-Based Algorithms)

> **Tư tưởng cốt lõi**: Thay vì cố gắng học trực tiếp một chính sách (policy), các thuật toán này tập trung vào việc học một **hàm giá trị**. Sau khi có được hàm giá trị tối ưu, chính sách tối ưu sẽ tự động xuất hiện: chỉ cần chọn hành động dẫn đến trạng thái có giá trị cao nhất.

### 5.1 Q-Learning
Q-Learning là thuật toán RL kinh điển. Mục tiêu của nó là học hàm **$Q^*(s, a)$**, là giá trị tối ưu của việc thực hiện hành động `a` trong trạng thái `s`.

-   **Q-Table**: Trong các môi trường đơn giản, ta có thể dùng một bảng để lưu giá trị Q cho mọi cặp (trạng thái, hành động).
-   **Công thức cập nhật**: Trái tim của Q-learning, dựa trên phương trình Bellman:
    `New_Q(s, a) = Old_Q(s, a) + α * [Reward + γ * max_Q(s', a') - Old_Q(s, a)]`
-   **Off-Policy**: Điểm mạnh nhất của Q-Learning. Nó có thể học chính sách tối ưu (luôn chọn hành động `max_Q`) trong khi đang thực thi một chính sách khác để thu thập dữ liệu (ví dụ: ε-greedy để khám phá). Điều này giống như việc xem người khác chơi cờ để học nước đi hay nhất, trong khi chính bạn thỉnh thoảng lại đi những nước ngớ ngẩn để thử nghiệm.

### 5.2 Deep Q-Networks (DQN)

-   **Vấn đề với Q-Learning**: Q-table trở nên bất khả thi khi không gian trạng thái quá lớn (ví dụ: màn hình game Atari có hàng triệu pixel).
-   **Giải pháp của DQN**: Dùng một **mạng nơ-ron** để **xấp xỉ** hàm Q-value. Mạng này nhận đầu vào là trạng thái `s` và trả về một vector các giá trị Q cho tất cả các hành động có thể có.
    $$ Q(s, a; \theta) \approx Q^*(s, a) $$
-   **Hai cải tiến đột phá để ổn định training**:
    1.  **Experience Replay**: Thay vì học ngay từ trải nghiệm vừa có, agent lưu lại các transition `(s, a, r, s')` vào một "bộ nhớ" (replay buffer). Khi training, nó lấy ra một mini-batch ngẫu nhiên từ bộ nhớ này.
        *   **Tại sao?** Việc này phá vỡ sự tương quan giữa các mẫu dữ liệu liên tiếp, giúp quá trình học ổn định hơn và hiệu quả hơn về mặt dữ liệu.
    2.  **Fixed Q-Targets**: Sử dụng hai mạng nơ-ron: một mạng chính (`Q_online`) được cập nhật liên tục, và một mạng mục tiêu (`Q_target`) được sao chép từ mạng chính sau mỗi `C` bước.
        *   **Tại sao?** Khi tính toán TD Target (`r + γ * max_Q(s')`), ta dùng mạng `Q_target` cũ và ổn định. Điều này ngăn chặn việc "mục tiêu" liên tục thay đổi, giống như việc bạn cố bắn vào một tấm bia đang di chuyển. Nó giúp quá trình training ổn định hơn rất nhiều.
    *   **Loss Function**: Thường là Mean Squared Error giữa TD Target và dự đoán của mạng online.
        $$ L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( \underbrace{r + \gamma \max_{a'} Q_{\text{target}}(s', a'; \theta^-)}_{\text{TD Target}} - \underbrace{Q_{\text{online}}(s, a; \theta)}_{\text{Prediction}} \right)^2 \right] $$

## 📈 6. Thuật toán dựa trên chính sách (Policy-Based Algorithms)

> **Tư tưởng cốt lõi**: Thay vì học hàm giá trị, các thuật toán này **học trực tiếp chính sách (policy) $\pi(a|s; \theta)$**. Mô hình sẽ là một hàm nhận vào trạng thái `s` và trả về một phân phối xác suất trên các hành động `a`.

-   **Ưu điểm**:
    -   Hoạt động tốt trong không gian hành động liên tục hoặc xác suất.
    -   Có thể học các chính sách ngẫu nhiên (stochastic policies), hữu ích trong một số môi trường.
-   **Thách thức**: Phương sai (variance) của gradient thường rất cao, khiến việc training không ổn định.

### 6.1 Policy Gradient Theorem
Đây là định lý nền tảng cho các thuật toán Policy-Based. Nó cung cấp một cách để tính gradient của tổng phần thưởng kỳ vọng theo các tham số $\theta$ của chính sách.

-   **Tư duy trực quan**: "Tăng xác suất của những hành động dẫn đến phần thưởng cao, và giảm xác suất của những hành động dẫn đến phần thưởng thấp."
-   **Công thức Gradient**:
    $$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) \left( \sum_{t=0}^{T} r(s_t, a_t) \right) \right] $$
    -   $\nabla_\theta \log \pi_\theta(a_t|s_t)$: Cho biết hướng để tăng xác suất của hành động $a_t$ tại trạng thái $s_t$.
    -   $\sum r(s_t, a_t)$: Tổng phần thưởng của cả một episode (trajectory $\tau$).
    -   Về cơ bản, ta điều chỉnh tham số theo hướng `log-probability` của các hành động, được "cân" bởi tổng phần thưởng nhận được.

### 6.2 REINFORCE Algorithm
REINFORCE là thuật toán Policy Gradient đơn giản nhất.
1.  Chạy một episode hoàn chỉnh theo chính sách hiện tại $\pi_\theta$ để thu thập một trajectory `(s0, a0, r1, s1, a1, ...)`
2.  Tính tổng phần thưởng (return) $G_t$ từ mỗi time step `t` đến cuối.
3.  Cập nhật tham số $\theta$ bằng cách sử dụng gradient ascent:
    $$ \theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t $$

## 🎭 7. Thuật toán Actor-Critic

> **Tư tưởng cốt lõi**: Kết hợp những điểm mạnh nhất của hai phương pháp Value-Based và Policy-Based.

Mô hình Actor-Critic có hai "bộ não" riêng biệt:
1.  **The Actor (Diễn viên)**: Là **policy** $\pi(a|s; \theta)$. Nó chịu trách nhiệm chọn hành động.
2.  **The Critic (Nhà phê bình)**: Là **value function** $V(s; w)$ hoặc $Q(s, a; w)$. Nó chịu trách nhiệm "phê bình" hành động của Actor bằng cách đánh giá xem hành động đó tốt đến đâu.

**Luồng hoạt động**:
1.  **Actor** chọn hành động $a$ tại trạng thái $s$.
2.  Agent thực hiện hành động, nhận được phần thưởng $r$ và trạng thái mới $s'$.
3.  **Critic** tính toán "TD Error": $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
    -   TD Error cho biết hành động vừa rồi "tốt hơn" hay "tệ hơn" so với kỳ vọng.
4.  **Cập nhật Critic**: Dùng TD Error để cập nhật Critic, giúp nó đưa ra những lời phê bình chính xác hơn trong tương lai.
5.  **Cập nhật Actor**: Dùng TD Error để cập nhật Actor.
    -   Nếu $\delta_t > 0$ (hành động tốt hơn kỳ vọng), tăng xác suất chọn hành động đó.
    -   Nếu $\delta_t < 0$ (hành động tệ hơn kỳ vọng), giảm xác suất chọn hành động đó.

### 7.1 PPO (Proximal Policy Optimization)
PPO là một thuật toán Actor-Critic hiện đại và là một trong những thuật toán được sử dụng rộng rãi nhất hiện nay.

-   **Vấn đề của Policy Gradient truyền thống**: Việc cập nhật policy có thể quá lớn, khiến chính sách mới hoàn toàn khác chính sách cũ, gây ra sự sụp đổ trong quá trình training.
-   **Giải pháp của PPO**: Sử dụng một "Clipped Surrogate Objective Function" để đảm bảo rằng mỗi lần cập nhật policy chỉ diễn ra trong một "vùng an toàn" nhỏ. Nó ngăn không cho chính sách thay đổi quá đột ngột, giúp quá trình học ổn định hơn rất nhiều.

## 🌍 8. Ứng dụng, Bài tập và Tham khảo

### 8.1 Ứng dụng thực tế
-   **Chơi game**: AlphaGo của DeepMind đánh bại kỳ thủ cờ vây thế giới; các agent chơi game Atari, Dota 2, StarCraft.
-   **Robotics**: Dạy robot cách đi lại, cầm nắm đồ vật, lắp ráp.
-   **Tài chính**: Tối ưu hóa danh mục đầu tư, giao dịch tự động.
-   **Quản lý tài nguyên**: Tối ưu hóa hoạt động của các trung tâm dữ liệu (Google DeepMind), quản lý lưới điện.

### 8.2 Bài tập thực hành
1.  **Grid World**: Implement thuật toán Q-Learning từ đầu để tìm đường đi ngắn nhất trong một mê cung đơn giản.
2.  **CartPole**: Sử dụng thư viện `gymnasium`, huấn luyện một agent DQN để giữ thăng bằng cho cây cột.
3.  **Policy Gradient**: Implement thuật toán REINFORCE cho bài toán CartPole và so sánh kết quả với DQN.
4.  **PPO**: Sử dụng một thư viện RL (như `stable-baselines3`) để huấn luyện một agent PPO trên một môi trường phức tạp hơn (ví dụ: BipedalWalker).

### 8.3 Tài liệu tham khảo
-   **Sách**: "Reinforcement Learning: An Introduction" của Sutton và Barto (được coi là "kinh thánh" của RL).
-   **Khóa học**:
    -   David Silver's Reinforcement Learning Course (DeepMind/UCL).
    -   CS285 Deep Reinforcement Learning (UC Berkeley).
-   **Thư viện**: `gymnasium`, `stable-baselines3`, `rl-baselines3-zoo`.
-   **Bài báo quan trọng**:
    -   "Playing Atari with Deep Reinforcement Learning" (DQN paper).
    -   "Asynchronous Methods for Deep Reinforcement Learning" (A3C paper).
    -   "Proximal Policy Optimization Algorithms" (PPO paper).

---
*Chúc bạn học tập hiệu quả! 🚀*

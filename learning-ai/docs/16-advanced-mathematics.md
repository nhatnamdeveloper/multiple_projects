# 🧮 Toán học nâng cao cho AI/ML

> **Mục tiêu**: Hiểu sâu các khái niệm toán học cốt lõi trong AI/ML, từ lý thuyết đến ứng dụng thực tế

## 📋 Tổng quan nội dung

```mermaid
graph TD
    A[🧮 Toán học nâng cao] --> B[📐 Đại số tuyến tính nâng cao]
    A --> C[📈 Giải tích & Tối ưu hóa]
    A --> D[🎲 Xác suất & Thống kê nâng cao]
    A --> E[🌐 Lý thuyết thông tin]
    A --> F[📊 Lý thuyết học máy]
    
    B --> B1[SVD & Matrix Factorizations]
    B --> B2[Eigenvalue Theory]
    B --> B3[Tensor Operations]
    
    C --> C1[Convex Optimization]
    C --> C2[Gradient Methods]
    C --> C3[Lagrange Multipliers]
    
    D --> D1[Bayesian Inference]
    D --> D2[Statistical Learning Theory]
    D --> D3[Information Theory]
    
    E --> E1[Entropy & Mutual Information]
    E --> E2[KL Divergence]
    E --> E3[Rate-Distortion Theory]
    
    F --> F1[VC Dimension]
    F --> F2[Generalization Bounds]
    F --> F3[PAC Learning]
```

## 🧩 Chương trình 50/50 (Lý thuyết : Thực hành)

| Mô-đun | Lý thuyết (50%) | Thực hành (50%) |
|---|---|---|
| Đại số tuyến tính | SVD, Eigenvalue, Tensor | PCA, LDA, Matrix operations |
| Giải tích | Convexity, Optimization | Gradient descent, Newton method |
| Xác suất | Bayesian, Information theory | MCMC, Entropy calculation |
| Lý thuyết học | VC dimension, PAC learning | Generalization bounds |

---

## 📐 1. Đại số tuyến tính nâng cao

### 1.1 Singular Value Decomposition (SVD)

**Định nghĩa toán học:**
```python
import numpy as np
from scipy import linalg

def demonstrate_svd():
    """Demonstrate SVD decomposition"""
    # Tạo ma trận mẫu
    A = np.random.randn(10, 8)
    
    # SVD decomposition
    U, s, Vt = linalg.svd(A, full_matrices=False)
    
    # Kiểm tra reconstruction
    A_reconstructed = U @ np.diag(s) @ Vt
    
    print(f"Original matrix shape: {A.shape}")
    print(f"U shape: {U.shape}, s length: {len(s)}, Vt shape: {Vt.shape}")
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")
    
    return U, s, Vt, A_reconstructed
```

**Ứng dụng trong ML:**
- **Dimensionality Reduction**: PCA sử dụng SVD
- **Recommendation Systems**: Matrix factorization
- **Image Compression**: SVD cho ma trận ảnh
- **Natural Language Processing**: Latent Semantic Analysis (LSA)

### 1.2 Eigenvalue Theory & Spectral Analysis

**Eigenvalue decomposition:**
```python
def spectral_analysis():
    """Spectral analysis of matrices"""
    # Tạo ma trận đối xứng
    n = 100
    A = np.random.randn(n, n)
    A = (A + A.T) / 2  # Đối xứng
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = linalg.eigh(A)
    
    # Spectral properties
    spectral_radius = np.max(np.abs(eigenvalues))
    condition_number = np.max(eigenvalues) / np.min(eigenvalues)
    
    print(f"Spectral radius: {spectral_radius:.4f}")
    print(f"Condition number: {condition_number:.4f}")
    
    return eigenvalues, eigenvectors
```

### 1.3 Tensor Operations

**Tensor basics:**
```python
def tensor_operations():
    """Basic tensor operations"""
    # 3D tensor (batch, height, width)
    tensor = np.random.randn(32, 64, 64)
    
    # Tensor reshaping
    flattened = tensor.reshape(32, -1)  # Flatten spatial dimensions
    transposed = np.transpose(tensor, (0, 2, 1))  # Swap height/width
    
    # Tensor contraction (Einstein summation)
    # C_ij = A_ik B_kj
    A = np.random.randn(10, 5)
    B = np.random.randn(5, 8)
    C = np.einsum('ik,kj->ij', A, B)
    
    return tensor, flattened, transposed, C
```

---

## 📈 2. Giải tích & Tối ưu hóa nâng cao

### 2.1 Convex Optimization

**Định nghĩa hàm lồi:**
```python
def convex_function_demo():
    """Demonstrate convex functions"""
    import matplotlib.pyplot as plt
    
    x = np.linspace(-2, 2, 100)
    
    # Hàm lồi: f(x) = x²
    f_convex = x**2
    
    # Hàm không lồi: f(x) = sin(x)
    f_nonconvex = np.sin(x)
    
    # Kiểm tra tính lồi bằng định nghĩa
    def is_convex(f, x):
        """Check if function is convex using definition"""
        alpha = 0.5
        for i in range(1, len(x)-1):
            x1, x2 = x[i-1], x[i+1]
            f1, f2 = f[i-1], f[i+1]
            f_mid = f[i]
            
            # Jensen's inequality: f(αx1 + (1-α)x2) ≤ αf(x1) + (1-α)f(x2)
            if f_mid > alpha * f1 + (1-alpha) * f2:
                return False
        return True
    
    print(f"x² is convex: {is_convex(f_convex, x)}")
    print(f"sin(x) is convex: {is_convex(f_nonconvex, x)}")
    
    return x, f_convex, f_nonconvex
```

### 2.2 Gradient Methods & Optimization

**Advanced gradient methods:**
```python
def advanced_optimization():
    """Advanced optimization methods"""
    
    def rosenbrock(x):
        """Rosenbrock function"""
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_gradient(x):
        """Gradient of Rosenbrock function"""
        dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dy = 200 * (x[1] - x[0]**2)
        return np.array([dx, dy])
    
    def rosenbrock_hessian(x):
        """Hessian of Rosenbrock function"""
        H = np.zeros((2, 2))
        H[0, 0] = 2 + 1200 * x[0]**2 - 400 * x[1]
        H[0, 1] = H[1, 0] = -400 * x[0]
        H[1, 1] = 200
        return H
    
    # Gradient Descent
    def gradient_descent(f, grad_f, x0, lr=0.001, max_iter=1000):
        x = x0.copy()
        trajectory = [x.copy()]
        
        for i in range(max_iter):
            grad = grad_f(x)
            x = x - lr * grad
            trajectory.append(x.copy())
            
            if np.linalg.norm(grad) < 1e-6:
                break
        
        return np.array(trajectory)
    
    # Newton's Method
    def newton_method(f, grad_f, hess_f, x0, max_iter=100):
        x = x0.copy()
        trajectory = [x.copy()]
        
        for i in range(max_iter):
            grad = grad_f(x)
            hess = hess_f(x)
            
            # Newton step: x = x - H⁻¹∇f(x)
            try:
                step = np.linalg.solve(hess, grad)
                x = x - step
                trajectory.append(x.copy())
                
                if np.linalg.norm(step) < 1e-6:
                    break
            except np.linalg.LinAlgError:
                print("Hessian is singular")
                break
        
        return np.array(trajectory)
    
    # Test optimization methods
    x0 = np.array([-1.5, -1.5])
    
    gd_traj = gradient_descent(rosenbrock, rosenbrock_gradient, x0)
    newton_traj = newton_method(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0)
    
    print(f"Gradient descent final: {gd_traj[-1]}")
    print(f"Newton method final: {newton_traj[-1]}")
    print(f"True minimum: [1, 1]")
    
    return gd_traj, newton_traj
```

### 2.3 Lagrange Multipliers & Constrained Optimization

**Constrained optimization:**
```python
def lagrange_multipliers():
    """Lagrange multipliers example"""
    
    def objective(x):
        """Objective function: f(x,y) = x² + y²"""
        return x[0]**2 + x[1]**2
    
    def constraint(x):
        """Constraint: g(x,y) = x + y - 1 = 0"""
        return x[0] + x[1] - 1
    
    def lagrange_function(x, lambda_val):
        """Lagrangian: L(x,λ) = f(x) + λg(x)"""
        return objective(x) + lambda_val * constraint(x)
    
    def solve_lagrange():
        """Solve using Lagrange multipliers"""
        # From ∇f + λ∇g = 0 and g(x) = 0
        # We get: 2x = λ, 2y = λ, x + y = 1
        # Therefore: x = y = 0.5, λ = 1
        
        x_opt = np.array([0.5, 0.5])
        lambda_opt = 1.0
        
        print(f"Optimal point: {x_opt}")
        print(f"Lagrange multiplier: {lambda_opt}")
        print(f"Objective value: {objective(x_opt)}")
        print(f"Constraint value: {constraint(x_opt)}")
        
        return x_opt, lambda_opt
    
    return solve_lagrange()
```

---

## 🎲 3. Xác suất & Thống kê nâng cao

### 3.1 Bayesian Inference

**Bayesian framework:**
```python
def bayesian_inference():
    """Bayesian inference demonstration"""
    
    # Prior distribution: Beta(α=2, β=5)
    alpha_prior, beta_prior = 2, 5
    
    # Data: 10 successes out of 20 trials
    n_success, n_trials = 10, 20
    
    # Likelihood: Binomial(n=20, p=θ)
    def likelihood(theta, n_success, n_trials):
        """Binomial likelihood"""
        from scipy.special import comb
        return comb(n_trials, n_success) * theta**n_success * (1-theta)**(n_trials-n_success)
    
    # Posterior: Beta(α + n_success, β + n_trials - n_success)
    alpha_posterior = alpha_prior + n_success
    beta_posterior = beta_prior + (n_trials - n_success)
    
    # Posterior mean and variance
    posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
    posterior_var = (alpha_posterior * beta_posterior) / ((alpha_posterior + beta_posterior)**2 * (alpha_posterior + beta_posterior + 1))
    
    print(f"Prior: Beta({alpha_prior}, {beta_prior})")
    print(f"Data: {n_success} successes out of {n_trials} trials")
    print(f"Posterior: Beta({alpha_posterior}, {beta_posterior})")
    print(f"Posterior mean: {posterior_mean:.4f}")
    print(f"Posterior std: {np.sqrt(posterior_var):.4f}")
    
    return alpha_posterior, beta_posterior, posterior_mean, posterior_var
```

### 3.2 Statistical Learning Theory

**VC Dimension & Generalization:**
```python
def statistical_learning_theory():
    """Statistical learning theory concepts"""
    
    def vc_dimension_example():
        """VC dimension example for linear classifiers in 2D"""
        # Linear classifiers in 2D can shatter 3 points but not 4 points
        # Therefore VC dimension = 3
        
        def can_shatter(n_points):
            """Check if n points can be shattered by linear classifiers"""
            if n_points <= 3:
                return True
            elif n_points == 4:
                # XOR pattern cannot be shattered by linear classifiers
                return False
            else:
                return False
        
        print("VC dimension for linear classifiers in 2D:")
        for n in range(1, 6):
            print(f"  {n} points: {'Yes' if can_shatter(n) else 'No'}")
    
    def generalization_bound():
        """Generalization bound using VC dimension"""
        # VC dimension
        d_vc = 3
        
        # Sample size
        n = 1000
        
        # Confidence level
        delta = 0.05
        
        # Generalization bound: R(h) ≤ R_emp(h) + √((d_vc * log(2n/d_vc) + log(1/δ))/n)
        empirical_risk = 0.1  # Assume 10% training error
        
        generalization_bound = empirical_risk + np.sqrt((d_vc * np.log(2*n/d_vc) + np.log(1/delta)) / n)
        
        print(f"VC dimension: {d_vc}")
        print(f"Sample size: {n}")
        print(f"Confidence: {1-delta:.1%}")
        print(f"Empirical risk: {empirical_risk:.3f}")
        print(f"Generalization bound: {generalization_bound:.3f}")
        
        return generalization_bound
    
    vc_dimension_example()
    return generalization_bound()
```

### 3.3 Information Theory

**Entropy & Mutual Information:**
```python
def information_theory():
    """Information theory concepts"""
    
    def entropy(p):
        """Calculate entropy H(X) = -Σ p(x) log p(x)"""
        # Remove zero probabilities to avoid log(0)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    def mutual_information(p_xy, p_x, p_y):
        """Calculate mutual information I(X;Y)"""
        # I(X;Y) = Σ p(x,y) log(p(x,y)/(p(x)p(y)))
        mutual_info = 0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[i,j] > 0:
                    mutual_info += p_xy[i,j] * np.log2(p_xy[i,j] / (p_x[i] * p_y[j]))
        return mutual_info
    
    def kl_divergence(p, q):
        """Calculate KL divergence D_KL(P||Q)"""
        # D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        return np.sum(p * np.log2(p / q))
    
    # Example: Binary random variables
    p_x = np.array([0.7, 0.3])  # P(X=0) = 0.7, P(X=1) = 0.3
    p_y = np.array([0.6, 0.4])  # P(Y=0) = 0.6, P(Y=1) = 0.4
    
    # Joint distribution (example)
    p_xy = np.array([[0.5, 0.2], [0.1, 0.2]])
    
    # Calculate entropies
    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_xy = entropy(p_xy.flatten())
    
    # Calculate mutual information
    i_xy = mutual_information(p_xy, p_x, p_y)
    
    print(f"Entropy H(X): {h_x:.4f}")
    print(f"Entropy H(Y): {h_y:.4f}")
    print(f"Joint entropy H(X,Y): {h_xy:.4f}")
    print(f"Mutual information I(X;Y): {i_xy:.4f}")
    print(f"Verification: I(X;Y) = H(X) + H(Y) - H(X,Y) = {h_x + h_y - h_xy:.4f}")
    
    return h_x, h_y, h_xy, i_xy
```

---

## 📊 4. Lý thuyết học máy

### 4.1 PAC Learning

**Probably Approximately Correct Learning:**
```python
def pac_learning():
    """PAC learning framework"""
    
    def pac_bound(epsilon, delta, d_vc, n):
        """PAC generalization bound"""
        # With probability at least 1-δ:
        # R(h) ≤ R_emp(h) + √((d_vc * log(2n/d_vc) + log(1/δ))/n)
        
        empirical_risk = 0.1  # Assume 10% training error
        generalization_bound = empirical_risk + np.sqrt((d_vc * np.log(2*n/d_vc) + np.log(1/delta)) / n)
        
        return generalization_bound
    
    def sample_complexity(epsilon, delta, d_vc):
        """Sample complexity for PAC learning"""
        # n ≥ (1/ε²) * (d_vc * log(1/ε) + log(1/δ))
        n = (1/epsilon**2) * (d_vc * np.log(1/epsilon) + np.log(1/delta))
        return int(np.ceil(n))
    
    # Example parameters
    epsilon = 0.1  # Accuracy parameter
    delta = 0.05   # Confidence parameter
    d_vc = 3       # VC dimension
    
    # Calculate sample complexity
    n_required = sample_complexity(epsilon, delta, d_vc)
    
    # Calculate generalization bound for given sample size
    n_actual = 1000
    gen_bound = pac_bound(epsilon, delta, d_vc, n_actual)
    
    print(f"PAC Learning Parameters:")
    print(f"  Accuracy (ε): {epsilon}")
    print(f"  Confidence (δ): {delta}")
    print(f"  VC dimension: {d_vc}")
    print(f"  Required sample size: {n_required}")
    print(f"  Actual sample size: {n_actual}")
    print(f"  Generalization bound: {gen_bound:.4f}")
    
    return n_required, gen_bound
```

### 4.2 Rademacher Complexity

**Rademacher complexity bounds:**
```python
def rademacher_complexity():
    """Rademacher complexity demonstration"""
    
    def empirical_rademacher_complexity(hypothesis_class, data, n_samples=1000):
        """Estimate empirical Rademacher complexity"""
        n = len(data)
        rademacher_sum = 0
        
        for _ in range(n_samples):
            # Generate random Rademacher variables
            sigma = np.random.choice([-1, 1], size=n)
            
            # Find best hypothesis for this Rademacher sequence
            best_value = -np.inf
            for h in hypothesis_class:
                value = np.sum(sigma * h(data))
                best_value = max(best_value, value)
            
            rademacher_sum += best_value
        
        return rademacher_sum / (n * n_samples)
    
    # Example: Linear classifiers in 1D
    def linear_classifier_1d(data, threshold):
        """Linear classifier in 1D"""
        return np.where(data > threshold, 1, -1)
    
    # Generate sample data
    data = np.random.randn(100)
    
    # Define hypothesis class (linear classifiers with different thresholds)
    hypothesis_class = [lambda x, t=t: linear_classifier_1d(x, t) for t in np.linspace(-2, 2, 20)]
    
    # Estimate Rademacher complexity
    rad_complexity = empirical_rademacher_complexity(hypothesis_class, data)
    
    print(f"Estimated Rademacher complexity: {rad_complexity:.4f}")
    print(f"Theoretical bound: O(1/√n) ≈ {1/np.sqrt(len(data)):.4f}")
    
    return rad_complexity
```

---

## 🧪 5. Thực hành & Ứng dụng

### 5.1 PCA với SVD

```python
def pca_implementation():
    """PCA implementation using SVD"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    
    # Create data with structure
    data = np.random.randn(n_samples, n_features)
    
    # Add some correlation structure
    data[:, 2] = 0.8 * data[:, 0] + 0.2 * np.random.randn(n_samples)
    data[:, 3] = 0.6 * data[:, 1] + 0.4 * np.random.randn(n_samples)
    
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # PCA using SVD
    U, s, Vt = linalg.svd(data_centered, full_matrices=False)
    
    # Project data onto principal components
    data_pca = data_centered @ Vt.T
    
    # Explained variance
    explained_variance = s**2 / (n_samples - 1)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    
    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("PCA Results:")
    print(f"Original dimensions: {n_features}")
    print(f"Top 3 explained variance ratios: {explained_variance_ratio[:3]}")
    print(f"Cumulative variance (top 3): {cumulative_variance[:3]}")
    
    return data_pca, explained_variance_ratio, cumulative_variance
```

### 5.2 Bayesian Linear Regression

```python
def bayesian_linear_regression():
    """Bayesian linear regression implementation"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    x = np.random.randn(n_samples, 1)
    true_w = 2.0
    true_b = 1.0
    y = true_w * x + true_b + 0.1 * np.random.randn(n_samples, 1)
    
    # Prior parameters
    prior_w_mean = 0.0
    prior_w_var = 1.0
    prior_b_mean = 0.0
    prior_b_var = 1.0
    
    # Noise variance
    noise_var = 0.1**2
    
    # Bayesian update
    def bayesian_update(x, y, prior_mean, prior_var, noise_var):
        """Bayesian update for linear regression"""
        # Design matrix
        X = np.column_stack([x, np.ones_like(x)])
        
        # Posterior precision (inverse of covariance)
        posterior_precision = X.T @ X / noise_var + 1/prior_var
        posterior_cov = np.linalg.inv(posterior_precision)
        
        # Posterior mean
        posterior_mean = posterior_cov @ (X.T @ y / noise_var + prior_mean / prior_var)
        
        return posterior_mean, posterior_cov
    
    # Update for both parameters
    prior_mean = np.array([prior_w_mean, prior_b_mean])
    prior_var = np.array([prior_w_var, prior_b_var])
    
    posterior_mean, posterior_cov = bayesian_update(x, y, prior_mean, prior_var, noise_var)
    
    print("Bayesian Linear Regression:")
    print(f"True parameters: w={true_w}, b={true_b}")
    print(f"Posterior mean: w={posterior_mean[0]:.4f}, b={posterior_mean[1]:.4f}")
    print(f"Posterior std: w={np.sqrt(posterior_cov[0,0]):.4f}, b={np.sqrt(posterior_cov[1,1]):.4f}")
    
    return posterior_mean, posterior_cov
```

---

## 📚 Tài liệu tham khảo

### **Sách giáo khoa:**
- [Mathematics for Machine Learning](https://mml-book.github.io/) - Deisenroth, Faisal, Ong
- [Linear Algebra Done Right](https://linear.axler.net/) - Sheldon Axler
- [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) - Boyd & Vandenberghe
- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) - Bishop
- [Information Theory, Inference, and Learning Algorithms](https://www.inference.org.uk/mackay/itila/) - MacKay

### **Papers quan trọng:**
- [Statistical Learning Theory](https://www.springer.com/gp/book/9780387943274) - Vapnik
- [PAC Learning](https://en.wikipedia.org/wiki/Probably_approximately_correct_learning) - Valiant
- [Information Theory](https://ieeexplore.ieee.org/document/6773024) - Shannon

### **Online Resources:**
- [MIT OpenCourseWare - Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [Stanford CS229 - Machine Learning](http://cs229.stanford.edu/)
- [CMU 10-701 - Introduction to Machine Learning](https://www.cs.cmu.edu/~tom/10701_sp11/)

---

## 🎯 Bài tập thực hành

### **Bài tập 1: SVD Analysis**
1. Tạo ma trận 10x8 với rank thấp
2. Thực hiện SVD và phân tích singular values
3. Reconstruct với k singular values đầu tiên
4. Tính reconstruction error

### **Bài tập 2: Bayesian Inference**
1. Implement Bayesian linear regression
2. So sánh với frequentist approach
3. Visualize posterior distributions
4. Tính credible intervals

### **Bài tập 3: Information Theory**
1. Tính entropy cho các distribution khác nhau
2. Implement mutual information calculation
3. Analyze KL divergence between distributions
4. Apply to feature selection

### **Bài tập 4: Optimization**
1. Implement gradient descent với momentum
2. So sánh với Newton's method
3. Analyze convergence rates
4. Apply to logistic regression

---

*Chúc bạn học tập hiệu quả! 🚀*

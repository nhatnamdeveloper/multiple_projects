# 🤖 Recommendation Engine - Lý thuyết

> **Mục tiêu**: Xây dựng hệ thống gợi ý sản phẩm thông minh với nhiều thuật toán khác nhau

## 🧠 **Lý thuyết cơ bản**

### **1. Recommendation Systems Overview**

**Khái niệm cốt lõi:**
- **Collaborative Filtering**: Dựa trên hành vi của người dùng tương tự
- **Content-Based Filtering**: Dựa trên đặc tính sản phẩm và profile người dùng
- **Hybrid Approaches**: Kết hợp nhiều phương pháp
- **Matrix Factorization**: Phân tích ma trận tương tác người dùng-sản phẩm

### **2. Types of Recommendation Algorithms**

**A. Collaborative Filtering:**
- **User-Based CF**: Tìm người dùng tương tự, gợi ý sản phẩm họ đã mua
- **Item-Based CF**: Tìm sản phẩm tương tự, gợi ý dựa trên sản phẩm đã tương tác
- **Matrix Factorization**: SVD, NMF, ALS (Alternating Least Squares)

**B. Content-Based Filtering:**
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **Cosine Similarity**: Độ tương tự cosine giữa vectors
- **Feature Engineering**: Trích xuất đặc tính sản phẩm

**C. Deep Learning Approaches:**
- **Neural Collaborative Filtering (NCF)**: Kết hợp MLP với matrix factorization
- **Wide & Deep**: Kết hợp linear model với deep neural network
- **Two-Tower Models**: User tower và item tower riêng biệt

### **3. Evaluation Metrics**

**Ranking Metrics:**
- **Precision@K**: Tỷ lệ sản phẩm gợi ý đúng trong top-K
- **Recall@K**: Tỷ lệ sản phẩm đúng được gợi ý trong top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision

**Business Metrics:**
- **Click-Through Rate (CTR)**: Tỷ lệ click vào gợi ý
- **Conversion Rate**: Tỷ lệ mua hàng từ gợi ý
- **Revenue Lift**: Tăng trưởng doanh thu từ gợi ý
- **Diversity**: Đa dạng sản phẩm được gợi ý

## 🔧 **Technical Architecture**

### **1. Recommendation System Architecture**

```python
class RecommendationArchitecture:
    """Architecture cho Recommendation System"""
    
    def __init__(self):
        self.components = {
            'data_collection': ['User Interactions', 'Product Metadata', 'User Profiles'],
            'feature_engineering': ['User Features', 'Item Features', 'Interaction Features'],
            'model_training': ['Collaborative Filtering', 'Content-Based', 'Deep Learning'],
            'model_serving': ['Real-time Inference', 'Batch Recommendations', 'A/B Testing'],
            'evaluation': ['Offline Metrics', 'Online Metrics', 'Business Impact']
        }
    
    def explain_data_flow(self):
        """Explain data flow trong hệ thống"""
        print("""
        **Recommendation System Data Flow:**
        
        1. **Data Collection Layer:**
           - User interactions (clicks, purchases, ratings)
           - Product metadata (category, price, brand, features)
           - User profiles (demographics, preferences, behavior)
        
        2. **Feature Engineering Layer:**
           - User embeddings (behavior patterns, preferences)
           - Item embeddings (product characteristics, popularity)
           - Interaction features (time, context, session)
        
        3. **Model Training Layer:**
           - Collaborative filtering models (SVD, ALS)
           - Content-based models (TF-IDF, embeddings)
           - Deep learning models (NCF, Wide & Deep)
        
        4. **Model Serving Layer:**
           - Real-time inference API
           - Batch recommendation generation
           - A/B testing framework
        
        5. **Evaluation Layer:**
           - Offline evaluation (precision, recall, NDCG)
           - Online evaluation (CTR, conversion rate)
           - Business metrics (revenue lift, user engagement)
        """)
```

### **2. Collaborative Filtering Implementation**

**User-Based CF:**
```python
class UserBasedCF:
    """User-Based Collaborative Filtering"""
    
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.user_similarities = None
    
    def calculate_user_similarity(self, user1_id, user2_id):
        """Calculate similarity between two users using cosine similarity"""
        user1_ratings = self.user_item_matrix[user1_id]
        user2_ratings = self.user_item_matrix[user2_id]
        
        # Find common items
        common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
        
        if len(common_items) == 0:
            return 0.0
        
        # Calculate cosine similarity
        numerator = sum(user1_ratings[item] * user2_ratings[item] for item in common_items)
        denominator = (sum(user1_ratings[item]**2 for item in common_items) ** 0.5) * \
                     (sum(user2_ratings[item]**2 for item in common_items) ** 0.5)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def find_similar_users(self, user_id, n_similar=10):
        """Find n most similar users"""
        similarities = []
        
        for other_user_id in self.user_item_matrix.index:
            if other_user_id != user_id:
                similarity = self.calculate_user_similarity(user_id, other_user_id)
                similarities.append((other_user_id, similarity))
        
        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def recommend_items(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        similar_users = self.find_similar_users(user_id)
        
        # Get items rated by similar users
        candidate_items = {}
        for similar_user_id, similarity in similar_users:
            user_ratings = self.user_item_matrix.loc[similar_user_id]
            
            for item_id, rating in user_ratings.items():
                if rating > 0:  # User has rated this item
                    if item_id not in candidate_items:
                        candidate_items[item_id] = []
                    candidate_items[item_id].append(rating * similarity)
        
        # Calculate weighted average ratings
        recommendations = []
        for item_id, weighted_ratings in candidate_items.items():
            avg_rating = sum(weighted_ratings) / len(weighted_ratings)
            recommendations.append((item_id, avg_rating))
        
        # Sort by rating and return top n
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
```

### **3. Matrix Factorization (SVD)**

**SVD Implementation:**
```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

class SVDRecommender:
    """Singular Value Decomposition for Recommendations"""
    
    def __init__(self, n_factors=100, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
    
    def fit(self, user_item_matrix):
        """Train SVD model"""
        n_users, n_items = user_item_matrix.shape
        
        # Initialize factors and biases
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = np.mean(user_item_matrix[user_item_matrix > 0])
        
        # Get non-zero ratings
        user_item_pairs = np.where(user_item_matrix > 0)
        
        # Stochastic gradient descent
        for epoch in range(self.n_epochs):
            for u, i in zip(user_item_pairs[0], user_item_pairs[1]):
                rating = user_item_matrix[u, i]
                
                # Predict rating
                pred = self.global_bias + self.user_biases[u] + self.item_biases[i] + \
                       np.dot(self.user_factors[u], self.item_factors[i])
                
                # Calculate error
                error = rating - pred
                
                # Update parameters
                self.global_bias += self.lr * error
                self.user_biases[u] += self.lr * (error - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (error - self.reg * self.item_biases[i])
                
                # Update factors
                user_factor_grad = error * self.item_factors[i] - self.reg * self.user_factors[u]
                item_factor_grad = error * self.user_factors[u] - self.reg * self.item_factors[i]
                
                self.user_factors[u] += self.lr * user_factor_grad
                self.item_factors[i] += self.lr * item_factor_grad
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        pred = self.global_bias + self.user_biases[user_id] + self.item_biases[item_id] + \
               np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return max(1, min(5, pred))  # Clip to rating range
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        predictions = []
        
        for item_id in range(self.item_factors.shape[0]):
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
```

### **4. Content-Based Filtering**

**TF-IDF Content-Based:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentBasedRecommender:
    """Content-Based Recommendation using TF-IDF"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.item_features = None
        self.item_ids = None
    
    def fit(self, items_data):
        """Train content-based model"""
        # Extract text features from items
        item_descriptions = []
        self.item_ids = []
        
        for item_id, item_data in items_data.items():
            # Combine item features into text
            text_features = f"{item_data.get('name', '')} {item_data.get('category', '')} {item_data.get('description', '')} {item_data.get('brand', '')}"
            item_descriptions.append(text_features)
            self.item_ids.append(item_id)
        
        # Create TF-IDF matrix
        self.item_features = self.tfidf_vectorizer.fit_transform(item_descriptions)
    
    def get_item_similarity(self, item_id1, item_id2):
        """Calculate similarity between two items"""
        idx1 = self.item_ids.index(item_id1)
        idx2 = self.item_ids.index(item_id2)
        
        similarity = cosine_similarity(
            self.item_features[idx1:idx1+1], 
            self.item_features[idx2:idx2+1]
        )[0][0]
        
        return similarity
    
    def recommend_similar_items(self, item_id, n_recommendations=10):
        """Find similar items to a given item"""
        if item_id not in self.item_ids:
            return []
        
        item_idx = self.item_ids.index(item_id)
        item_vector = self.item_features[item_idx:item_idx+1]
        
        # Calculate similarities with all items
        similarities = cosine_similarity(item_vector, self.item_features).flatten()
        
        # Get top similar items (excluding the item itself)
        similar_indices = similarities.argsort()[-n_recommendations-1:-1][::-1]
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append((self.item_ids[idx], similarities[idx]))
        
        return recommendations
    
    def recommend_for_user_profile(self, user_profile, n_recommendations=10):
        """Recommend items based on user profile"""
        # Create user profile vector
        user_text = " ".join(user_profile.get('preferences', []))
        user_vector = self.tfidf_vectorizer.transform([user_text])
        
        # Calculate similarities with all items
        similarities = cosine_similarity(user_vector, self.item_features).flatten()
        
        # Get top recommendations
        top_indices = similarities.argsort()[-n_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append((self.item_ids[idx], similarities[idx]))
        
        return recommendations
```

## 📊 **Evaluation Framework**

### **1. Offline Evaluation**

**Metrics Implementation:**
```python
class RecommendationEvaluator:
    """Evaluation framework for recommendation systems"""
    
    def __init__(self, test_data, ground_truth):
        self.test_data = test_data
        self.ground_truth = ground_truth
    
    def precision_at_k(self, recommendations, k=10):
        """Calculate Precision@K"""
        if len(recommendations) == 0:
            return 0.0
        
        relevant_items = set(self.ground_truth)
        recommended_items = set([item_id for item_id, _ in recommendations[:k]])
        
        if len(recommended_items) == 0:
            return 0.0
        
        precision = len(relevant_items & recommended_items) / len(recommended_items)
        return precision
    
    def recall_at_k(self, recommendations, k=10):
        """Calculate Recall@K"""
        if len(recommendations) == 0:
            return 0.0
        
        relevant_items = set(self.ground_truth)
        recommended_items = set([item_id for item_id, _ in recommendations[:k]])
        
        if len(relevant_items) == 0:
            return 0.0
        
        recall = len(relevant_items & recommended_items) / len(relevant_items)
        return recall
    
    def ndcg_at_k(self, recommendations, k=10):
        """Calculate NDCG@K"""
        if len(recommendations) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, (item_id, score) in enumerate(recommendations[:k]):
            if item_id in self.ground_truth:
                dcg += score / np.log2(i + 2)  # log2(i+2) because i starts from 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        ideal_scores = sorted([score for _, score in recommendations if _ in self.ground_truth], reverse=True)
        for i, score in enumerate(ideal_scores[:k]):
            idcg += score / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def diversity(self, recommendations, item_features, k=10):
        """Calculate diversity of recommendations"""
        if len(recommendations) < 2:
            return 0.0
        
        recommended_items = [item_id for item_id, _ in recommendations[:k]]
        
        total_diversity = 0.0
        count = 0
        
        for i in range(len(recommended_items)):
            for j in range(i + 1, len(recommended_items)):
                item1_id = recommended_items[i]
                item2_id = recommended_items[j]
                
                # Calculate dissimilarity (1 - similarity)
                similarity = self._calculate_item_similarity(item1_id, item2_id, item_features)
                diversity = 1 - similarity
                
                total_diversity += diversity
                count += 1
        
        return total_diversity / count if count > 0 else 0.0
    
    def _calculate_item_similarity(self, item1_id, item2_id, item_features):
        """Calculate similarity between two items"""
        if item1_id not in item_features or item2_id not in item_features:
            return 0.0
        
        features1 = item_features[item1_id]
        features2 = item_features[item2_id]
        
        # Cosine similarity
        dot_product = sum(features1[k] * features2[k] for k in set(features1) & set(features2))
        norm1 = sum(features1[k]**2 for k in features1) ** 0.5
        norm2 = sum(features2[k]**2 for k in features2) ** 0.5
        
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0
```

## 🎯 **Business Impact**

### **Expected Outcomes:**
- **Increased Sales**: 15-30% increase in conversion rate
- **Better User Experience**: Personalized recommendations
- **Higher Engagement**: More time spent on platform
- **Reduced Churn**: Better user retention through relevant suggestions
- **Cross-selling**: Increased average order value

---

**📚 References:**
- "Recommender Systems: An Introduction" by Jannach et al.
- "Matrix Factorization Techniques for Recommender Systems" by Koren et al.
- "Deep Learning based Recommender System: A Survey and New Perspectives" by Zhang et al.
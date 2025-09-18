"""
Hybrid Recommendation System for Vendor Product Recommendations


This system combines Collaborative Filtering (CF) and Content-Based Filtering (CBF)
to recommend vendors the best products to upload to an e-commerce platform.

Features:
- Collaborative Filtering using Surprise library (SVD/KNNBaseline)
- Content-Based Filtering using product similarity
- Hybrid approach with weighted combination
- Comprehensive evaluation metrics
- FastAPI deployment ready


"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class HybridRecommendationSystem:
    """
    Hybrid Recommendation System combining Collaborative Filtering and Content-Based Filtering
    """
    
    def __init__(self, alpha=0.5):
        """
        Initialize the hybrid recommendation system
        
        Args:
            alpha (float): Weight for collaborative filtering (0-1)
                          alpha * CF_score + (1-alpha) * CBF_score
        """
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.vendor_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        
        # Models
        self.cf_model = None
        self.product_similarity_matrix = None
        self.vendor_product_matrix = None
        
        # Data
        self.df = None
        self.df_processed = None
        
    def load_data(self, filepath):
        """
        Load and preprocess the dataset
        
        Args:
            filepath (str): Path to the CSV file
        """
        print("Loading data...")
        self.df = pd.read_csv(filepath)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        
        # Display basic statistics
        print("\nDataset Info:")
        print(self.df.info())
        print("\nFirst few rows:")
        print(self.df.head())
        
        return self.df
    
    def preprocess_data(self):
        """
        Clean and preprocess the data
        """
        print("\nPreprocessing data...")
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # Handle missing values
        print("Checking for missing values...")
        print(df_processed.isnull().sum())
        
        # Fill missing values if any (only for numeric columns)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        
        # Normalize numeric features
        numeric_features = ['price', 'historical_sales', 'stock_level', 'rate']
        print(f"\nNormalizing numeric features: {numeric_features}")
        
        # Log transform price and historical_sales to handle skewness
        df_processed['price_log'] = np.log1p(df_processed['price'])
        df_processed['historical_sales_log'] = np.log1p(df_processed['historical_sales'])
        
        # Normalize features (only numeric columns)
        features_to_normalize = ['price_log', 'historical_sales_log', 'stock_level', 'rate']
        df_processed[features_to_normalize] = self.scaler.fit_transform(
            df_processed[features_to_normalize].values
        )
        
        # Encode categorical features
        print("Encoding categorical features...")
        df_processed['vendor_id_encoded'] = self.vendor_encoder.fit_transform(df_processed['vendor_id'])
        df_processed['product_id_encoded'] = self.product_encoder.fit_transform(df_processed['product_id'])
        df_processed['category_encoded'] = self.category_encoder.fit_transform(df_processed['category'])
        
        # Create vendor-product interaction matrix for collaborative filtering
        print("Creating vendor-product interaction matrix...")
        self.vendor_product_matrix = df_processed.pivot_table(
            index='vendor_id_encoded',
            columns='product_id_encoded',
            values='rate',  # Using rating as interaction strength
            fill_value=0
        )
        
        # Create success score based on historical sales and rating
        # Higher sales and rating = higher success probability
        df_processed['success_score'] = (
            df_processed['historical_sales_log'] * 0.4 +
            df_processed['rate'] * 0.6
        )
        
        self.df_processed = df_processed
        print(f"Processed dataset shape: {self.df_processed.shape}")
        
        return self.df_processed
    
    def train_collaborative_filtering(self):
        """
        Train Collaborative Filtering model using scikit-learn
        """
        print("\nTraining Collaborative Filtering model...")
        
        # Create vendor-product interaction matrix
        interaction_matrix = self.vendor_product_matrix.values
        
        # Handle missing values by filling with mean rating
        mean_rating = np.nanmean(interaction_matrix)
        interaction_matrix = np.nan_to_num(interaction_matrix, nan=mean_rating)
        
        # Train SVD model using scikit-learn
        print("Training SVD model...")
        self.cf_model = TruncatedSVD(n_components=50, random_state=42)
        
        # Fit the model
        self.cf_model.fit(interaction_matrix)
        
        # Transform the matrix to get latent factors
        self.vendor_factors = self.cf_model.transform(interaction_matrix)
        self.product_factors = self.cf_model.components_.T
        
        # Create reconstructed matrix for evaluation
        reconstructed_matrix = self.vendor_factors @ self.product_factors.T
        
        # Calculate RMSE
        mask = self.vendor_product_matrix.notna()
        actual_values = self.vendor_product_matrix.values[mask]
        predicted_values = reconstructed_matrix[mask]
        
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        
        print(f"CF RMSE: {rmse:.4f}")
        print(f"SVD Components: {self.cf_model.n_components}")
        print(f"Explained Variance Ratio: {self.cf_model.explained_variance_ratio_.sum():.4f}")
        
        return self.cf_model
    
    def train_content_based_filtering(self):
        """
        Train Content-Based Filtering model using product similarity
        """
        print("\nTraining Content-Based Filtering model...")
        
        # Create product feature matrix
        product_features = self.df_processed.groupby('product_id_encoded').agg({
            'category_encoded': 'first',
            'price_log': 'mean',
            'historical_sales_log': 'mean',
            'stock_level': 'mean',
            'rate': 'mean'
        }).reset_index()
        
        # Create feature matrix for similarity calculation
        feature_columns = ['category_encoded', 'price_log', 'historical_sales_log', 'stock_level', 'rate']
        product_feature_matrix = product_features[feature_columns].values
        
        # Calculate cosine similarity between products
        print("Calculating product similarity matrix...")
        self.product_similarity_matrix = cosine_similarity(product_feature_matrix)
        
        print(f"Product similarity matrix shape: {self.product_similarity_matrix.shape}")
        
        return self.product_similarity_matrix
    
    def get_cf_score(self, vendor_id, product_id):
        """
        Get Collaborative Filtering score for vendor-product pair
        
        Args:
            vendor_id (int): Vendor ID
            product_id (int): Product ID
            
        Returns:
            float: CF score
        """
        if self.cf_model is None or not hasattr(self, 'vendor_factors'):
            return 0.0
        
        try:
            # Get vendor and product encoded IDs
            vendor_encoded = self.vendor_encoder.transform([vendor_id])[0]
            product_encoded = self.product_encoder.transform([product_id])[0]
            
            # Get vendor and product factors
            vendor_factor = self.vendor_factors[vendor_encoded]
            product_factor = self.product_factors[product_encoded]
            
            # Calculate dot product for prediction
            prediction = np.dot(vendor_factor, product_factor)
            
            # Ensure prediction is within valid rating range
            prediction = np.clip(prediction, 1.0, 5.0)
            
            return prediction
        except:
            return 0.0
    
    def get_cbf_score(self, vendor_id, product_id, top_k=5):
        """
        Get Content-Based Filtering score for vendor-product pair
        
        Args:
            vendor_id (int): Vendor ID
            product_id (int): Product ID
            top_k (int): Number of similar products to consider
            
        Returns:
            float: CBF score
        """
        if self.product_similarity_matrix is None:
            return 0.0
        
        try:
            # Get vendor's successful products
            vendor_products = self.df_processed[
                self.df_processed['vendor_id'] == vendor_id
            ]['product_id_encoded'].unique()
            
            if len(vendor_products) == 0:
                return 0.0
            
            # Get product encoded ID
            product_encoded = self.product_encoder.transform([product_id])[0]
            
            # Calculate similarity with vendor's successful products
            similarities = []
            for vendor_product in vendor_products:
                if vendor_product < self.product_similarity_matrix.shape[0]:
                    sim = self.product_similarity_matrix[product_encoded][vendor_product]
                    similarities.append(sim)
            
            if len(similarities) == 0:
                return 0.0
            
            # Return average similarity
            return np.mean(similarities)
        except:
            return 0.0
    
    def get_hybrid_score(self, vendor_id, product_id):
        """
        Get hybrid score combining CF and CBF
        
        Args:
            vendor_id (int): Vendor ID
            product_id (int): Product ID
            
        Returns:
            float: Hybrid score
        """
        cf_score = self.get_cf_score(vendor_id, product_id)
        cbf_score = self.get_cbf_score(vendor_id, product_id)
        
        hybrid_score = self.alpha * cf_score + (1 - self.alpha) * cbf_score
        return hybrid_score, cf_score, cbf_score
    
    def recommend_products(self, vendor_id, top_k=10, category_filter=None):
        """
        Recommend top products for a vendor
        
        Args:
            vendor_id (int): Vendor ID
            top_k (int): Number of recommendations
            category_filter (str): Optional category filter
            
        Returns:
            list: List of recommended products with scores
        """
        print(f"\nGenerating recommendations for vendor {vendor_id}...")
        
        # Get all products (excluding those already uploaded by vendor)
        vendor_products = set(self.df_processed[
            self.df_processed['vendor_id'] == vendor_id
        ]['product_id'].tolist())
        
        all_products = set(self.df_processed['product_id'].unique())
        candidate_products = all_products - vendor_products
        
        # Apply category filter if specified
        if category_filter:
            candidate_products = [
                p for p in candidate_products
                if self.df_processed[
                    self.df_processed['product_id'] == p
                ]['category'].iloc[0] == category_filter
            ]
        
        # Calculate hybrid scores for all candidate products
        recommendations = []
        for product_id in candidate_products:
            hybrid_score, cf_score, cbf_score = self.get_hybrid_score(vendor_id, product_id)
            
            # Get product details
            product_info = self.df_processed[
                self.df_processed['product_id'] == product_id
            ].iloc[0]
            
            recommendations.append({
                'product_id': product_id,
                'category': product_info['category'],
                'price': product_info['price'],
                'historical_sales': product_info['historical_sales'],
                'stock_level': product_info['stock_level'],
                'rate': product_info['rate'],
                'hybrid_score': hybrid_score,
                'cf_score': cf_score,
                'cbf_score': cbf_score
            })
        
        # Sort by hybrid score and return top_k
        recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return recommendations[:top_k]
    
    def evaluate_model(self, test_size=0.2):
        """
        Evaluate the hybrid model using various metrics
        
        Args:
            test_size (float): Proportion of data for testing
            
        Returns:
            dict: Evaluation metrics
        """
        print("\nEvaluating hybrid model...")
        
        # Split data
        train_data, test_data = train_test_split(
            self.df_processed, test_size=test_size, random_state=42
        )
        
        # Create test set for evaluation
        test_vendors = test_data['vendor_id'].unique()
        
        # Calculate metrics
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for vendor_id in test_vendors[:10]:  # Evaluate on subset for efficiency
            # Get actual successful products for vendor
            actual_products = set(test_data[
                test_data['vendor_id'] == vendor_id
            ]['product_id'].tolist())
            
            if len(actual_products) == 0:
                continue
            
            # Get recommendations
            recommendations = self.recommend_products(vendor_id, top_k=10)
            recommended_products = set([r['product_id'] for r in recommendations])
            
            # Calculate metrics
            if len(recommended_products) > 0:
                precision = len(actual_products & recommended_products) / len(recommended_products)
                recall = len(actual_products & recommended_products) / len(actual_products)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
        
        # Calculate average metrics
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        
        evaluation_results = {
            'precision@10': avg_precision,
            'recall@10': avg_recall,
            'f1@10': avg_f1,
            'num_vendors_evaluated': len(precision_scores)
        }
        
        print(f"Evaluation Results:")
        print(f"Precision@10: {avg_precision:.4f}")
        print(f"Recall@10: {avg_recall:.4f}")
        print(f"F1@10: {avg_f1:.4f}")
        
        return evaluation_results
    
    def tune_alpha(self, alpha_values=[0.3, 0.4, 0.5, 0.6, 0.7]):
        """
        Tune the alpha parameter for hybrid model
        
        Args:
            alpha_values (list): List of alpha values to test
            
        Returns:
            dict: Best alpha and corresponding metrics
        """
        print("\nTuning alpha parameter...")
        
        best_alpha = self.alpha
        best_f1 = 0
        
        results = {}
        
        for alpha in alpha_values:
            print(f"Testing alpha = {alpha}")
            self.alpha = alpha
            
            # Evaluate model
            eval_results = self.evaluate_model()
            f1_score = eval_results['f1@10']
            
            results[alpha] = eval_results
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_alpha = alpha
        
        # Restore best alpha
        self.alpha = best_alpha
        
        print(f"Best alpha: {best_alpha}")
        print(f"Best F1@10: {best_f1:.4f}")
        
        return {
            'best_alpha': best_alpha,
            'best_f1': best_f1,
            'all_results': results
        }
    
    def save_model(self, filepath='hybrid_model.pkl'):
        """
        Save the trained model and preprocessing objects
        
        Args:
            filepath (str): Path to save the model
        """
        print(f"\nSaving model to {filepath}...")
        
        model_data = {
            'alpha': self.alpha,
            'scaler': self.scaler,
            'vendor_encoder': self.vendor_encoder,
            'product_encoder': self.product_encoder,
            'category_encoder': self.category_encoder,
            'cf_model': self.cf_model,
            'vendor_factors': getattr(self, 'vendor_factors', None),
            'product_factors': getattr(self, 'product_factors', None),
            'product_similarity_matrix': self.product_similarity_matrix,
            'vendor_product_matrix': self.vendor_product_matrix,
            'df_processed': self.df_processed
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved successfully!")
    
    def load_model(self, filepath='hybrid_model.pkl'):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to the saved model
        """
        print(f"Loading model from {filepath}...")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.alpha = model_data['alpha']
        self.scaler = model_data['scaler']
        self.vendor_encoder = model_data['vendor_encoder']
        self.product_encoder = model_data['product_encoder']
        self.category_encoder = model_data['category_encoder']
        self.cf_model = model_data['cf_model']
        self.vendor_factors = model_data.get('vendor_factors', None)
        self.product_factors = model_data.get('product_factors', None)
        self.product_similarity_matrix = model_data['product_similarity_matrix']
        self.vendor_product_matrix = model_data['vendor_product_matrix']
        self.df_processed = model_data['df_processed']
        
        print("Model loaded successfully!")


def main():
    """
    Main function to train and evaluate the hybrid recommendation system
    """
    print("=" * 60)
    print("HYBRID RECOMMENDATION SYSTEM FOR VENDOR PRODUCT RECOMMENDATIONS")
    print("=" * 60)
    
    # Initialize the system
    hybrid_system = HybridRecommendationSystem(alpha=0.5)
    
    # Load and preprocess data
    hybrid_system.load_data('reccommendation_datas.csv')
    hybrid_system.preprocess_data()
    
    # Train models
    hybrid_system.train_collaborative_filtering()
    hybrid_system.train_content_based_filtering()
    
    # Tune alpha parameter
    tuning_results = hybrid_system.tune_alpha()
    
    # Evaluate the final model
    evaluation_results = hybrid_system.evaluate_model()
    
    # Save the model
    hybrid_system.save_model('hybrid_model.pkl')
    
    # Example recommendations
    print("\n" + "=" * 60)
    print("EXAMPLE RECOMMENDATIONS")
    print("=" * 60)
    
    # Get recommendations for a few vendors
    sample_vendors = hybrid_system.df_processed['vendor_id'].unique()[:3]
    
    for vendor_id in sample_vendors:
        print(f"\nRecommendations for Vendor {vendor_id}:")
        recommendations = hybrid_system.recommend_products(vendor_id, top_k=5)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Product {rec['product_id']} ({rec['category']})")
            print(f"   Price: ${rec['price']:,.2f}")
            print(f"   Historical Sales: {rec['historical_sales']}")
            print(f"   Rating: {rec['rate']:.1f}")
            print(f"   Hybrid Score: {rec['hybrid_score']:.4f}")
            print(f"   CF Score: {rec['cf_score']:.4f}, CBF Score: {rec['cbf_score']:.4f}")
            print()
    
    print("\n" + "=" * 60)
    print("FUTURE WORK SUGGESTIONS")
    print("=" * 60)
    print("""
    Future enhancements to consider:
    
    1. Temporal Features:
       - Add seasonality patterns for product uploads
       - Include time-series trends for better timing recommendations
       - Consider day-of-week and month-of-year patterns
    
    2. Reinforcement Learning:
       - Implement real-time vendor feedback system
       - Use multi-armed bandit algorithms for dynamic recommendations
       - Add A/B testing framework for recommendation strategies
    
    3. Scalability Improvements:
       - Implement Apache Spark for big data processing
       - Use PyTorch/TensorFlow for deep learning models
       - Add distributed computing capabilities
    
    4. Advanced Features:
       - Include competitor analysis
       - Add market demand forecasting
       - Implement dynamic pricing recommendations
       - Add inventory optimization suggestions
    
    5. Real-time Processing:
       - Implement streaming data processing
       - Add real-time model updates
       - Include live performance monitoring
    """)


if __name__ == "__main__":
    main()

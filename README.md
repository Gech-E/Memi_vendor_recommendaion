# Hybrid Recommendation System for Vendor Product Recommendations

A comprehensive hybrid recommendation system that combines Collaborative Filtering (CF) and Content-Based Filtering (CBF) to recommend vendors the best products to upload to an e-commerce platform.

## 🚀 Features

- **Hybrid Approach**: Combines Collaborative Filtering and Content-Based Filtering
- **Advanced ML Models**: Uses SVD for collaborative filtering and cosine similarity for content-based filtering
- **Comprehensive Evaluation**: Implements Precision@K, Recall@K, F1@K, and RMSE metrics
- **REST API**: FastAPI-based web service for easy integration
- **Parameter Tuning**: Automatic alpha parameter optimization
- **Scalable Architecture**: Ready for production deployment

## 📊 Dataset

The system works with vendor-product interaction data containing:
- `vendor_id`: Unique vendor identifier
- `product_id`: Unique product identifier  
- `category`: Product category
- `price`: Product price
- `historical_sales`: Historical sales data
- `stock_level`: Current stock level
- `rate`: Product rating

## 🏗️ Architecture

### 1. Data Preprocessing
- Clean and normalize numeric features
- Encode categorical features
- Create vendor-product interaction matrix
- Handle missing values and outliers

### 2. Collaborative Filtering (CF)
- Uses Surprise library with SVD algorithm
- Models vendor-product interactions
- Predicts likelihood of successful uploads

### 3. Content-Based Filtering (CBF)
- Computes product similarity using cosine similarity
- Based on category, price, sales, stock level, and rating
- Recommends products similar to vendor's successful uploads

### 4. Hybrid Model
- Combines CF and CBF scores: `final_score = α * CF_score + (1-α) * CBF_score`
- Automatic α parameter tuning (default: 0.5, tested range: 0.3-0.7)
- Outputs ranked list of recommended products

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd vendor-recommendation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Train the model**:
```bash
python hybrid_recommendation_system.py
```

4. **Start the API server**:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📚 Usage

### Training the Model

```python
from hybrid_recommendation_system import HybridRecommendationSystem

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

# Evaluate the model
evaluation_results = hybrid_system.evaluate_model()

# Save the model
hybrid_system.save_model('hybrid_model.pkl')
```

### Using the API

#### Get Recommendations for a Vendor

```bash
curl "http://localhost:8000/recommend?vendor_id=12&top_k=10"
```

#### Get Recommendations with Category Filter

```bash
curl "http://localhost:8000/recommend?vendor_id=12&top_k=10&category_filter=Fashion"
```

#### Batch Recommendations

```bash
curl -X POST "http://localhost:8000/batch_recommend" \
     -H "Content-Type: application/json" \
     -d '{"vendor_ids": [12, 25, 30], "top_k": 5}'
```

#### Get Available Vendors

```bash
curl "http://localhost:8000/vendors"
```

#### Get Available Categories

```bash
curl "http://localhost:8000/categories"
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/recommend` | GET | Get product recommendations for a vendor |
| `/batch_recommend` | POST | Get recommendations for multiple vendors |
| `/vendors` | GET | Get list of available vendors |
| `/categories` | GET | Get list of available categories |
| `/vendor/{vendor_id}/stats` | GET | Get detailed vendor statistics |
| `/product/{product_id}/info` | GET | Get detailed product information |
| `/model/info` | GET | Get model information and statistics |

## 📈 Evaluation Metrics

The system evaluates performance using:

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **F1@K**: Harmonic mean of precision and recall
- **RMSE**: Root Mean Square Error for collaborative filtering

## 🎯 Example Output

```json
{
  "vendor_id": 12,
  "recommendations": [
    {
      "product_id": 45,
      "category": "Fashion",
      "price": 267236.42,
      "historical_sales": 217.0,
      "stock_level": 191,
      "rate": 4.9,
      "hybrid_score": 4.75,
      "cf_score": 4.6,
      "cbf_score": 4.9
    }
  ],
  "total_recommendations": 10,
  "category_filter": null
}
```

## 🔮 Future Enhancements

### Temporal Features
- Add seasonality patterns for product uploads
- Include time-series trends for better timing recommendations
- Consider day-of-week and month-of-year patterns

### Reinforcement Learning
- Implement real-time vendor feedback system
- Use multi-armed bandit algorithms for dynamic recommendations
- Add A/B testing framework for recommendation strategies

### Scalability Improvements
- Implement Apache Spark for big data processing
- Use PyTorch/TensorFlow for deep learning models
- Add distributed computing capabilities

### Advanced Features
- Include competitor analysis
- Add market demand forecasting
- Implement dynamic pricing recommendations
- Add inventory optimization suggestions

### Real-time Processing
- Implement streaming data processing
- Add real-time model updates
- Include live performance monitoring

## 📝 Model Files

- `hybrid_model.pkl`: Trained model with preprocessing pipeline
- `hybrid_recommendation_system.py`: Core ML implementation
- `main.py`: FastAPI web service
- `requirements.txt`: Python dependencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request




## 👥 Authors
Getachew Ekubay
Mulaw G/michael
Meron Kidane
Atsbeha Tesfay
Mikias Mengesha


- **ML Engineer** - *Initial work* - [Your GitHub(https://github.com/Gech-E)]

## 🙏 Acknowledgments

- Surprise library for collaborative filtering
- Scikit-learn for machine learning utilities
- FastAPI for the web framework
- Pandas and NumPy for data processing

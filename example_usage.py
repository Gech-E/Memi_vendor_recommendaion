"""
Example Usage of Hybrid Recommendation System
=============================================

This script demonstrates how to use the hybrid recommendation system
both programmatically and via the API.

Author: ML Engineer
Date: 2024
"""

import requests
import json
from hybrid_recommendation_system import HybridRecommendationSystem

def example_programmatic_usage():
    """
    Example of using the hybrid recommendation system programmatically
    """
    print("=" * 60)
    print("PROGRAMMATIC USAGE EXAMPLE")
    print("=" * 60)
    
    try:
        # Load the trained model
        print("Loading trained model...")
        hybrid_system = HybridRecommendationSystem()
        hybrid_system.load_model('hybrid_model.pkl')
        
        # Get some sample vendors
        sample_vendors = hybrid_system.df_processed['vendor_id'].unique()[:3]
        
        for vendor_id in sample_vendors:
            print(f"\n🏪 Recommendations for Vendor {vendor_id}:")
            print("-" * 40)
            
            # Get recommendations
            recommendations = hybrid_system.recommend_products(vendor_id, top_k=5)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. Product {rec['product_id']} ({rec['category']})")
                print(f"   💰 Price: ${rec['price']:,.2f}")
                print(f"   📈 Historical Sales: {rec['historical_sales']}")
                print(f"   📦 Stock Level: {rec['stock_level']}")
                print(f"   ⭐ Rating: {rec['rate']:.1f}")
                print(f"   🎯 Hybrid Score: {rec['hybrid_score']:.4f}")
                print(f"   🤝 CF Score: {rec['cf_score']:.4f}")
                print(f"   📝 CBF Score: {rec['cbf_score']:.4f}")
                print()
        
        # Example with category filter
        print("\n🔍 Recommendations with Category Filter (Fashion):")
        print("-" * 50)
        
        fashion_recs = hybrid_system.recommend_products(
            vendor_id=sample_vendors[0],
            top_k=3,
            category_filter='Fashion'
        )
        
        for i, rec in enumerate(fashion_recs, 1):
            print(f"{i}. Product {rec['product_id']} - {rec['category']}")
            print(f"   💰 Price: ${rec['price']:,.2f}")
            print(f"   🎯 Hybrid Score: {rec['hybrid_score']:.4f}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure to train the model first with: python train_model.py")

def example_api_usage():
    """
    Example of using the hybrid recommendation system via API
    """
    print("\n" + "=" * 60)
    print("API USAGE EXAMPLE")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    try:
        # Check if API is running
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not running!")
            print("   Start it with: python main.py")
            return
        
        print("✅ API server is running!")
        
        # Get model info
        print("\n📊 Model Information:")
        model_info = requests.get(f"{base_url}/model/info").json()
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Alpha: {model_info['alpha']}")
        print(f"   Total Vendors: {model_info['total_vendors']}")
        print(f"   Total Products: {model_info['total_products']}")
        
        # Get available vendors
        print("\n🏪 Available Vendors:")
        vendors = requests.get(f"{base_url}/vendors").json()
        sample_vendors = vendors[:3]
        
        for vendor in sample_vendors:
            print(f"   Vendor {vendor['vendor_id']}: {vendor['total_products']} products, "
                  f"avg rating {vendor['avg_rating']:.2f}")
        
        # Get recommendations for first vendor
        if sample_vendors:
            vendor_id = sample_vendors[0]['vendor_id']
            
            print(f"\n🎯 Recommendations for Vendor {vendor_id}:")
            rec_response = requests.get(f"{base_url}/recommend?vendor_id={vendor_id}&top_k=5")
            recommendations = rec_response.json()
            
            for i, rec in enumerate(recommendations['recommendations'], 1):
                print(f"{i}. Product {rec['product_id']} ({rec['category']})")
                print(f"   💰 Price: ${rec['price']:,.2f}")
                print(f"   📈 Historical Sales: {rec['historical_sales']}")
                print(f"   ⭐ Rating: {rec['rate']:.1f}")
                print(f"   🎯 Hybrid Score: {rec['hybrid_score']:.4f}")
                print()
        
        # Get available categories
        print("\n📂 Available Categories:")
        categories = requests.get(f"{base_url}/categories").json()
        
        for category in categories[:5]:  # Show first 5 categories
            print(f"   {category['category']}: {category['total_products']} products, "
                  f"avg price ${category['avg_price']:,.2f}")
        
        # Example with category filter
        if categories and sample_vendors:
            category_name = categories[0]['category']
            vendor_id = sample_vendors[0]['vendor_id']
            
            print(f"\n🔍 Recommendations filtered by '{category_name}':")
            filtered_rec = requests.get(
                f"{base_url}/recommend?vendor_id={vendor_id}&top_k=3&category_filter={category_name}"
            ).json()
            
            for i, rec in enumerate(filtered_rec['recommendations'], 1):
                print(f"{i}. Product {rec['product_id']} - {rec['category']}")
                print(f"   💰 Price: ${rec['price']:,.2f}")
                print(f"   🎯 Hybrid Score: {rec['hybrid_score']:.4f}")
                print()
        
        # Batch recommendations example
        print("\n📦 Batch Recommendations:")
        batch_data = {
            "vendor_ids": [vendor['vendor_id'] for vendor in sample_vendors[:2]],
            "top_k": 3
        }
        
        batch_response = requests.post(f"{base_url}/batch_recommend", json=batch_data)
        batch_recs = batch_response.json()
        
        for result in batch_recs['results']:
            print(f"   Vendor {result['vendor_id']}: {result['total_recommendations']} recommendations")
            if result['recommendations']:
                top_rec = result['recommendations'][0]
                print(f"     Top: Product {top_rec['product_id']} ({top_rec['category']}) - "
                      f"Score: {top_rec['hybrid_score']:.4f}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {str(e)}")
        print("   Make sure the API server is running: python main.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """
    Main function to run examples
    """
    print("🚀 HYBRID RECOMMENDATION SYSTEM - USAGE EXAMPLES")
    print("=" * 60)
    
    # Run programmatic example
    example_programmatic_usage()
    
    # Run API example
    example_api_usage()
    
    print("\n" + "=" * 60)
    print("📚 USAGE SUMMARY")
    print("=" * 60)
    print("""
    The Hybrid Recommendation System can be used in two ways:
    
    1. 📊 PROGRAMMATIC USAGE:
       - Load the trained model directly
       - Get recommendations programmatically
       - Full control over the recommendation process
    
    2. 🌐 API USAGE:
       - Use REST API endpoints
       - Easy integration with web applications
       - Scalable and production-ready
    
    🎯 KEY FEATURES:
    - Combines Collaborative Filtering and Content-Based Filtering
    - Automatic parameter tuning (alpha optimization)
    - Comprehensive evaluation metrics
    - Category filtering support
    - Batch processing capabilities
    
    🔧 NEXT STEPS:
    1. Train the model: python train_model.py
    2. Start API server: python main.py
    3. Test the system: python test_api.py
    4. Integrate with your application
    """)

if __name__ == "__main__":
    main()

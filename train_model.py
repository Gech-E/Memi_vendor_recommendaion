"""
Training Script for Hybrid Recommendation System
===============================================

This script trains the hybrid recommendation system and saves the model.
It can be run independently to train the model before starting the API server.

Usage:
    python train_model.py

Author: ML Engineer
Date: 2024
"""

import os
import sys
import time
from hybrid_recommendation_system import HybridRecommendationSystem

def main():
    """
    Main training function
    """
    print("=" * 80)
    print("HYBRID RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("=" * 80)
    
    # Check if data file exists
    data_file = 'reccommendation_datas.csv'
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file '{data_file}' not found!")
        print("Please ensure the CSV file is in the current directory.")
        sys.exit(1)
    
    try:
        # Initialize the system
        print("🚀 Initializing Hybrid Recommendation System...")
        hybrid_system = HybridRecommendationSystem(alpha=0.5)
        
        # Load and preprocess data
        print("\n📊 Loading and preprocessing data...")
        start_time = time.time()
        
        hybrid_system.load_data(data_file)
        hybrid_system.preprocess_data()
        
        load_time = time.time() - start_time
        print(f"✅ Data preprocessing completed in {load_time:.2f} seconds")
        
        # Train Collaborative Filtering model
        print("\n🤝 Training Collaborative Filtering model...")
        cf_start_time = time.time()
        
        hybrid_system.train_collaborative_filtering()
        
        cf_time = time.time() - cf_start_time
        print(f"✅ Collaborative Filtering training completed in {cf_time:.2f} seconds")
        
        # Train Content-Based Filtering model
        print("\n📝 Training Content-Based Filtering model...")
        cbf_start_time = time.time()
        
        hybrid_system.train_content_based_filtering()
        
        cbf_time = time.time() - cbf_start_time
        print(f"✅ Content-Based Filtering training completed in {cbf_time:.2f} seconds")
        
        # Tune alpha parameter
        print("\n🎯 Tuning alpha parameter...")
        tuning_start_time = time.time()
        
        tuning_results = hybrid_system.tune_alpha()
        
        tuning_time = time.time() - tuning_start_time
        print(f"✅ Alpha tuning completed in {tuning_time:.2f} seconds")
        print(f"🎯 Best alpha: {tuning_results['best_alpha']}")
        print(f"🎯 Best F1@10: {tuning_results['best_f1']:.4f}")
        
        # Evaluate the model
        print("\n📈 Evaluating model performance...")
        eval_start_time = time.time()
        
        evaluation_results = hybrid_system.evaluate_model()
        
        eval_time = time.time() - eval_start_time
        print(f"✅ Model evaluation completed in {eval_time:.2f} seconds")
        print(f"📈 Precision@10: {evaluation_results['precision@10']:.4f}")
        print(f"📈 Recall@10: {evaluation_results['recall@10']:.4f}")
        print(f"📈 F1@10: {evaluation_results['f1@10']:.4f}")
        
        # Save the model
        print("\n💾 Saving trained model...")
        save_start_time = time.time()
        
        model_file = 'hybrid_model.pkl'
        hybrid_system.save_model(model_file)
        
        save_time = time.time() - save_start_time
        print(f"✅ Model saved to '{model_file}' in {save_time:.2f} seconds")
        
        # Calculate total training time
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️  Total training time: {total_time:.2f} seconds")
        
        # Display model statistics
        print("\n📊 Model Statistics:")
        print(f"   • Total vendors: {len(hybrid_system.df_processed['vendor_id'].unique())}")
        print(f"   • Total products: {len(hybrid_system.df_processed['product_id'].unique())}")
        print(f"   • Total categories: {len(hybrid_system.df_processed['category'].unique())}")
        print(f"   • Total interactions: {len(hybrid_system.df_processed)}")
        print(f"   • Alpha parameter: {hybrid_system.alpha}")
        
        # Example recommendations
        print("\n🔍 Example Recommendations:")
        print("-" * 50)
        
        sample_vendors = hybrid_system.df_processed['vendor_id'].unique()[:3]
        
        for vendor_id in sample_vendors:
            print(f"\n🏪 Vendor {vendor_id}:")
            recommendations = hybrid_system.recommend_products(vendor_id, top_k=3)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. Product {rec['product_id']} ({rec['category']})")
                print(f"      💰 Price: ${rec['price']:,.2f}")
                print(f"      📈 Sales: {rec['historical_sales']}")
                print(f"      ⭐ Rating: {rec['rate']:.1f}")
                print(f"      🎯 Hybrid Score: {rec['hybrid_score']:.4f}")
        
        print("\n" + "=" * 80)
        print("🎯 NEXT STEPS:")
        print("=" * 80)
        print("1. Start the API server: python main.py")
        print("2. Access the API documentation: http://localhost:8000/docs")
        print("3. Test the recommendations endpoint")
        print("4. Integrate with your e-commerce platform")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure the CSV file exists and is properly formatted")
        print("2. Check that all required dependencies are installed")
        print("3. Verify the data file has the expected columns")
        print("4. Check the error message above for specific issues")
        sys.exit(1)

if __name__ == "__main__":
    main()

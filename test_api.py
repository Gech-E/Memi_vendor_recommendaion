"""
Test Script for Hybrid Recommendation System API
===============================================

This script tests the FastAPI endpoints to ensure they work correctly.
Run this after starting the API server to verify functionality.

Usage:
    python test_api.py

Author: ML Engineer
Date: 2024
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", data=None, expected_status=200):
    """
    Test a single API endpoint
    
    Args:
        endpoint (str): API endpoint to test
        method (str): HTTP method (GET, POST)
        data (dict): Request data for POST requests
        expected_status (int): Expected HTTP status code
    
    Returns:
        dict: Response data or None if error
    """
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"❌ Unsupported method: {method}")
            return None
        
        if response.status_code == expected_status:
            print(f"✅ {method} {endpoint} - Status: {response.status_code}")
            return response.json()
        else:
            print(f"❌ {method} {endpoint} - Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {method} {endpoint} - Error: {str(e)}")
        return None

def main():
    """
    Main test function
    """
    print("=" * 60)
    print("HYBRID RECOMMENDATION SYSTEM API - TESTING")
    print("=" * 60)
    
    # Check if API server is running
    print("🔍 Checking if API server is running...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running!")
        else:
            print("❌ API server is not responding correctly")
            return
    except requests.exceptions.RequestException:
        print("❌ API server is not running!")
        print("   Please start the server with: python main.py")
        return
    
    print("\n🧪 Running API tests...")
    
    # Test 1: Health check
    print("\n1. Testing health check endpoint...")
    health_data = test_endpoint("/")
    if health_data:
        print(f"   Status: {health_data.get('status')}")
        print(f"   Model loaded: {health_data.get('model_loaded')}")
    
    # Test 2: Get model info
    print("\n2. Testing model info endpoint...")
    model_info = test_endpoint("/model/info")
    if model_info:
        print(f"   Model type: {model_info.get('model_type')}")
        print(f"   Alpha: {model_info.get('alpha')}")
        print(f"   Total vendors: {model_info.get('total_vendors')}")
        print(f"   Total products: {model_info.get('total_products')}")
    
    # Test 3: Get vendors
    print("\n3. Testing vendors endpoint...")
    vendors_data = test_endpoint("/vendors")
    if vendors_data:
        print(f"   Found {len(vendors_data)} vendors")
        if len(vendors_data) > 0:
            sample_vendor = vendors_data[0]
            print(f"   Sample vendor: ID={sample_vendor.get('vendor_id')}, "
                  f"Products={sample_vendor.get('total_products')}")
    
    # Test 4: Get categories
    print("\n4. Testing categories endpoint...")
    categories_data = test_endpoint("/categories")
    if categories_data:
        print(f"   Found {len(categories_data)} categories")
        if len(categories_data) > 0:
            sample_category = categories_data[0]
            print(f"   Sample category: {sample_category.get('category')}, "
                  f"Products={sample_category.get('total_products')}")
    
    # Test 5: Get recommendations (if we have vendors)
    if vendors_data and len(vendors_data) > 0:
        test_vendor_id = vendors_data[0]['vendor_id']
        
        print(f"\n5. Testing recommendations endpoint for vendor {test_vendor_id}...")
        rec_data = test_endpoint(f"/recommend?vendor_id={test_vendor_id}&top_k=5")
        if rec_data:
            print(f"   Vendor ID: {rec_data.get('vendor_id')}")
            print(f"   Recommendations: {rec_data.get('total_recommendations')}")
            
            if rec_data.get('recommendations'):
                sample_rec = rec_data['recommendations'][0]
                print(f"   Sample recommendation:")
                print(f"     Product ID: {sample_rec.get('product_id')}")
                print(f"     Category: {sample_rec.get('category')}")
                print(f"     Price: ${sample_rec.get('price')}")
                print(f"     Hybrid Score: {sample_rec.get('hybrid_score')}")
        
        # Test 6: Batch recommendations
        print(f"\n6. Testing batch recommendations endpoint...")
        batch_data = {
            "vendor_ids": [test_vendor_id],
            "top_k": 3
        }
        batch_rec_data = test_endpoint("/batch_recommend", method="POST", data=batch_data)
        if batch_rec_data:
            print(f"   Processed vendors: {batch_rec_data.get('total_vendors')}")
            if batch_rec_data.get('results'):
                result = batch_rec_data['results'][0]
                print(f"   Vendor {result.get('vendor_id')}: {result.get('total_recommendations')} recommendations")
        
        # Test 7: Vendor stats
        print(f"\n7. Testing vendor stats endpoint...")
        vendor_stats = test_endpoint(f"/vendor/{test_vendor_id}/stats")
        if vendor_stats:
            print(f"   Vendor ID: {vendor_stats.get('vendor_id')}")
            print(f"   Total products: {vendor_stats.get('total_products')}")
            print(f"   Average rating: {vendor_stats.get('avg_rating')}")
            print(f"   Categories: {len(vendor_stats.get('categories', []))}")
    
    # Test 8: Test with category filter (if we have categories)
    if categories_data and len(categories_data) > 0 and vendors_data and len(vendors_data) > 0:
        test_category = categories_data[0]['category']
        test_vendor_id = vendors_data[0]['vendor_id']
        
        print(f"\n8. Testing recommendations with category filter '{test_category}'...")
        filtered_rec_data = test_endpoint(
            f"/recommend?vendor_id={test_vendor_id}&top_k=3&category_filter={test_category}"
        )
        if filtered_rec_data:
            print(f"   Category filter: {filtered_rec_data.get('category_filter')}")
            print(f"   Filtered recommendations: {filtered_rec_data.get('total_recommendations')}")
    
    # Test 9: Error handling
    print("\n9. Testing error handling...")
    
    # Test with non-existent vendor
    error_data = test_endpoint("/recommend?vendor_id=99999&top_k=5", expected_status=404)
    if error_data is None:
        print("   ✅ Correctly handled non-existent vendor")
    
    # Test with invalid parameters
    error_data = test_endpoint("/recommend?vendor_id=abc&top_k=5", expected_status=422)
    if error_data is None:
        print("   ✅ Correctly handled invalid parameters")
    
    print("\n" + "=" * 60)
    print("🎉 API TESTING COMPLETED!")
    print("=" * 60)
    
    print("\n📚 Available endpoints:")
    print("   • GET  /                    - Health check")
    print("   • GET  /recommend           - Get recommendations")
    print("   • POST /batch_recommend     - Batch recommendations")
    print("   • GET  /vendors             - List vendors")
    print("   • GET  /categories          - List categories")
    print("   • GET  /vendor/{id}/stats   - Vendor statistics")
    print("   • GET  /product/{id}/info   - Product information")
    print("   • GET  /model/info          - Model information")
    
    print("\n🌐 API Documentation:")
    print(f"   • Swagger UI: {BASE_URL}/docs")
    print(f"   • ReDoc: {BASE_URL}/redoc")

if __name__ == "__main__":
    main()

"""
FastAPI Application for Hybrid Recommendation System
===================================================

This FastAPI application provides REST API endpoints for the hybrid recommendation system
to recommend vendors the best products to upload to an e-commerce platform.

Endpoints:
- GET /: Health check
- GET /recommend: Get product recommendations for a vendor
- GET /vendors: Get list of available vendors
- GET /categories: Get list of available categories
- POST /batch_recommend: Get recommendations for multiple vendors

Author: ML Engineer
Date: 2024
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import pickle
import pandas as pd
import numpy as np
from hybrid_recommendation_system import HybridRecommendationSystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Recommendation System API",
    description="API for recommending products to vendors using hybrid collaborative and content-based filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded model
hybrid_system = None

# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    """Request model for single vendor recommendation"""
    vendor_id: int = Field(..., description="Vendor ID to get recommendations for")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    category_filter: Optional[str] = Field(default=None, description="Optional category filter")

class BatchRecommendationRequest(BaseModel):
    """Request model for batch recommendations"""
    vendor_ids: List[int] = Field(..., description="List of vendor IDs")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations per vendor")
    category_filter: Optional[str] = Field(default=None, description="Optional category filter")

class ProductRecommendation(BaseModel):
    """Model for individual product recommendation"""
    product_id: int
    category: str
    price: float
    historical_sales: float
    stock_level: float  # Changed to float to handle normalized values
    rate: float
    hybrid_score: float
    cf_score: float
    cbf_score: float

class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    vendor_id: int
    recommendations: List[ProductRecommendation]
    total_recommendations: int
    category_filter: Optional[str] = None

class BatchRecommendationResponse(BaseModel):
    """Response model for batch recommendations"""
    results: List[RecommendationResponse]
    total_vendors: int

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    model_loaded: bool

class VendorInfo(BaseModel):
    """Model for vendor information"""
    vendor_id: int
    total_products: int
    categories: List[str]
    avg_rating: float

class CategoryInfo(BaseModel):
    """Model for category information"""
    category: str
    total_products: int
    avg_price: float
    avg_rating: float

@app.on_event("startup")
async def startup_event():
    """Load the trained model on startup"""
    global hybrid_system
    try:
        logger.info("Loading hybrid recommendation model...")
        hybrid_system = HybridRecommendationSystem()
        hybrid_system.load_model('hybrid_model.pkl')
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        hybrid_system = None

@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy",
        message="Hybrid Recommendation System API is running",
        model_loaded=hybrid_system is not None
    )

@app.get("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    vendor_id: int = Query(..., description="Vendor ID to get recommendations for"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of recommendations"),
    category_filter: Optional[str] = Query(default=None, description="Optional category filter")
):
    """
    Get product recommendations for a specific vendor
    
    Args:
        vendor_id: ID of the vendor
        top_k: Number of recommendations to return (1-50)
        category_filter: Optional category to filter recommendations
    
    Returns:
        List of recommended products with scores
    """
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate vendor exists
        if vendor_id not in hybrid_system.df_processed['vendor_id'].values:
            raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
        
        # Get recommendations
        recommendations = hybrid_system.recommend_products(
            vendor_id=vendor_id,
            top_k=top_k,
            category_filter=category_filter
        )
        
        # Convert to response format
        product_recommendations = [
            ProductRecommendation(**rec) for rec in recommendations
        ]
        
        return RecommendationResponse(
            vendor_id=vendor_id,
            recommendations=product_recommendations,
            total_recommendations=len(product_recommendations),
            category_filter=category_filter
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations for vendor {vendor_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/batch_recommend", response_model=BatchRecommendationResponse)
async def get_batch_recommendations(request: BatchRecommendationRequest):
    """
    Get product recommendations for multiple vendors
    
    Args:
        request: BatchRecommendationRequest containing vendor IDs and parameters
    
    Returns:
        List of recommendations for each vendor
    """
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for vendor_id in request.vendor_ids:
            try:
                # Validate vendor exists
                if vendor_id not in hybrid_system.df_processed['vendor_id'].values:
                    logger.warning(f"Vendor {vendor_id} not found, skipping")
                    continue
                
                # Get recommendations
                recommendations = hybrid_system.recommend_products(
                    vendor_id=vendor_id,
                    top_k=request.top_k,
                    category_filter=request.category_filter
                )
                
                # Convert to response format
                product_recommendations = [
                    ProductRecommendation(**rec) for rec in recommendations
                ]
                
                results.append(RecommendationResponse(
                    vendor_id=vendor_id,
                    recommendations=product_recommendations,
                    total_recommendations=len(product_recommendations),
                    category_filter=request.category_filter
                ))
                
            except Exception as e:
                logger.error(f"Error processing vendor {vendor_id}: {str(e)}")
                continue
        
        return BatchRecommendationResponse(
            results=results,
            total_vendors=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error in batch recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/vendors", response_model=List[VendorInfo])
async def get_vendors():
    """
    Get list of all available vendors with basic statistics
    """
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        vendor_stats = hybrid_system.df_processed.groupby('vendor_id').agg({
            'product_id': 'count',
            'category': lambda x: list(x.unique()),
            'rate': 'mean'
        }).reset_index()
        
        vendor_stats.columns = ['vendor_id', 'total_products', 'categories', 'avg_rating']
        
        vendors = [
            VendorInfo(
                vendor_id=int(row['vendor_id']),
                total_products=int(row['total_products']),
                categories=row['categories'],
                avg_rating=float(row['avg_rating'])
            )
            for _, row in vendor_stats.iterrows()
        ]
        
        return vendors
        
    except Exception as e:
        logger.error(f"Error getting vendors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/categories", response_model=List[CategoryInfo])
async def get_categories():
    """
    Get list of all available categories with basic statistics
    """
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        category_stats = hybrid_system.df_processed.groupby('category').agg({
            'product_id': 'count',
            'price': 'mean',
            'rate': 'mean'
        }).reset_index()
        
        category_stats.columns = ['category', 'total_products', 'avg_price', 'avg_rating']
        
        categories = [
            CategoryInfo(
                category=row['category'],
                total_products=int(row['total_products']),
                avg_price=float(row['avg_price']),
                avg_rating=float(row['avg_rating'])
            )
            for _, row in category_stats.iterrows()
        ]
        
        return categories
        
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/vendor/{vendor_id}/stats")
async def get_vendor_stats(vendor_id: int):
    """
    Get detailed statistics for a specific vendor
    
    Args:
        vendor_id: ID of the vendor
    
    Returns:
        Detailed vendor statistics
    """
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate vendor exists
        if vendor_id not in hybrid_system.df_processed['vendor_id'].values:
            raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
        
        vendor_data = hybrid_system.df_processed[
            hybrid_system.df_processed['vendor_id'] == vendor_id
        ]
        
        stats = {
            'vendor_id': vendor_id,
            'total_products': len(vendor_data),
            'categories': vendor_data['category'].unique().tolist(),
            'avg_rating': float(vendor_data['rate'].mean()),
            'total_sales': float(vendor_data['historical_sales'].sum()),
            'avg_price': float(vendor_data['price'].mean()),
            'price_range': {
                'min': float(vendor_data['price'].min()),
                'max': float(vendor_data['price'].max())
            },
            'rating_distribution': vendor_data['rate'].value_counts().to_dict(),
            'category_distribution': vendor_data['category'].value_counts().to_dict()
        }
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting vendor stats for {vendor_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/product/{product_id}/info")
async def get_product_info(product_id: int):
    """
    Get detailed information for a specific product
    
    Args:
        product_id: ID of the product
    
    Returns:
        Detailed product information
    """
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate product exists
        if product_id not in hybrid_system.df_processed['product_id'].values:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        
        product_data = hybrid_system.df_processed[
            hybrid_system.df_processed['product_id'] == product_id
        ].iloc[0]
        
        info = {
            'product_id': product_id,
            'category': product_data['category'],
            'price': float(product_data['price']),
            'historical_sales': float(product_data['historical_sales']),
            'stock_level': int(product_data['stock_level']),
            'rate': float(product_data['rate']),
            'vendors_count': len(hybrid_system.df_processed[
                hybrid_system.df_processed['product_id'] == product_id
            ])
        }
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product info for {product_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model
    
    Returns:
        Model information and statistics
    """
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = {
            'model_type': 'Hybrid Recommendation System',
            'alpha': hybrid_system.alpha,
            'total_vendors': len(hybrid_system.df_processed['vendor_id'].unique()),
            'total_products': len(hybrid_system.df_processed['product_id'].unique()),
            'total_categories': len(hybrid_system.df_processed['category'].unique()),
            'total_interactions': len(hybrid_system.df_processed),
            'cf_model_loaded': hybrid_system.cf_model is not None,
            'cbf_matrix_shape': hybrid_system.product_similarity_matrix.shape if hybrid_system.product_similarity_matrix is not None else None,
            'data_shape': hybrid_system.df_processed.shape
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

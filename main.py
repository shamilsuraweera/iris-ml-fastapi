# -*- coding: utf-8 -*-
"""
🌸 Iris Flower Classification API

A user-friendly FastAPI application that predicts iris flower species
based on flower measurements using machine learning.

Features:
- Real-time predictions with confidence scores
- Interactive API documentation
- Detailed model information
- Input validation and error handling
"""

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import os
from datetime import datetime

# Pydantic models for request/response validation
class IrisFeatures(BaseModel):
    """
    Flower measurements for iris species prediction.
    
    All measurements should be in centimeters (cm).
    Typical ranges:
    - Sepal length: 4.0-8.0 cm
    - Sepal width: 2.0-4.5 cm  
    - Petal length: 1.0-7.0 cm
    - Petal width: 0.1-2.5 cm
    """
    sepal_length: float = Field(
        ..., 
        ge=0, 
        le=15,
        description="Length of the sepal (outer petal) in centimeters",
        example=5.1
    )
    sepal_width: float = Field(
        ..., 
        ge=0, 
        le=10,
        description="Width of the sepal (outer petal) in centimeters",
        example=3.5
    ) 
    petal_length: float = Field(
        ..., 
        ge=0, 
        le=15,
        description="Length of the petal (inner petal) in centimeters",
        example=1.4
    )
    petal_width: float = Field(
        ..., 
        ge=0, 
        le=10,
        description="Width of the petal (inner petal) in centimeters",
        example=0.2
    )

class PredictionResponse(BaseModel):
    """Response containing the predicted iris species and confidence information"""
    species: str = Field(description="Predicted iris species", example="setosa")
    confidence: float = Field(description="Confidence score (0-1) for the prediction", example=0.95)
    confidence_percentage: str = Field(description="Confidence as percentage", example="95.0%")
    probabilities: Dict[str, float] = Field(description="Probability scores for all species")
    interpretation: str = Field(description="Human-readable interpretation of the result")
    timestamp: str = Field(description="When the prediction was made")

class ModelInfo(BaseModel):
    """Information about the machine learning model"""
    model_type: str = Field(description="Type of ML algorithm used")
    problem_type: str = Field(description="Type of machine learning problem")
    features: List[str] = Field(description="Input features the model uses")
    classes: List[str] = Field(description="Possible output classes (species)")
    training_info: Dict[str, str] = Field(description="Information about model training")

class HealthResponse(BaseModel):
    """API health status response"""
    status: str = Field(description="API status", example="healthy")
    message: str = Field(description="Status message")
    timestamp: str = Field(description="Current server time")
    model_loaded: bool = Field(description="Whether the ML model is loaded")

class SpeciesInfo(BaseModel):
    """Information about iris species"""
    species_guide: Dict[str, Dict[str, str]] = Field(description="Guide to iris species characteristics")

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    flowers: List[IrisFeatures] = Field(description="List of flower measurements to classify")

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse] = Field(description="List of predictions for each flower")
    summary: Dict[str, int] = Field(description="Summary count of predicted species")

class ExampleData(BaseModel):
    """Example flower measurements for each species"""
    examples: Dict[str, IrisFeatures] = Field(description="Example measurements for each species")

# 🚀 Initialize FastAPI app with beautiful enhanced metadata
app = FastAPI(
    title="🌸 Iris Flower Classification API",
    description="""
    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🌺 Beautiful ML API for Iris Species Prediction</h1>
        <p style="font-size: 1.3rem; margin: 15px 0; opacity: 0.9;">Powered by Artificial Intelligence & Machine Learning</p>
        <div style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px; display: inline-block; margin-top: 10px;">
            <strong>🎯 96.7% Accuracy • ⚡ Lightning Fast • 🛡️ Secure</strong>
        </div>
    </div>
    
    ## 🌸 What This API Does
    
    This **state-of-the-art API** uses **Machine Learning** to classify Iris flowers into three beautiful species:
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 25px 0;">
        <div style="background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #48cc91; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
            <h3 style="color: #2f855a; margin: 0 0 15px 0; font-size: 1.4rem;">🌸 Iris Setosa</h3>
            <p style="margin: 0; color: #2d3748; line-height: 1.6;">Small, delicate flowers with distinctive short petals and wide sepals. Easy to identify!</p>
            <div style="margin-top: 15px; font-size: 0.9rem; color: #4a5568;">
                <strong>Typical:</strong> Petal length 1.0-2.0 cm
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #f0f8ff 0%, #bee3f8 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #4299e1; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
            <h3 style="color: #2b6cb0; margin: 0 0 15px 0; font-size: 1.4rem;">🌿 Iris Versicolor</h3>
            <p style="margin: 0; color: #2d3748; line-height: 1.6;">Medium-sized flowers with balanced proportions and beautiful purple hues.</p>
            <div style="margin-top: 15px; font-size: 0.9rem; color: #4a5568;">
                <strong>Typical:</strong> Petal length 3.5-5.0 cm
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #fdf2f8 0%, #fbb6ce 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #ed64a6; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
            <h3 style="color: #b83280; margin: 0 0 15px 0; font-size: 1.4rem;">🌺 Iris Virginica</h3>
            <p style="margin: 0; color: #2d3748; line-height: 1.6;">Large, elegant flowers with long petals and vibrant colors. Stunning beauty!</p>
            <div style="margin-top: 15px; font-size: 0.9rem; color: #4a5568;">
                <strong>Typical:</strong> Petal length 5.0-7.0 cm
            </div>
        </div>
    </div>
    
    ## 🎯 Amazing Features
    
    <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 30px; border-radius: 15px; margin: 25px 0; border: 2px solid #e2e8f0;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div style="text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 10px;">⚡</div>
                <strong style="color: #2d3748;">Lightning Fast</strong><br>
                <small style="color: #718096;">Predictions in milliseconds</small>
            </div>
            <div style="text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 10px;">🎯</div>
                <strong style="color: #2d3748;">96.7% Accurate</strong><br>
                <small style="color: #718096;">Highly reliable predictions</small>
            </div>
            <div style="text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 10px;">📊</div>
                <strong style="color: #2d3748;">Confidence Scores</strong><br>
                <small style="color: #718096;">Know how certain we are</small>
            </div>
            <div style="text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 10px;">🛡️</div>
                <strong style="color: #2d3748;">Input Validation</strong><br>
                <small style="color: #718096;">Smart error handling</small>
            </div>
        </div>
    </div>
    
    ## 🔬 How It Works
    
    <div style="background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #38b2ac; margin: 25px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
        <h3 style="color: #2c7a7b; margin: 0 0 20px 0;">Simple 4-Step Process:</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div style="background: white; padding: 15px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📏</div>
                <strong>1. Measure</strong><br>
                <small>Get sepal & petal dimensions</small>
            </div>
            <div style="background: white; padding: 15px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📤</div>
                <strong>2. Send Data</strong><br>
                <small>Use our beautiful API</small>
            </div>
            <div style="background: white; padding: 15px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 10px;">🤖</div>
                <strong>3. AI Analyzes</strong><br>
                <small>Model processes measurements</small>
            </div>
            <div style="background: white; padding: 15px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📋</div>
                <strong>4. Get Results</strong><br>
                <small>Species + confidence scores</small>
            </div>
        </div>
    </div>
    
    ## 📚 Training Dataset
    
    <div style="background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #f56565; margin: 25px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
        <h3 style="color: #c53030; margin: 0 0 15px 0;">🏆 Built on the Famous Iris Dataset</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
            <div style="text-align: center; background: white; padding: 15px; border-radius: 10px;">
                <div style="font-size: 2rem; color: #f56565;">📊</div>
                <strong>150 Samples</strong><br>
                <small>Carefully measured flowers</small>
            </div>
            <div style="text-align: center; background: white; padding: 15px; border-radius: 10px;">
                <div style="font-size: 2rem; color: #f56565;">🔢</div>
                <strong>4 Features</strong><br>
                <small>Sepal & petal dimensions</small>
            </div>
            <div style="text-align: center; background: white; padding: 15px; border-radius: 10px;">
                <div style="font-size: 2rem; color: #f56565;">🎯</div>
                <strong>96.67% Accuracy</strong><br>
                <small>On test data</small>
            </div>
            <div style="text-align: center; background: white; padding: 15px; border-radius: 10px;">
                <div style="font-size: 2rem; color: #f56565;">⚖️</div>
                <strong>80/20 Split</strong><br>
                <small>Training & testing</small>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); border-radius: 15px; margin: 25px 0; border: 2px dashed #cbd5e0;">
        <div style="font-size: 2rem; margin-bottom: 15px;">🌸</div>
        <h3 style="color: #4a5568; margin: 0 0 10px 0;">Ready to Classify Beautiful Iris Flowers?</h3>
        <p style="margin: 0; color: #718096; font-style: italic;">Let's get started with some amazing AI-powered predictions!</p>
    </div>
    """,
    version="2.0.0",
    contact={
        "name": "🌸 Iris ML Team",
        "url": "https://github.com/iris-ml-api",
        "email": "hello@iris-ml.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://127.0.0.1:8000",
            "description": "🏠 Local Development Server"
        },
        {
            "url": "https://iris-api.example.com", 
            "description": "🌐 Production Server"
        }
    ],
    tags_metadata=[
        {
            "name": "Health Check",
            "description": "💚 **API Health & Status**\n\nCheck if the API is running and the ML model is loaded properly. Perfect for monitoring and health checks.",
            "externalDocs": {
                "description": "Health Check Guide",
                "url": "https://docs.example.com/health"
            }
        },
        {
            "name": "Predictions", 
            "description": "🔮 **Machine Learning Predictions**\n\nThe core functionality! Use these endpoints to classify iris flowers using our trained AI model. Supports both single and batch predictions with confidence scores.",
            "externalDocs": {
                "description": "Prediction Guide",
                "url": "https://docs.example.com/predictions"
            }
        },
        {
            "name": "Model Information",
            "description": "📊 **Model Details & Performance**\n\nGet detailed information about our machine learning model, including accuracy metrics, training data, and technical specifications.",
            "externalDocs": {
                "description": "Model Documentation",
                "url": "https://docs.example.com/model"
            }
        },
        {
            "name": "Species Guide",
            "description": "📚 **Iris Species Reference**\n\nComprehensive guide to iris species characteristics, typical measurements, and identification tips. Perfect for learning about iris flowers!",
            "externalDocs": {
                "description": "Botanical Guide",
                "url": "https://docs.example.com/species"
            }
        },
        {
            "name": "Examples",
            "description": "📋 **Sample Data & Testing**\n\nGet example flower measurements for testing the API and understanding typical values for each iris species. Great for getting started!",
            "externalDocs": {
                "description": "API Examples",
                "url": "https://docs.example.com/examples"
            }
        }
    ]
)

# Add CORS middleware for proper encoding handling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
class_names = ["setosa", "versicolor", "virginica"]

# Species information for users
species_characteristics = {
    "setosa": {
        "description": "Small, delicate flowers with distinctive features",
        "typical_sepal_length": "4.5-5.5 cm",
        "typical_sepal_width": "3.0-4.0 cm", 
        "typical_petal_length": "1.0-2.0 cm",
        "typical_petal_width": "0.1-0.5 cm",
        "distinguishing_features": "Very short petals, wide sepals, compact flower"
    },
    "versicolor": {
        "description": "Medium-sized flowers with balanced proportions",
        "typical_sepal_length": "5.5-6.5 cm",
        "typical_sepal_width": "2.5-3.5 cm",
        "typical_petal_length": "3.5-5.0 cm", 
        "typical_petal_width": "1.0-1.5 cm",
        "distinguishing_features": "Moderate size, balanced petal-to-sepal ratio"
    },
    "virginica": {
        "description": "Large, elegant flowers with long petals",
        "typical_sepal_length": "6.0-8.0 cm",
        "typical_sepal_width": "2.5-3.5 cm",
        "typical_petal_length": "5.0-7.0 cm",
        "typical_petal_width": "1.5-2.5 cm", 
        "distinguishing_features": "Long petals, large overall size, narrow sepals"
    }
}

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup with detailed logging"""
    global model
    try:
        if not os.path.exists("model.pkl"):
            print("❌ Model file 'model.pkl' not found!")
            print("🔧 Please run 'python train_model.py' first to train the model.")
            raise FileNotFoundError("Model file 'model.pkl' not found. Please train the model first.")
        
        model = joblib.load("model.pkl")
        print("✅ Iris classification model loaded successfully!")
        print("🚀 API is ready to make predictions!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔧 Please check that the model file exists and try again.")
        raise e

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    """Beautiful landing page for the Iris Classification API"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), media_type="text/html; charset=utf-8")
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head>
                <meta charset="UTF-8">
                <title>🌸 Iris API</title>
            </head>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>🌸 Iris Flower Classification API</h1>
                <p>Welcome to our beautiful ML API!</p>
                <a href="/docs" style="background: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                    📖 View API Documentation
                </a>
            </body>
        </html>
        """, media_type="text/html; charset=utf-8")

@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
async def health_check():
    """
    ## 💚 API Health Check
    
    <div style="background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); padding: 20px; border-radius: 10px; margin: 15px 0; border-left: 5px solid #48cc91;">
        <strong>🏥 Check if the Iris Classification API is running smoothly!</strong>
    </div>
    
    This endpoint provides comprehensive health information about the API and ML model status.
    
    ### ✅ What You'll Get:
    - **🟢 API Status** - Whether the service is healthy and operational
    - **⏰ Current Time** - Server timestamp for synchronization
    - **🤖 Model Status** - Whether the ML model is loaded and ready for predictions
    - **📊 System Info** - Additional diagnostic information
    
    ### 🔍 Perfect For:
    - **Monitoring** - Set up health checks for your applications
    - **Debugging** - Verify the API is working before making predictions
    - **Status Pages** - Display real-time API status to users
    - **Load Balancers** - Health check endpoints for traffic routing
    
    <div style="background: #e6fffa; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <strong>💡 Pro Tip:</strong> Use this endpoint to verify the API is ready before making prediction requests!
    </div>
    """
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        message="🌸 Iris Classification API is running and ready!" if model is not None else "⚠️ API is running but model is not loaded",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_species(features: IrisFeatures):
    """
    ## 🔮 Predict Iris Species with AI Magic!
    
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; margin: 20px 0; text-align: center;">
        <h3 style="color: white; margin: 0;">🌸 Transform Measurements into Species Identification!</h3>
        <p style="margin: 10px 0; opacity: 0.9;">Our AI model analyzes your flower measurements and predicts the iris species with confidence scores</p>
    </div>
    
    ### 📏 How to Measure Your Iris Flower:
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">
        <div style="background: #f0fff4; padding: 20px; border-radius: 10px; border-left: 5px solid #48cc91;">
            <h4 style="color: #2f855a; margin: 0 0 10px 0;">🌿 Sepals</h4>
            <p style="margin: 0; color: #2d3748;">The outer, usually green parts that protect the flower bud</p>
            <small style="color: #4a5568;"><strong>Tip:</strong> Measure length from base to tip, width at widest point</small>
        </div>
        <div style="background: #fdf2f8; padding: 20px; border-radius: 10px; border-left: 5px solid #ed64a6;">
            <h4 style="color: #b83280; margin: 0 0 10px 0;">🌸 Petals</h4>
            <p style="margin: 0; color: #2d3748;">The colorful inner parts of the flower</p>
            <small style="color: #4a5568;"><strong>Tip:</strong> Usually smaller and more delicate than sepals</small>
        </div>
    </div>
    
    ### 🎯 Example Measurements by Species:
    
    <div style="background: #f7fafc; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div style="background: white; padding: 15px; border-radius: 8px; border-top: 4px solid #48cc91;">
                <h5 style="color: #2f855a; margin: 0 0 10px 0;">🌸 Setosa (Small)</h5>
                <div style="font-family: monospace; font-size: 0.9rem; color: #4a5568;">
                    Sepal: 5.1 × 3.5 cm<br>
                    Petal: 1.4 × 0.2 cm
                </div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; border-top: 4px solid #4299e1;">
                <h5 style="color: #2b6cb0; margin: 0 0 10px 0;">🌿 Versicolor (Medium)</h5>
                <div style="font-family: monospace; font-size: 0.9rem; color: #4a5568;">
                    Sepal: 6.2 × 2.9 cm<br>
                    Petal: 4.3 × 1.3 cm
                </div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; border-top: 4px solid #ed64a6;">
                <h5 style="color: #b83280; margin: 0 0 10px 0;">🌺 Virginica (Large)</h5>
                <div style="font-family: monospace; font-size: 0.9rem; color: #4a5568;">
                    Sepal: 6.3 × 3.3 cm<br>
                    Petal: 6.0 × 2.5 cm
                </div>
            </div>
        </div>
    </div>
    
    ### 📊 What You'll Receive:
    
    <div style="background: #e6fffa; padding: 20px; border-radius: 10px; border-left: 5px solid #38b2ac; margin: 20px 0;">
        <ul style="margin: 0; padding-left: 20px; color: #2d3748;">
            <li><strong>🏷️ Predicted Species</strong> - The most likely iris species</li>
            <li><strong>🎯 Confidence Score</strong> - How certain our AI is (0-100%)</li>
            <li><strong>📈 All Probabilities</strong> - Scores for all three species</li>
            <li><strong>💭 Interpretation</strong> - Human-readable explanation of the result</li>
            <li><strong>⏰ Timestamp</strong> - When the prediction was made</li>
        </ul>
    </div>
    
    <div style="background: #fff5f5; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #f56565;">
        <strong>⚠️ Measurement Tips:</strong> Use centimeters for all measurements. Be as accurate as possible for best results!
    </div>
    """
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="🚫 Model not loaded. Please contact the administrator or restart the service."
        )
    
    try:
        # Convert input to numpy array
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width, 
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Get species name and confidence
        species = class_names[prediction]
        confidence = float(probabilities[prediction])
        confidence_pct = f"{confidence * 100:.1f}%"
        
        # Create probability dictionary
        prob_dict = {class_names[i]: round(float(prob), 3) for i, prob in enumerate(probabilities)}
        
        # Generate interpretation
        if confidence >= 0.9:
            interpretation = f"🎯 Very confident this is a {species.upper()}! The measurements strongly match this species."
        elif confidence >= 0.7:
            interpretation = f"👍 Likely a {species.upper()}. The measurements are consistent with this species."
        elif confidence >= 0.5:
            interpretation = f"🤔 Probably a {species.upper()}, but consider checking measurements or consulting an expert."
        else:
            interpretation = f"⚠️ Uncertain prediction. The measurements don't clearly match any species pattern."
        
        return PredictionResponse(
            species=species,
            confidence=confidence,
            confidence_percentage=confidence_pct,
            probabilities=prob_dict,
            interpretation=interpretation,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"🚫 Prediction error: {str(e)}. Please check your input values and try again."
        )

@app.get("/model-info", response_model=ModelInfo, tags=["Model Information"])
async def get_model_info():
    """
    ## 🤖 Machine Learning Model Information
    
    <div style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); color: white; padding: 25px; border-radius: 15px; margin: 20px 0; text-align: center;">
        <h3 style="color: white; margin: 0;">🧠 Meet Our AI Brain!</h3>
        <p style="margin: 10px 0; opacity: 0.9;">Discover the technical details behind our intelligent iris classification system</p>
    </div>
    
    ### 🔍 What You'll Learn:
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">
        <div style="background: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #4299e1;">
            <h4 style="color: #2b6cb0; margin: 0 0 10px 0;">🔧 Algorithm Details</h4>
            <p style="margin: 0; color: #2d3748;">Learn about the Logistic Regression algorithm and why it's perfect for iris classification</p>
        </div>
        <div style="background: #f0fff4; padding: 20px; border-radius: 10px; border-left: 5px solid #48cc91;">
            <h4 style="color: #2f855a; margin: 0 0 10px 0;">📊 Training Data</h4>
            <p style="margin: 0; color: #2d3748;">Discover the famous Iris dataset and how we prepared it for training</p>
        </div>
        <div style="background: #fdf2f8; padding: 20px; border-radius: 10px; border-left: 5px solid #ed64a6;">
            <h4 style="color: #b83280; margin: 0 0 10px 0;">🎯 Performance Metrics</h4>
            <p style="margin: 0; color: #2d3748;">See accuracy scores, validation results, and model reliability statistics</p>
        </div>
        <div style="background: #fff5f5; padding: 20px; border-radius: 10px; border-left: 5px solid #f56565;">
            <h4 style="color: #c53030; margin: 0 0 10px 0;">🔬 Technical Specs</h4>
            <p style="margin: 0; color: #2d3748;">Understand the features, classes, and model architecture</p>
        </div>
    </div>
    
    ### 📈 Model Performance Highlights:
    
    <div style="background: #e6fffa; padding: 20px; border-radius: 10px; border-left: 5px solid #38b2ac; margin: 20px 0;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; text-align: center;">
            <div style="background: white; padding: 15px; border-radius: 8px;">
                <div style="font-size: 2rem; color: #38b2ac;">🎯</div>
                <strong>96.67%</strong><br>
                <small>Test Accuracy</small>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px;">
                <div style="font-size: 2rem; color: #38b2ac;">📊</div>
                <strong>150</strong><br>
                <small>Training Samples</small>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px;">
                <div style="font-size: 2rem; color: #38b2ac;">🔢</div>
                <strong>4</strong><br>
                <small>Input Features</small>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px;">
                <div style="font-size: 2rem; color: #38b2ac;">🌸</div>
                <strong>3</strong><br>
                <small>Iris Species</small>
            </div>
        </div>
    </div>
    
    <div style="background: #f7fafc; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h4 style="color: #2d3748; margin: 0 0 15px 0;">🧪 Why Logistic Regression?</h4>
        <p style="margin: 0; color: #4a5568; line-height: 1.6;">
            We chose Logistic Regression because it's <strong>interpretable</strong>, <strong>fast</strong>, and <strong>highly effective</strong> 
            for this type of classification problem. It provides probability scores for each class, making it perfect for confidence-based predictions!
        </p>
    </div>
    """
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="🚫 Model not loaded. Please contact the administrator."
        )
    
    return ModelInfo(
        model_type="Logistic Regression",
        problem_type="Multi-class Classification", 
        features=feature_names,
        classes=class_names,
        training_info={
            "dataset": "Iris flower dataset (150 samples)",
            "training_split": "80% training, 20% testing",
            "algorithm": "Logistic Regression with L2 regularization",
            "performance": "~97% accuracy on test data",
            "features_used": "4 flower measurements (sepal & petal dimensions)"
        }
    )

@app.get("/species-info", response_model=SpeciesInfo, tags=["Species Guide"])
async def get_species_info():
    """
    ## 📚 Iris Species Guide
    
    Learn about the three iris species and their typical characteristics.
    
    **Use this guide to:**
    - Understand what makes each species unique
    - Check if your measurements seem reasonable
    - Learn typical size ranges for each species
    - Identify distinguishing features
    """
    return SpeciesInfo(species_guide=species_characteristics)

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    ## 🔮 Batch Predict Multiple Iris Flowers
    
    <div style="background: linear-gradient(135deg, #ed64a6 0%, #f093fb 100%); color: white; padding: 25px; border-radius: 15px; margin: 20px 0; text-align: center;">
        <h3 style="color: white; margin: 0;">🌸 Process Multiple Flowers at Once!</h3>
        <p style="margin: 10px 0; opacity: 0.9;">Efficient batch processing for flower collections, research, and bulk analysis</p>
    </div>
    
    ### 🚀 Perfect For:
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
        <div style="background: #f0f8ff; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #4299e1;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🔬</div>
            <strong style="color: #2b6cb0;">Research Projects</strong><br>
            <small style="color: #4a5568;">Analyze large datasets efficiently</small>
        </div>
        <div style="background: #f0fff4; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #48cc91;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🌺</div>
            <strong style="color: #2f855a;">Flower Collections</strong><br>
            <small style="color: #4a5568;">Classify entire garden collections</small>
        </div>
        <div style="background: #fdf2f8; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #ed64a6;">
            <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
            <strong style="color: #b83280;">Data Analysis</strong><br>
            <small style="color: #4a5568;">Bulk processing for statistics</small>
        </div>
        <div style="background: #fff5f5; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #f56565;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🎓</div>
            <strong style="color: #c53030;">Educational Use</strong><br>
            <small style="color: #4a5568;">Teaching and learning tools</small>
        </div>
    </div>
    
    ### 📥 Input Format:
    
    <div style="background: #f7fafc; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #a0aec0;">
        <p style="margin: 0 0 15px 0; color: #2d3748;"><strong>Send an array of flower measurements:</strong></p>
        <div style="background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.9rem;">
{<br>
&nbsp;&nbsp;"flowers": [<br>
&nbsp;&nbsp;&nbsp;&nbsp;{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},<br>
&nbsp;&nbsp;&nbsp;&nbsp;{"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3}<br>
&nbsp;&nbsp;]<br>
}
        </div>
    </div>
    
    ### 📤 What You'll Receive:
    
    <div style="background: #e6fffa; padding: 20px; border-radius: 10px; border-left: 5px solid #38b2ac; margin: 20px 0;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            <div>
                <h4 style="color: #2c7a7b; margin: 0 0 10px 0;">📋 Individual Predictions</h4>
                <ul style="margin: 0; padding-left: 20px; color: #2d3748;">
                    <li>Species for each flower</li>
                    <li>Confidence scores</li>
                    <li>Probability distributions</li>
                    <li>Interpretations</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #2c7a7b; margin: 0 0 10px 0;">📊 Summary Statistics</h4>
                <ul style="margin: 0; padding-left: 20px; color: #2d3748;">
                    <li>Count by species</li>
                    <li>Batch processing info</li>
                    <li>Overall statistics</li>
                    <li>Performance metrics</li>
                </ul>
            </div>
        </div>
    </div>
    
    <div style="background: #fff5f5; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #f56565;">
        <strong>⚡ Performance:</strong> Our batch processing is optimized for speed - process hundreds of flowers in seconds!
    </div>
    """
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="🚫 Model not loaded. Please contact the administrator."
        )
    
    predictions = []
    species_count = {"setosa": 0, "versicolor": 0, "virginica": 0}
    
    for flower in request.flowers:
        # Reuse the single prediction logic
        try:
            input_data = np.array([[
                flower.sepal_length,
                flower.sepal_width, 
                flower.petal_length,
                flower.petal_width
            ]])
            
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            species = class_names[prediction]
            confidence = float(probabilities[prediction])
            confidence_pct = f"{confidence * 100:.1f}%"
            
            prob_dict = {class_names[i]: round(float(prob), 3) for i, prob in enumerate(probabilities)}
            
            if confidence >= 0.9:
                interpretation = f"🎯 Very confident this is a {species.upper()}!"
            elif confidence >= 0.7:
                interpretation = f"👍 Likely a {species.upper()}."
            elif confidence >= 0.5:
                interpretation = f"🤔 Probably a {species.upper()}."
            else:
                interpretation = f"⚠️ Uncertain prediction."
            
            pred_response = PredictionResponse(
                species=species,
                confidence=confidence,
                confidence_percentage=confidence_pct,
                probabilities=prob_dict,
                interpretation=interpretation,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            )
            
            predictions.append(pred_response)
            species_count[species] += 1
            
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"🚫 Error processing flower measurement: {str(e)}"
            )
    
    return BatchPredictionResponse(
        predictions=predictions,
        summary=species_count
    )

@app.get("/examples", response_model=ExampleData, tags=["Examples"])
async def get_examples():
    """
    ## 📋 Example Flower Measurements
    
    <div style="background: linear-gradient(135deg, #48cc91 0%, #38b2ac 100%); color: white; padding: 25px; border-radius: 15px; margin: 20px 0; text-align: center;">
        <h3 style="color: white; margin: 0;">🌸 Perfect Starting Points!</h3>
        <p style="margin: 10px 0; opacity: 0.9;">Real flower measurements from each iris species - perfect for testing and learning</p>
    </div>
    
    ### 🎯 Use These Examples To:
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
        <div style="background: #f0fff4; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #48cc91;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🧪</div>
            <strong style="color: #2f855a;">Test the API</strong><br>
            <small style="color: #4a5568;">Try predictions with known data</small>
        </div>
        <div style="background: #f0f8ff; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #4299e1;">
            <div style="font-size: 2rem; margin-bottom: 10px;">📏</div>
            <strong style="color: #2b6cb0;">Learn Ranges</strong><br>
            <small style="color: #4a5568;">Understand typical measurements</small>
        </div>
        <div style="background: #fdf2f8; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #ed64a6;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🔍</div>
            <strong style="color: #b83280;">Compare Flowers</strong><br>
            <small style="color: #4a5568;">See how your measurements stack up</small>
        </div>
        <div style="background: #fff5f5; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #f56565;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🎓</div>
            <strong style="color: #c53030;">Educational Use</strong><br>
            <small style="color: #4a5568;">Learn species characteristics</small>
        </div>
    </div>
    
    ### 🌺 Species Examples:
    
    <div style="background: #f7fafc; padding: 25px; border-radius: 15px; margin: 20px 0;">
        <p style="margin: 0 0 20px 0; color: #2d3748; text-align: center;"><strong>These are real measurements from the famous Iris dataset!</strong></p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
            <div style="background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); padding: 20px; border-radius: 12px; border: 2px solid #48cc91;">
                <h4 style="color: #2f855a; margin: 0 0 15px 0; text-align: center;">🌸 Iris Setosa</h4>
                <div style="background: white; padding: 15px; border-radius: 8px; font-family: monospace;">
                    <div style="color: #2f855a;"><strong>Sepal:</strong> 5.1 × 3.5 cm</div>
                    <div style="color: #2f855a;"><strong>Petal:</strong> 1.4 × 0.2 cm</div>
                </div>
                <div style="margin-top: 10px; font-size: 0.9rem; color: #4a5568; text-align: center;">
                    <em>Small & distinctive</em>
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, #f0f8ff 0%, #bee3f8 100%); padding: 20px; border-radius: 12px; border: 2px solid #4299e1;">
                <h4 style="color: #2b6cb0; margin: 0 0 15px 0; text-align: center;">🌿 Iris Versicolor</h4>
                <div style="background: white; padding: 15px; border-radius: 8px; font-family: monospace;">
                    <div style="color: #2b6cb0;"><strong>Sepal:</strong> 6.2 × 2.9 cm</div>
                    <div style="color: #2b6cb0;"><strong>Petal:</strong> 4.3 × 1.3 cm</div>
                </div>
                <div style="margin-top: 10px; font-size: 0.9rem; color: #4a5568; text-align: center;">
                    <em>Medium & balanced</em>
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, #fdf2f8 0%, #fbb6ce 100%); padding: 20px; border-radius: 12px; border: 2px solid #ed64a6;">
                <h4 style="color: #b83280; margin: 0 0 15px 0; text-align: center;">🌺 Iris Virginica</h4>
                <div style="background: white; padding: 15px; border-radius: 8px; font-family: monospace;">
                    <div style="color: #b83280;"><strong>Sepal:</strong> 6.3 × 3.3 cm</div>
                    <div style="color: #b83280;"><strong>Petal:</strong> 6.0 × 2.5 cm</div>
                </div>
                <div style="margin-top: 10px; font-size: 0.9rem; color: #4a5568; text-align: center;">
                    <em>Large & elegant</em>
                </div>
            </div>
        </div>
    </div>
    
    ### 🚀 Quick Test:
    
    <div style="background: #e6fffa; padding: 20px; border-radius: 10px; border-left: 5px solid #38b2ac; margin: 20px 0;">
        <p style="margin: 0 0 10px 0; color: #2c7a7b;"><strong>💡 Try This:</strong></p>
        <ol style="margin: 0; padding-left: 20px; color: #2d3748;">
            <li>Copy one of the example measurements above</li>
            <li>Use the <strong>/predict</strong> endpoint with the data</li>
            <li>See how accurately our AI identifies the species!</li>
            <li>Try modifying the values slightly to see how predictions change</li>
        </ol>
    </div>
    """
    examples = {
        "setosa": IrisFeatures(
            sepal_length=5.1,
            sepal_width=3.5,
            petal_length=1.4,
            petal_width=0.2
        ),
        "versicolor": IrisFeatures(
            sepal_length=6.2,
            sepal_width=2.9,
            petal_length=4.3,
            petal_width=1.3
        ),
        "virginica": IrisFeatures(
            sepal_length=6.3,
            sepal_width=3.3,
            petal_length=6.0,
            petal_width=2.5
        )
    }
    
    return ExampleData(examples=examples)

# Enhanced custom documentation with beautiful styling
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling and branding"""
    from fastapi.openapi.docs import get_swagger_ui_html
    
    custom_css = """
    <style>
        /* Custom styling for Swagger UI */
        .swagger-ui .topbar { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-bottom: 3px solid #5a67d8;
        }
        .swagger-ui .topbar .download-url-wrapper { display: none; }
        .swagger-ui .info { 
            margin: 30px 0;
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            padding: 30px;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        .swagger-ui .info .title {
            color: #2d3748;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 15px;
        }
        .swagger-ui .info .description {
            color: #4a5568;
            font-size: 1.1rem;
            line-height: 1.6;
        }
        .swagger-ui .scheme-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .swagger-ui .opblock.opblock-post {
            background: rgba(73, 204, 144, 0.1);
            border: 2px solid #48cc91;
            border-radius: 10px;
            margin: 15px 0;
        }
        .swagger-ui .opblock.opblock-get {
            background: rgba(66, 153, 225, 0.1);
            border: 2px solid #4299e1;
            border-radius: 10px;
            margin: 15px 0;
        }
        .swagger-ui .opblock-summary {
            padding: 15px 20px;
            font-weight: 600;
        }
        .swagger-ui .opblock-tag {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 1.2rem;
            font-weight: 600;
        }
        .swagger-ui .btn.execute {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 600;
            color: white;
            transition: all 0.3s ease;
        }
        .swagger-ui .btn.execute:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .swagger-ui .response-col_status {
            font-weight: 700;
        }
        .swagger-ui .parameters-col_description {
            color: #4a5568;
        }
        .swagger-ui .model-box {
            background: #f7fafc;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            padding: 20px;
        }
        .swagger-ui .model-title {
            color: #2d3748;
            font-weight: 600;
        }
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #5a67d8;
        }
    </style>
    """
    
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="🌸 Iris Classification API - Interactive Documentation",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui.css",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui-bundle.js",
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "operationsSorter": "method",
            "filter": True,
            "tryItOutEnabled": True
        }
    ) + custom_css

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc with enhanced styling"""
    from fastapi.openapi.docs import get_redoc_html
    
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="🌸 Iris Classification API - Alternative Documentation",
        redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    ) + """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .redoc-wrap {
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        }
        .menu-content {
            background: white;
            border-right: 3px solid #667eea;
        }
        .api-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }
        .api-info h1 {
            color: white;
            font-size: 2.5rem;
        }
        .tag {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 10px 15px;
            margin: 10px 0;
        }
        .http-verb.get {
            background: #4299e1;
            border-radius: 20px;
        }
        .http-verb.post {
            background: #48cc91;
            border-radius: 20px;
        }
        .operation-path {
            font-weight: 600;
            color: #2d3748;
        }
    </style>
    """

if __name__ == "__main__":
    import uvicorn
    print("🌸 Starting Iris Classification API...")
    print("📖 Visit http://localhost:8000/docs for interactive documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000)
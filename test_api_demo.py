#!/usr/bin/env python3
"""
🌸 Iris Classification API Demo Script

This script demonstrates how to use the Iris Classification API
with beautiful output and comprehensive testing.
"""

import requests
import json
from datetime import datetime
import time

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def print_header(title):
    """Print a beautiful header"""
    print("\n" + "="*60)
    print(f"🌸 {title}")
    print("="*60)

def print_result(data, title="Result"):
    """Print JSON data in a beautiful format"""
    print(f"\n📊 {title}:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

def test_health_check():
    """Test the health check endpoint"""
    print_header("Health Check Test")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is healthy!")
            print_result(response.json())
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection error: {e}")

def test_model_info():
    """Test the model info endpoint"""
    print_header("Model Information Test")
    
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            print("✅ Model info retrieved!")
            print_result(response.json())
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_species_info():
    """Test the species info endpoint"""
    print_header("Species Information Test")
    
    try:
        response = requests.get(f"{BASE_URL}/species-info")
        if response.status_code == 200:
            print("✅ Species info retrieved!")
            data = response.json()
            for species, info in data["species_guide"].items():
                print(f"\n🌺 {species.upper()}:")
                print(f"   Description: {info['description']}")
                print(f"   Features: {info['distinguishing_features']}")
        else:
            print(f"❌ Species info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_examples():
    """Test the examples endpoint"""
    print_header("Example Data Test")
    
    try:
        response = requests.get(f"{BASE_URL}/examples")
        if response.status_code == 200:
            print("✅ Examples retrieved!")
            data = response.json()
            for species, measurements in data["examples"].items():
                print(f"\n🌸 {species.upper()} example:")
                print(f"   Sepal: {measurements['sepal_length']}cm × {measurements['sepal_width']}cm")
                print(f"   Petal: {measurements['petal_length']}cm × {measurements['petal_width']}cm")
        else:
            print(f"❌ Examples failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_single_prediction():
    """Test single flower prediction"""
    print_header("Single Prediction Test")
    
    # Test with a typical Setosa example
    test_flower = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    print(f"🌸 Testing with flower measurements:")
    print(f"   Sepal: {test_flower['sepal_length']}cm × {test_flower['sepal_width']}cm")
    print(f"   Petal: {test_flower['petal_length']}cm × {test_flower['petal_width']}cm")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_flower,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prediction successful!")
            print(f"🏷️  Species: {result['species'].upper()}")
            print(f"🎯 Confidence: {result['confidence_percentage']}")
            print(f"💭 Interpretation: {result['interpretation']}")
            print(f"\n📊 All probabilities:")
            for species, prob in result['probabilities'].items():
                print(f"   {species}: {prob*100:.1f}%")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {e}")

def test_batch_prediction():
    """Test batch prediction with multiple flowers"""
    print_header("Batch Prediction Test")
    
    # Test with examples of all three species
    test_flowers = {
        "flowers": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.2,
                "sepal_width": 2.9,
                "petal_length": 4.3,
                "petal_width": 1.3
            },
            {
                "sepal_length": 6.3,
                "sepal_width": 3.3,
                "petal_length": 6.0,
                "petal_width": 2.5
            }
        ]
    }
    
    print(f"🌸 Testing batch prediction with {len(test_flowers['flowers'])} flowers...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict-batch",
            json=test_flowers,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Batch prediction successful!")
            
            print(f"\n📊 Summary:")
            for species, count in result['summary'].items():
                print(f"   {species}: {count} flowers")
            
            print(f"\n🔍 Individual predictions:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"   Flower {i}: {pred['species'].upper()} ({pred['confidence_percentage']})")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all API tests"""
    print("🌸" * 20)
    print("🌸 IRIS CLASSIFICATION API DEMO")
    print("🌸" * 20)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Testing API at: {BASE_URL}")
    
    # Run all tests
    test_health_check()
    time.sleep(0.5)
    
    test_model_info()
    time.sleep(0.5)
    
    test_species_info()
    time.sleep(0.5)
    
    test_examples()
    time.sleep(0.5)
    
    test_single_prediction()
    time.sleep(0.5)
    
    test_batch_prediction()
    
    print_header("Demo Complete")
    print("🎉 All tests completed!")
    print("📖 Visit http://127.0.0.1:8001/docs for interactive documentation")
    print("🏠 Visit http://127.0.0.1:8001/ for the beautiful web interface")

if __name__ == "__main__":
    main()
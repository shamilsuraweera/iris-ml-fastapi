#!/usr/bin/env python3
"""
🧪 Iris Classification API Test Script

This script tests all the API endpoints to make sure everything is working correctly.
Run this after starting your API server to verify functionality.
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

# Test data for each iris species
TEST_CASES = [
    {
        "name": "Setosa (Small flower)",
        "data": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "expected_species": "setosa"
    },
    {
        "name": "Versicolor (Medium flower)", 
        "data": {
            "sepal_length": 6.2,
            "sepal_width": 2.9,
            "petal_length": 4.3,
            "petal_width": 1.3
        },
        "expected_species": "versicolor"
    },
    {
        "name": "Virginica (Large flower)",
        "data": {
            "sepal_length": 6.3,
            "sepal_width": 3.3,
            "petal_length": 6.0,
            "petal_width": 2.5
        },
        "expected_species": "virginica"
    }
]

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🧪 {title}")
    print("="*60)

def print_test(test_name: str):
    """Print a test header"""
    print(f"\n🔍 Testing: {test_name}")
    print("-" * 40)

def test_endpoint(method: str, endpoint: str, data: Dict[Any, Any] = None) -> bool:
    """Test an API endpoint and return success status"""
    try:
        url = f"{BASE_URL}{endpoint}"
        
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        if response.status_code == 200:
            print(f"✅ {method} {endpoint} - Success (200)")
            return True
        else:
            print(f"❌ {method} {endpoint} - Failed ({response.status_code})")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ {method} {endpoint} - Connection failed")
        print("   Make sure the API server is running!")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ {method} {endpoint} - Request timed out")
        return False
    except Exception as e:
        print(f"❌ {method} {endpoint} - Error: {e}")
        return False

def test_prediction_accuracy(test_case: Dict[str, Any]) -> bool:
    """Test a prediction and check if it matches expected result"""
    try:
        url = f"{BASE_URL}/predict"
        response = requests.post(url, json=test_case["data"], timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Prediction failed with status {response.status_code}")
            return False
        
        result = response.json()
        predicted_species = result.get("species")
        confidence = result.get("confidence", 0)
        
        if predicted_species == test_case["expected_species"]:
            print(f"✅ Correct prediction: {predicted_species} (confidence: {confidence:.1%})")
            print(f"   Interpretation: {result.get('interpretation', 'N/A')}")
            return True
        else:
            print(f"❌ Wrong prediction: got {predicted_species}, expected {test_case['expected_species']}")
            print(f"   Confidence: {confidence:.1%}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
        return False

def display_response_sample(endpoint: str, method: str = "GET", data: Dict[Any, Any] = None):
    """Display a sample response from an endpoint"""
    try:
        url = f"{BASE_URL}{endpoint}"
        
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("📄 Sample Response:")
            print(json.dumps(result, indent=2)[:500] + "..." if len(str(result)) > 500 else json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"❌ Could not get sample response: {e}")

def main():
    """Run all API tests"""
    print_header("IRIS CLASSIFICATION API TESTS")
    print("This script will test all API endpoints to ensure they're working correctly.")
    print("Make sure your API server is running at http://localhost:8000")
    
    # Wait a moment for user to read
    time.sleep(2)
    
    total_tests = 0
    passed_tests = 0
    
    # Test 1: Health Check
    print_test("Health Check Endpoint")
    total_tests += 1
    if test_endpoint("GET", "/"):
        passed_tests += 1
        display_response_sample("/")
    
    # Test 2: Model Info
    print_test("Model Information Endpoint")
    total_tests += 1
    if test_endpoint("GET", "/model-info"):
        passed_tests += 1
        display_response_sample("/model-info")
    
    # Test 3: Species Info
    print_test("Species Information Endpoint")
    total_tests += 1
    if test_endpoint("GET", "/species-info"):
        passed_tests += 1
    
    # Test 4: Prediction Accuracy
    print_test("Prediction Accuracy Tests")
    for test_case in TEST_CASES:
        print(f"\n🌸 Testing {test_case['name']}:")
        print(f"   Input: {test_case['data']}")
        total_tests += 1
        if test_prediction_accuracy(test_case):
            passed_tests += 1
    
    # Test 5: Invalid Input Handling
    print_test("Invalid Input Handling")
    invalid_data = {
        "sepal_length": -1,  # Invalid negative value
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    total_tests += 1
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_data, timeout=10)
        if response.status_code == 422:  # Validation error expected
            print("✅ Invalid input correctly rejected (422)")
            passed_tests += 1
        else:
            print(f"❌ Expected validation error, got {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing invalid input: {e}")
    
    # Final Results
    print_header("TEST RESULTS")
    print(f"📊 Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your API is working perfectly!")
        print("\n🚀 Ready to use:")
        print("   • Visit http://localhost:8000/docs for interactive documentation")
        print("   • Use the /predict endpoint to classify iris flowers")
        print("   • Check /species-info for identification tips")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("   • Make sure the API server is running")
        print("   • Verify the model was trained successfully")
        print("   • Check for any error messages in the server logs")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        exit(1)
#!/usr/bin/env python3
"""
🌸 Iris Classification API Setup Script

This script helps you set up the Iris Classification API quickly and easily.
It will guide you through the installation and setup process step by step.
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🌸 {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def run_command(command, description):
    """Run a command and handle errors gracefully"""
    try:
        print(f"⚙️  Running: {description}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor} is compatible")
    return True

def main():
    """Main setup function"""
    print_header("IRIS CLASSIFICATION API SETUP")
    print("Welcome! This script will help you set up the Iris Classification API.")
    print("Please follow the steps below to get everything running.")
    
    # Check Python version
    print_step(1, "Checking Python Version")
    if not check_python_version():
        print("\n🔧 Please upgrade Python and try again.")
        return False
    
    # Check if virtual environment exists
    print_step(2, "Setting Up Virtual Environment")
    if not os.path.exists("venv"):
        if not run_command("python3 -m venv venv", "Creating virtual environment"):
            print("\n🔧 Please install python3-venv and try again:")
            print("   sudo apt install python3-venv  # On Ubuntu/Debian")
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Install dependencies
    print_step(3, "Installing Dependencies")
    activate_cmd = "source venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate"
    install_cmd = f"{activate_cmd} && pip install -r requirements.txt"
    
    if not run_command(install_cmd, "Installing Python packages"):
        print("\n🔧 Please check your internet connection and try again.")
        return False
    
    # Train the model
    print_step(4, "Training the Machine Learning Model")
    train_cmd = f"{activate_cmd} && python train_model.py"
    
    if not run_command(train_cmd, "Training the iris classification model"):
        print("\n🔧 Please check the error messages above and try again.")
        return False
    
    # Success message
    print_header("SETUP COMPLETE! 🎉")
    print("Your Iris Classification API is ready to use!")
    
    print("\n🚀 To start the API server:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print("   source venv/bin/activate")
    print("   uvicorn main:app --reload")
    
    print("\n🌐 Then open your browser to:")
    print("   http://localhost:8000/docs")
    
    print("\n💡 Quick test commands:")
    print("   # Health check")
    print("   curl http://localhost:8000/")
    print()
    print("   # Make a prediction")
    print('   curl -X POST "http://localhost:8000/predict" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}\'')
    
    print("\n📚 For more information, check the README.md file!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error and try again, or set up manually using the README.md instructions.")
        sys.exit(1)
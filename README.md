# 🌸 Iris Flower Classification API

> **A beginner-friendly machine learning API that identifies iris flower species from measurements**

Transform your flower measurements into species predictions with this easy-to-use AI-powered API! Perfect for botanists, students, gardeners, or anyone curious about iris flowers.

## 🌺 What Does This Do?

This API uses artificial intelligence to identify three types of iris flowers:

- **🌸 Setosa** - Small, delicate flowers with short petals
- **🌼 Versicolor** - Medium-sized flowers with balanced proportions  
- **🌺 Virginica** - Large, elegant flowers with long petals

Simply measure your iris flower's sepals and petals, send the data to the API, and get an instant species identification with confidence scores!

## 🚀 Quick Start Guide

### Step 1: Set Up Your Environment
```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Train the AI Model
```bash
python train_model.py
```
You'll see a detailed training process with accuracy reports and helpful explanations!

### Step 3: Start the API Server
```bash
uvicorn main:app --reload
```

### Step 4: Open Your Browser
Visit **http://localhost:8000/docs** for the interactive API playground!

## 📖 How to Use the API

### 🏥 Health Check
**GET** `/` - Check if the API is running
```bash
curl http://localhost:8000/
```

### 🔮 Make Predictions  
**POST** `/predict` - Identify your iris species

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5, 
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

**Example Response:**
```json
{
  "species": "setosa",
  "confidence": 0.95,
  "confidence_percentage": "95.0%",
  "interpretation": "🎯 Very confident this is a SETOSA! The measurements strongly match this species.",
  "probabilities": {
    "setosa": 0.95,
    "versicolor": 0.03,
    "virginica": 0.02
  },
  "timestamp": "2024-01-15 14:30:22 UTC"
}
```

### 🤖 Model Information
**GET** `/model-info` - Learn about the AI model
```bash
curl http://localhost:8000/model-info
```

### 📚 Species Guide
**GET** `/species-info` - Get detailed species characteristics
```bash
curl http://localhost:8000/species-info
```

## 🌸 How to Measure Your Iris

### What to Measure:
1. **Sepal Length & Width** - The outer green parts that protect the bud
2. **Petal Length & Width** - The colorful inner parts of the flower

### Measurement Tips:
- Use centimeters (cm) for all measurements
- Measure length from base to tip
- Measure width at the widest point
- Be as accurate as possible for best results

### Typical Ranges:
| Species | Sepal Length | Sepal Width | Petal Length | Petal Width |
|---------|--------------|-------------|--------------|-------------|
| **Setosa** | 4.5-5.5 cm | 3.0-4.0 cm | 1.0-2.0 cm | 0.1-0.5 cm |
| **Versicolor** | 5.5-6.5 cm | 2.5-3.5 cm | 3.5-5.0 cm | 1.0-1.5 cm |
| **Virginica** | 6.0-8.0 cm | 2.5-3.5 cm | 5.0-7.0 cm | 1.5-2.5 cm |

## 🎯 Try These Example Measurements

### Setosa Example:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Versicolor Example:
```json
{
  "sepal_length": 6.2,
  "sepal_width": 2.9,
  "petal_length": 4.3,
  "petal_width": 1.3
}
```

### Virginica Example:
```json
{
  "sepal_length": 6.3,
  "sepal_width": 3.3,
  "petal_length": 6.0,
  "petal_width": 2.5
}
```

## 🛠️ Technical Details

### Machine Learning Model:
- **Algorithm**: Logistic Regression
- **Training Data**: 150 iris flower samples
- **Accuracy**: ~97% on test data
- **Features**: 4 measurements per flower
- **Classes**: 3 iris species

### API Features:
- ✅ Input validation with helpful error messages
- ✅ Confidence scores and probability distributions
- ✅ Human-readable interpretations
- ✅ Interactive documentation
- ✅ Comprehensive error handling
- ✅ Species identification guide

### Project Structure:
```
iris-classification/
├── main.py              # FastAPI application
├── train_model.py       # Model training script
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── model.pkl           # Trained model (created after training)
```

## 🔧 Troubleshooting

### Model Not Found Error?
Run the training script first:
```bash
python train_model.py
```

### Import Errors?
Make sure you've installed dependencies:
```bash
pip install -r requirements.txt
```

### API Not Starting?
Check if port 8000 is available or use a different port:
```bash
uvicorn main:app --reload --port 8001
```

## 🎓 Educational Use

This project is perfect for:
- **Learning machine learning basics**
- **Understanding REST APIs**
- **Exploring data science workflows**
- **Teaching classification concepts**
- **Botanical education**

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to open an issue or submit a pull request!

---

**Happy flower classifying! 🌸🤖**# iris-ml-fastapi

"""
🌸 Iris Flower Classification Model Training Script

This script trains a machine learning model to classify iris flowers into three species:
- Setosa (usually has smaller petals)
- Versicolor (medium-sized flowers) 
- Virginica (largest flowers)

The model uses four flower measurements to make predictions:
- Sepal Length & Width (the outer petals)
- Petal Length & Width (the inner petals)
"""

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🌸 {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def train_iris_model():
    """
    Train an Iris flower classification model with detailed progress reporting
    """
    
    print_header("IRIS FLOWER CLASSIFICATION MODEL TRAINING")
    print("Welcome! This script will train a machine learning model to identify iris flowers.")
    print("The model learns from 150 flower samples with known species labels.")
    
    # Step 1: Load the dataset
    print_step(1, "Loading the Iris Dataset")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   📊 Total samples: {len(X)}")
    print(f"   🔢 Features per sample: {len(iris.feature_names)}")
    print(f"   🏷️  Species to classify: {len(iris.target_names)}")
    print(f"\n   📝 Feature names: {', '.join(iris.feature_names)}")
    print(f"   🌺 Species names: {', '.join(iris.target_names)}")
    
    # Show dataset distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n   📈 Dataset distribution:")
    for i, (species, count) in enumerate(zip(iris.target_names, counts)):
        print(f"      • {species.capitalize()}: {count} samples")
    
    # Step 2: Split the data
    print_step(2, "Splitting Data for Training and Testing")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Data split completed!")
    print(f"   🎯 Training samples: {len(X_train)} (80%)")
    print(f"   🧪 Testing samples: {len(X_test)} (20%)")
    print(f"   ⚖️  Split maintains equal representation of each species")
    
    # Step 3: Train the model
    print_step(3, "Training the Logistic Regression Model")
    print("🤖 Training in progress... The model is learning patterns from flower measurements.")
    
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)
    
    print("✅ Model training completed!")
    print("   🧠 Algorithm: Logistic Regression")
    print("   ⚙️  The model learned to distinguish between species using mathematical patterns")
    
    # Step 4: Evaluate the model
    print_step(4, "Evaluating Model Performance")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Accuracy: {accuracy:.1%}")
    if accuracy >= 0.95:
        print("   🌟 Excellent! The model performs very well.")
    elif accuracy >= 0.90:
        print("   👍 Good performance! The model is reliable.")
    else:
        print("   ⚠️  Moderate performance. Consider more training data.")
    
    print(f"\n📊 Detailed Performance Report:")
    print("   (Precision = accuracy for each species)")
    print("   (Recall = how many of each species were found)")
    print("   (F1-score = balanced measure of precision and recall)")
    print("\n" + classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Show confusion matrix in a user-friendly way
    cm = confusion_matrix(y_test, y_pred)
    print("🔍 Prediction Results Breakdown:")
    print("   (Rows = Actual species, Columns = Predicted species)")
    for i, actual_species in enumerate(iris.target_names):
        for j, predicted_species in enumerate(iris.target_names):
            count = cm[i][j]
            if i == j:  # Correct predictions
                if count > 0:
                    print(f"   ✅ {actual_species.capitalize()}: {count} correctly identified")
            else:  # Incorrect predictions
                if count > 0:
                    print(f"   ❌ {actual_species.capitalize()}: {count} misclassified as {predicted_species}")
    
    # Step 5: Save the model
    print_step(5, "Saving the Trained Model")
    joblib.dump(model, 'model.pkl')
    print("✅ Model saved successfully as 'model.pkl'")
    print("   💾 The trained model is now ready to make predictions on new flower data!")
    
    # Final summary
    print_header("TRAINING COMPLETE! 🎉")
    print("Your iris classification model is ready to use!")
    print("\n🚀 Next steps:")
    print("   1. Run the FastAPI server: uvicorn main:app --reload")
    print("   2. Open your browser to: http://localhost:8000/docs")
    print("   3. Try predicting iris species with flower measurements!")
    print("\n💡 Example measurements to try:")
    print("   • Setosa: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
    print("   • Versicolor: sepal_length=6.2, sepal_width=2.9, petal_length=4.3, petal_width=1.3")
    print("   • Virginica: sepal_length=6.3, sepal_width=3.3, petal_length=6.0, petal_width=2.5")
    
    return model

if __name__ == "__main__":
    try:
        train_iris_model()
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Please check your Python environment and try again.")
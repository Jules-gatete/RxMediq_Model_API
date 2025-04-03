from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import uvicorn
import io
import base64
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Drug Prescription Prediction API",
    description="API for predicting drug prescriptions based on patient data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data model for prediction
class PatientData(BaseModel):
    disease: str
    age: int
    gender: str
    severity: str

# Load pre-trained model and preprocessors
try:
    model = load_model('drug_prescription_model.keras')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('label_enc.pkl')
except Exception as e:
    raise Exception(f"Error loading model or preprocessors: {str(e)}")

# [Other unchanged functions like preprocess_input, predict_drug, preprocess_data, build_and_train_model, calculate_metrics remain the same]

# Function to generate visualizations dynamically
def generate_visualizations(history, y_test, y_pred, target_encoder):
    """Generate training history plot, confusion matrix, and class distribution dynamically in memory"""
    visualizations = {}

    # 1. Training History Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    visualizations['training_history'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    buf.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    visualizations['confusion_matrix'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    buf.close()

    # 3. Class Distribution of Predictions
    predicted_labels = target_encoder.inverse_transform(y_pred)
    plt.figure(figsize=(10, 6))
    sns.countplot(x=predicted_labels, palette='viridis')
    plt.title('Predicted Class Distribution')
    plt.xlabel('Drug')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    visualizations['class_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    buf.close()

    return visualizations

# POST endpoint for retraining the model
@app.post("/retrain/")
async def retrain_model(file: UploadFile = File(...)):
    """Retrain the model with a new dataset and provide metrics and visualizations"""
    global model, scaler, label_encoders, target_encoder
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        required_columns = {"disease", "age", "gender", "severity", "drug"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        
        # Preprocess data
        X, y, new_label_encoders, new_scaler, new_target_encoder = preprocess_data(df)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
        unique_classes, counts = np.unique(y_temp, return_counts=True)
        rare_classes = unique_classes[counts < 2]
        if len(rare_classes) > 0:
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.40, random_state=42)
        else:
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.40, random_state=42, stratify=y_temp)
        
        # Train model
        num_classes = len(np.unique(y))
        new_model, history = build_and_train_model(X_train, y_train, X_val, y_val, num_classes)
        
        # Evaluate model and calculate metrics
        y_pred = np.argmax(new_model.predict(X_test), axis=1)
        metrics = calculate_metrics(y_test, y_pred)
        
        # Generate visualizations dynamically
        visualizations = generate_visualizations(history, y_test, y_pred, new_target_encoder)
        
        # Save model and preprocessors
        new_model.save('drug_prescription_model.keras', overwrite=True)
        joblib.dump(new_scaler, 'scaler.pkl')
        joblib.dump(new_target_encoder, 'label_enc.pkl')
        joblib.dump(new_label_encoders, 'label_encoders.pkl')
        
        # Update global variables
        model = new_model
        scaler = new_scaler
        label_encoders = new_label_encoders
        target_encoder = new_target_encoder
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "metrics": metrics,
            "visualizations": {
                "training_history": f"data:image/png;base64,{visualizations['training_history']}",
                "confusion_matrix": f"data:image/png;base64,{visualizations['confusion_matrix']}",
                "class_distribution": f"data:image/png;base64,{visualizations['class_distribution']}"
            },
            "dataset_size": len(df)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during retraining: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

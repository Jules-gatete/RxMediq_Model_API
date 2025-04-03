from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import uvicorn
import io
import base64
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Drug Prescription Prediction API",
    description="API for predicting drug prescriptions with advanced retraining and visualization features",
    version="1.1.0"
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

    class Config:
        schema_extra = {
            "example": {
                "disease": "Hypertension",  # Fixed syntax and corrected value
                "age": 45,
                "gender": "Male",
                "severity": "Moderate"
            }
        }

# Load pre-trained model and preprocessors
try:
    model = load_model('drug_prescription_model.keras')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('label_enc.pkl')
    logger.info("Pre-trained model and preprocessors loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or preprocessors: {str(e)}")
    raise Exception(f"Error loading model or preprocessors: {str(e)}")

# Store latest visualizations globally
last_visualizations = {}

# Preprocessing function for prediction
def preprocess_input(data: pd.DataFrame) -> np.ndarray:
    """Preprocess input data for prediction"""
    try:
        df = data.copy()
        for col, le in label_encoders.items():
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            if not set(df[col]).issubset(le.classes_):
                raise ValueError(f"Unknown value in column {col}. Expected: {le.classes_}")
            df[col] = le.transform(df[col])
        df[["age"]] = scaler.transform(df[["age"]])
        return df.values
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error in preprocessing: {str(e)}")

# Prediction function
def predict_drug(X_processed: np.ndarray) -> List[str]:
    """Make predictions using the loaded model"""
    try:
        predictions = model.predict(X_processed, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = target_encoder.inverse_transform(predicted_classes)
        return predicted_labels.tolist()
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

# Preprocessing function for training
def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocesses the dataset for training."""
    try:
        if df.empty:
            raise ValueError("Input dataset is empty")
        X = df.drop('drug', axis=1)
        y = df['drug']
        label_encoders = {}
        categorical_cols = ["disease", "gender", "severity"]
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        scaler = StandardScaler()
        numerical_cols = ["age"]
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        return X, y_encoded, label_encoders, scaler, target_encoder
    except Exception as e:
        logger.error(f"Data preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error in preprocessing data: {str(e)}")

# Model building and training function
def build_and_train_model(X_train, y_train, X_val, y_val, num_classes) -> tuple:
    """Build and train a neural network with optimization techniques."""
    try:
        model = Sequential([
            Dense(128, activation='relu', input_dim=X_train.shape[1]),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=Adagrad(learning_rate=0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
        ]
        history = model.fit(
            X_train, y_train,
            epochs=300,
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        return model, history
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in training: {str(e)}")

# Function to calculate metrics
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate performance metrics"""
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
    except Exception as e:
        logger.error(f"Metrics calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")

# Function to generate visualizations dynamically
def generate_visualizations(history, y_test, y_pred, target_encoder) -> Dict[str, str]:
    """Generate training history plot, confusion matrix, and class distribution in memory"""
    visualizations = {}

    # 1. Training History Plot
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    visualizations['training_history'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    buf.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 12})
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    visualizations['confusion_matrix'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    buf.close()

    # 3. Class Distribution of Predictions
    predicted_labels = target_encoder.inverse_transform(y_pred)
    plt.figure(figsize=(12, 6))
    sns.countplot(x=predicted_labels, palette='viridis', order=np.unique(predicted_labels))
    plt.title('Predicted Class Distribution', fontsize=14)
    plt.xlabel('Drug', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    visualizations['class_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    buf.close()

    return visualizations

# GET endpoint for single prediction
@app.get("/predict/")
async def get_prediction(disease: str, age: int, gender: str, severity: str):
    """Get drug prediction for a single patient"""
    try:
        logger.info(f"Received prediction request: disease={disease}, age={age}, gender={gender}, severity={severity}")
        if age < 0:
            raise HTTPException(status_code=400, detail="Age cannot be negative")
        input_data = pd.DataFrame([{"disease": disease, "age": age, "gender": gender, "severity": severity}])
        X_processed = preprocess_input(input_data)
        predictions = predict_drug(X_processed)
        logger.info(f"Prediction successful: {predictions[0]}")
        return {
            "status": "success",
            "prediction": predictions[0],
            "input_data": input_data.to_dict(orient="records")[0]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# POST endpoint for batch prediction
@app.post("/predict/batch/")
async def post_batch_prediction(patients: List[PatientData]):
    """Get drug predictions for multiple patients"""
    try:
        logger.info(f"Received batch prediction request for {len(patients)} patients")
        if not patients:
            raise HTTPException(status_code=400, detail="Empty patient list provided")
        input_data = pd.DataFrame([p.dict() for p in patients])
        if (input_data["age"] < 0).any():
            raise HTTPException(status_code=400, detail="Age cannot be negative")
        X_processed = preprocess_input(input_data)
        predictions = predict_drug(X_processed)
        results = [{"input": patients[i].dict(), "prediction": pred} for i, pred in enumerate(predictions)]
        logger.info(f"Batch prediction successful: {len(results)} predictions made")
        return {
            "status": "success",
            "predictions": results,
            "count": len(results)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# GET endpoint to retrieve latest visualizations
@app.get("/retrain/")
async def get_latest_visualizations():
    """Retrieve the latest training visualizations"""
    try:
        if not last_visualizations:
            raise HTTPException(status_code=404, detail="No visualizations available. Please retrain the model first.")
        return {
            "status": "success",
            "visualizations": last_visualizations
        }
    except Exception as e:
        logger.error(f"Error fetching visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching visualizations: {str(e)}")

# POST endpoint for retraining the model
@app.post("/retrain/")
async def retrain_model(file: UploadFile = File(...)):
    """Retrain the model with a new dataset and provide metrics and visualizations"""
    global model, scaler, label_encoders, target_encoder, last_visualizations
    try:
        logger.info(f"Received retrain request with file: {file.filename}")
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        required_columns = {"disease", "age", "gender", "severity", "drug"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        if (df["age"] < 0).any():
            raise HTTPException(status_code=400, detail="Age cannot be negative")
        
        # Preprocess data
        X, y, new_label_encoders, new_scaler, new_target_encoder = preprocess_data(df)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
        unique_classes, counts = np.unique(y_temp, return_counts=True)
        rare_classes = unique_classes[counts < 2]
        if len(rare_classes) > 0:
            logger.warning("Rare classes detected; stratification skipped for validation/test split")
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.40, random_state=42)
        else:
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.40, random_state=42, stratify=y_temp)
        
        # Train model
        num_classes = len(np.unique(y))
        new_model, history = build_and_train_model(X_train, y_train, X_val, y_val, num_classes)
        
        # Evaluate model and calculate metrics
        y_pred = np.argmax(new_model.predict(X_test, verbose=0), axis=1)
        metrics = calculate_metrics(y_test, y_pred)
        
        # Generate and store visualizations
        visualizations = generate_visualizations(history, y_test, y_pred, new_target_encoder)
        last_visualizations = {
            "training_history": f"data:image/png;base64,{visualizations['training_history']}",
            "confusion_matrix": f"data:image/png;base64,{visualizations['confusion_matrix']}",
            "class_distribution": f"data:image/png;base64,{visualizations['class_distribution']}"
        }
        
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
        logger.info(f"Model retrained successfully. Test accuracy: {test_accuracy}")
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "metrics": metrics,
            "num_classes": num_classes
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during retraining: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    try:
        logger.info("Health check requested")
        return {
            "status": "healthy",
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_loaded": model is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

from fastapi import FastAPI, HTTPException, UploadFile, File
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
from typing import List
import uvicorn
import io
import os

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

# Define static directory (though not used for visualizations anymore)
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

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

# Preprocessing function for prediction
def preprocess_input(data: pd.DataFrame) -> np.ndarray:
    """Preprocess input data for prediction"""
    try:
        df = data.copy()
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
            else:
                raise ValueError(f"Missing required column: {col}")
        df[["age"]] = scaler.transform(df[["age"]])
        return df.values
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in preprocessing: {str(e)}")

# Prediction function
def predict_drug(X_processed: np.ndarray) -> List[str]:
    """Make predictions using the loaded model"""
    try:
        predictions = model.predict(X_processed)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = target_encoder.inverse_transform(predicted_classes)
        return predicted_labels.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

# Preprocessing function for training
def preprocess_data(df: pd.DataFrame):
    """Preprocesses the dataset for training."""
    try:
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
        raise HTTPException(status_code=400, detail=f"Error in preprocessing data: {str(e)}")

# Model building and training function
def build_and_train_model(X_train, y_train, X_val, y_val, num_classes):
    """Build and train a neural network with optimization techniques."""
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        BatchNormalization(),
        Dropout(0.2),
        Dense(48, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adagrad(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        epochs=200,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        verbose=0,
                        callbacks=[EarlyStopping(monitor='val_loss',
                                                patience=20,
                                                restore_best_weights=True)])
    return model, history

# GET endpoint for single prediction
@app.get("/predict/")
async def get_prediction(disease: str, age: int, gender: str, severity: str):
    """Get drug prediction for a single patient"""
    try:
        input_data = pd.DataFrame([{"disease": disease, "age": age, "gender": gender, "severity": severity}])
        X_processed = preprocess_input(input_data)
        predictions = predict_drug(X_processed)
        return {"status": "success", "prediction": predictions[0], "input_data": input_data.to_dict(orient="records")[0]}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# POST endpoint for batch prediction
@app.post("/predict/batch/")
async def post_batch_prediction(patients: List[PatientData]):
    """Get drug predictions for multiple patients"""
    try:
        input_data = pd.DataFrame([p.dict() for p in patients])
        X_processed = preprocess_input(input_data)
        predictions = predict_drug(X_processed)
        results = [{"input": patients[i].dict(), "prediction": pred} for i, pred in enumerate(predictions)]
        return {"status": "success", "predictions": results, "count": len(results)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# POST endpoint for retraining the model
@app.post("/retrain/")
async def retrain_model(file: UploadFile = File(...)):
    """Retrain the model with a new dataset"""
    global model, scaler, label_encoders, target_encoder
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        required_columns = {"disease", "age", "gender", "severity", "drug"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        X, y, new_label_encoders, new_scaler, new_target_encoder = preprocess_data(df)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
        unique_classes, counts = np.unique(y_temp, return_counts=True)
        rare_classes = unique_classes[counts < 2]
        if len(rare_classes) > 0:
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.40, random_state=42)
        else:
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.40, random_state=42, stratify=y_temp)
        num_classes = len(np.unique(y))
        new_model, history = build_and_train_model(X_train, y_train, X_val, y_val, num_classes)
        new_model.save('drug_prescription_model.keras', overwrite=True)
        joblib.dump(new_scaler, 'scaler.pkl')
        joblib.dump(new_target_encoder, 'label_enc.pkl')
        joblib.dump(new_label_encoders, 'label_encoders.pkl')
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

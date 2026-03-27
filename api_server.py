import os
import numpy as np
import joblib
import torch
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
import time


app = Flask(__name__)

# Global variables to hold loaded models and device
DINO_MODEL = None
SCALER = None
SVM_MODEL = None
KNN_MODEL = None
DEVICE = None
TRANSFORM = None


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_models():
    """Load all models into memory on startup."""
    global DINO_MODEL, SCALER, SVM_MODEL, KNN_MODEL, DEVICE, TRANSFORM
    
    print("[INFO] Loading models...")
    start_time = time.time()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {DEVICE}")
    
    # Load DINOv2
    DINO_MODEL = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE)
    DINO_MODEL.eval()
    print("[INFO] Loaded DINOv2")
    
    # Load scaler and classifiers
    SCALER = joblib.load("scaler.joblib")
    SVM_MODEL = joblib.load("svm_model.joblib")
    KNN_MODEL = joblib.load("knn_model.joblib")
    print("[INFO] Loaded scaler and classifiers")
    
    TRANSFORM = build_transform()
    
    elapsed = time.time() - start_time
    print(f"[INFO] Models loaded in {elapsed:.2f}s")


def extract_embedding(image_path):
    """Extract DINOv2 embedding from image (model already in memory)."""
    img = Image.open(image_path).convert("RGB")
    img = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        emb = DINO_MODEL(img).cpu().numpy().squeeze()
    
    return emb


def predict_with_model(clf, emb_scaled):
    """Predict with a single classifier."""
    pred = int(clf.predict(emb_scaled)[0])
    
    label_map = {0: "noise", 1: "drone"}
    label = label_map.get(pred, "unknown")
    
    confidence = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(emb_scaled)[0]
        confidence = float(proba[pred])
    elif hasattr(clf, "decision_function"):
        score = float(clf.decision_function(emb_scaled)[0])
        confidence = float(1.0 / (1.0 + np.exp(-score)))
    
    return label, confidence, pred


@app.route("/predict", methods=["POST", "GET"])
def predict():
    """
    Predict drone/noise from a spectrogram image.
    
    POST: Upload image file
        curl -X POST -F "image=@path/to/image.png" -F "mode=both" http://localhost:5000/predict
        
    GET: Provide image path
        curl "http://localhost:5000/predict?image=path/to/image.png&mode=both"
    """
    try:
        # Get image and mode parameters
        image_path = None
        mode = request.args.get("mode", "both")
        
        if request.method == "POST":
            if "image" in request.files:
                # Handle file upload
                file = request.files["image"]
                image_path = os.path.join("/tmp", file.filename)
                file.save(image_path)
                mode = request.form.get("mode", "both")
            else:
                return jsonify({"error": "No image file provided"}), 400
        else:
            # Handle GET with image path
            image_path = request.args.get("image")
            if not image_path:
                return jsonify({"error": "No image path provided"}), 400
        
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image file not found: {image_path}"}), 404
        
        if mode not in ["svm", "knn", "both"]:
            return jsonify({"error": "mode must be 'svm', 'knn', or 'both'"}), 400
        
        # Extract embedding (fast because DINO is already in memory)
        start_time = time.time()
        emb = extract_embedding(image_path)
        emb = np.expand_dims(emb, axis=0)
        emb_scaled = SCALER.transform(emb)
        
        results = {}
        
        if mode in ["svm", "both"]:
            label, confidence, pred_class = predict_with_model(SVM_MODEL, emb_scaled)
            results["svm"] = {
                "prediction": label,
                "confidence": confidence,
                "confidence_percent": f"{confidence * 100:.2f}%" if confidence else None,
                "class": int(pred_class)
            }
        
        if mode in ["knn", "both"]:
            label, confidence, pred_class = predict_with_model(KNN_MODEL, emb_scaled)
            results["knn"] = {
                "prediction": label,
                "confidence": confidence,
                "confidence_percent": f"{confidence * 100:.2f}%" if confidence else None,
                "class": int(pred_class)
            }
        
        elapsed = time.time() - start_time
        
        return jsonify({
            "status": "success",
            "results": results,
            "inference_time_seconds": round(elapsed, 3),
            "device": DEVICE
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "device": DEVICE}), 200


if __name__ == "__main__":
    print("[INFO] Initializing Flask API server...")
    load_models()
    print("[INFO] Starting server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

import argparse
import numpy as np
import joblib
import torch
from PIL import Image
from torchvision import transforms


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


def extract_embedding(image_path, device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    model.eval()

    transform = build_transform()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img).cpu().numpy().squeeze()

    return emb


def predict_with_model(clf, emb_scaled):
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

    return label, confidence


def run_inference(image_path, scaler_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = joblib.load(scaler_path)

    emb = extract_embedding(image_path, device)
    emb = np.expand_dims(emb, axis=0)
    return scaler.transform(emb)


def main():
    parser = argparse.ArgumentParser(
        description="Predict drone/noise from one spectrogram image"
    )
    parser.add_argument("image", help="Path to spectrogram image")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["svm", "knn", "both"],
        help="Choose model to run: svm, knn, or both for comparison",
    )
    parser.add_argument(
        "--svm-model",
        default="svm_model.joblib",
        help="Path to trained SVM model (.joblib)",
    )
    parser.add_argument(
        "--knn-model",
        default="knn_model.joblib",
        help="Path to trained KNN model (.joblib)",
    )
    parser.add_argument(
        "--scaler",
        default="scaler.joblib",
        help="Path to saved scaler (.joblib)",
    )
    args = parser.parse_args()

    emb_scaled = run_inference(args.image, scaler_path=args.scaler)

    models_to_run = []
    if args.mode in ["svm", "both"]:
        models_to_run.append(("SVM", args.svm_model))
    if args.mode in ["knn", "both"]:
        models_to_run.append(("KNN", args.knn_model))

    for model_name, model_path in models_to_run:
        clf = joblib.load(model_path)
        result, confidence = predict_with_model(clf, emb_scaled)

        print(f"=== {model_name} ===")
        if result == "drone":
            print("Ket qua: CO drone")
        elif result == "noise":
            print("Ket qua: KHONG co drone")
        else:
            print("Ket qua: khong xac dinh")

        if confidence is not None:
            print(f"Do tin cay: {confidence:.4f} ({confidence * 100:.2f}%)")
        print()


if __name__ == "__main__":
    main()

import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import pickle
import tensorflow as tf
import traceback
from werkzeug.utils import secure_filename

MODEL_PATH = "model.h5"
ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"
HAAR_PATH = "haarcascade_frontalface_default.xml"

try:
    from flask_cors import CORS
except:
    CORS = None

app = Flask(__name__)
if CORS:
    CORS(app)

# Load model & sklearn objects
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

face_cascade = cv2.CascadeClassifier(HAAR_PATH)
if face_cascade.empty():
    raise FileNotFoundError("Haar cascade not found!")


# ==========================================
# TRAINING-COMPATIBLE PREPROCESSING
# ==========================================
def preprocess_face(face_roi, size=(100, 100)):
    face = cv2.resize(face_roi, size)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    face = clahe.apply(face)
    return face


# ==========================================
# TRAINING-COMPATIBLE GABOR KERNELS
# ==========================================
def build_gabor_kernels():
    kernels = []
    ksize = 9
    sigmas = [2.0, 4.0, 6.0]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    lambdas = [3.0, 6.0, 9.0]
    gammas = [0.5, 0.75, 1.0]

    for sigma in sigmas:
        for theta in thetas:
            for lam in lambdas:
                for gamma in gammas:
                    k = cv2.getGaborKernel(
                        (ksize, ksize), sigma, theta, lam, gamma, 
                        psi=0, ktype=cv2.CV_32F
                    )
                    kernels.append(k)

    print(f"API: Built {len(kernels)} Gabor kernels (should be 108)")
    return kernels

kernels = build_gabor_kernels()


# ==========================================
# TRAINING-COMPATIBLE FEATURE EXTRACTION
# ==========================================
def extract_advanced_features(face, kernels):
    features = []

    # 1. Gabor mean + std (2 features per kernel)
    for kern in kernels:
        filtered = cv2.filter2D(face.astype(np.float32), -1, kern)
        features.append(np.mean(filtered))
        features.append(np.std(filtered))

    # 2. Edge density
    edges = cv2.Canny(face, 50, 150)
    edge_density = np.sum(edges > 0) / (face.shape[0] * face.shape[1])
    features.append(edge_density)

    # 3. Laplacian variance
    laplacian = cv2.Laplacian(face, cv2.CV_64F)
    features.append(np.var(laplacian))

    # 4. Statistical features
    features.extend([
        np.mean(face),
        np.std(face),
        np.median(face),
        np.min(face),
        np.max(face),
        np.percentile(face, 25),
        np.percentile(face, 75),
    ])

    # 5. Histogram features (16 bins)
    hist = cv2.calcHist([face], [0], None, [16], [0, 256])
    hist = hist.flatten() / (face.size + 1e-6)
    features.extend(hist)

    return np.array(features, dtype=np.float32)


# ==========================================
# IMAGE READING HELPERS
# ==========================================
def read_image_from_request():
    file_storage = request.files.get("image", None)
    if file_storage:
        _ = secure_filename(file_storage.filename or "upload.jpg")
        return Image.open(file_storage.stream).convert("RGB")

    json_data = request.get_json(silent=True)
    if json_data and "image_base64" in json_data:
        img_data = base64.b64decode(json_data["image_base64"])
        return Image.open(io.BytesIO(img_data)).convert("RGB")

    return None


# ==========================================
# PREDICT ENDPOINT
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_pil = read_image_from_request()
        if img_pil is None:
            return jsonify({"error": "No image provided"}), 400

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return jsonify({"error": "No face detected"}), 200

        x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
        face_roi = gray[y:y+h, x:x+w]

        # TRAINING-COMPATIBLE PREPROCESSING
        face = preprocess_face(face_roi)

        # TRAINING-COMPATIBLE FEATURE EXTRACTION
        features = extract_advanced_features(face, kernels)
        features = features.reshape(1, -1)

        # Check match with scaler
        if scaler.n_features_in_ != features.shape[1]:
            msg = f"Feature length mismatch: scaler expects {scaler.n_features_in_}, got {features.shape[1]}"
            print(msg)
            return jsonify({"error": msg}), 500

        # Transform & predict
        scaled = scaler.transform(features)
        preds = model.predict(scaled)
        idx = int(np.argmax(preds, axis=1)[0])
        prob = float(np.max(preds))

        pred_label = le.inverse_transform([idx])[0]

        return jsonify({
            "predicted_age": str(pred_label),
            "confidence": prob,
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return """
    <h3>Age Detection API</h3>
    <form method="POST" action="/predict" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" />
      <button type="submit">Predict</button>
    </form>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

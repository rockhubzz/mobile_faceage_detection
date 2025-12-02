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

# -------------------------
# Load model + artifacts
# -------------------------
MODEL_PATH = "coba_gabor.h5"
ENCODER_PATH = "label_encoder_coba.pkl"
SCALER_PATH = "scaler_coba.pkl"
HAAR_PATH = "haarcascade_frontalface_default.xml"

# Optional: allow CORS if your Flutter app is on a different host during dev
try:
    from flask_cors import CORS
except Exception:
    CORS = None

app = Flask(__name__)
if CORS:
    CORS(app)

# Load TF model and sklearn objects
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Haar cascade
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
if face_cascade.empty():
    raise FileNotFoundError(f"Haar cascade file not found or failed to load: {HAAR_PATH}")

# -------------------------
# Build Gabor kernels
# -------------------------
def build_gabor_kernels():
    ksize = 31
    scales = 10
    orientations = 8
    kernels = []

    for theta in np.linspace(0, np.pi, orientations, endpoint=False):
        for sigma in np.linspace(1, 5, scales):
            kern = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta,
                10.0,
                0.5,
                0,
                ktype=cv2.CV_32F
            )
            kernels.append(kern)

    print("API kernels:", len(kernels))  # should print 80
    return kernels

kernels = build_gabor_kernels()

# -------------------------
# Apply Gabor and extract features
# -------------------------
def apply_gabor_filters(img_gray, kernels):
    """
    img_gray: single-channel 2D numpy array (grayscale)
    returns: list of scalar features, one per kernel (mean of filtered response)
    """
    feats = []
    # ensure float32 for filtering
    img_f = img_gray.astype(np.float32)
    for kern in kernels:
        # use CV_32F ddepth to preserve values (avoid truncation)
        fimg = cv2.filter2D(img_f, ddepth=cv2.CV_32F, kernel=kern)
        feats.append(float(np.mean(fimg)))
    return feats

# -------------------------
# Helper: read image from Flask request
# -------------------------
def read_image_from_request():
    """
    Accepts:
     - multipart form 'image' (file)
     - JSON {'image_base64': '<base64...>'}
    Returns PIL.Image (RGB)
    """
    # Prefer multipart form file
    file_storage = request.files.get("image", None)
    if file_storage:
        # secure filename (not strictly required here)
        _ = secure_filename(file_storage.filename or "upload.jpg")
        img = Image.open(file_storage.stream).convert("RGB")
        return img

    # Try JSON base64
    json_data = request.get_json(silent=True)
    if json_data and "image_base64" in json_data:
        img_data = base64.b64decode(json_data["image_base64"])
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        return img

    # nothing found
    return None

# -------------------------
# Predict endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_pil = read_image_from_request()
        if img_pil is None:
            return jsonify({"error": "No image provided. Send multipart 'image' or JSON {'image_base64':...}"}), 400

        # Convert to OpenCV BGR
        img = np.array(img_pil)  # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces (safe call)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            # return 200 with helpful message so client can handle it gracefully
            return jsonify({"error": "No face detected"}), 200

        # choose largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        # ensure ROI inside image bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
        face_roi = gray[y1:y2, x1:x2]

        # resize to the size used in training (you used 100x100 in your colab snippet)
        face_roi = cv2.resize(face_roi, (100, 100))

        # extract Gabor features
        feat_list = apply_gabor_filters(face_roi, kernels)
        feat = np.array(feat_list, dtype=np.float32).reshape(1, -1)

        # Verify feature length matches scaler expectation
        expected_n = getattr(scaler, "n_features_in_", None)
        got_n = feat.shape[1]
        if expected_n is not None and expected_n != got_n:
            # Helpful error message for debugging
            msg = f"Feature length mismatch: scaler expects {expected_n} features, but extracted {got_n}."
            # include a little more info in server log
            print(msg)
            return jsonify({"error": msg, "expected": expected_n, "got": got_n}), 500

        # scale
        feat_scaled = scaler.transform(feat)

        # predict
        pred = model.predict(feat_scaled)
        pred_idx = int(np.argmax(pred, axis=1)[0])
        try:
            pred_class = le.inverse_transform([pred_idx])[0]
        except Exception:
            # if label encoder fails, fall back to index string
            pred_class = str(pred_idx)

        prob = float(np.max(pred))

        # return also face bbox if client wants
        return jsonify({
            "predicted_age": str(pred_class),
            "confidence": prob,
            "bbox": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}
        })

    except Exception as ex:
        # print full traceback to server terminal for debugging
        print("\n===== SERVER ERROR =====")
        traceback.print_exc()
        print("========================\n")
        # Return a short error message to client (avoid returning full traceback in production)
        return jsonify({"error": str(ex)}), 500

# -------------------------
# Optional: simple browser test page
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return """
    <h3>Face Age Detection API (test page)</h3>
    <form method="POST" action="/predict" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*"/>
      <button type="submit">Upload & Predict</button>
    </form>
    <p>Use multipart form field name <code>image</code> or JSON <code>{'image_base64': '...'}</code></p>
    """

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    # keep debug=True so you get helpful auto-restarts during development
    app.run(host="0.0.0.0", port=5000, debug=True)

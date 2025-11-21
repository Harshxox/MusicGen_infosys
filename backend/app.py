# backend/app.py
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os, uuid
from gtzan import predict_genre_from_file  # uses the model loaded in gtzan.py
from flask_cors import CORS

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # in production restrict origins

@app.route("/api/classify", methods=["POST"])
def classify():
    # Optionally verify Firebase token here (recommended)
    # header = request.headers.get("Authorization","")
    # Verify with firebase-admin then continue

    if 'file' not in request.files:
        return jsonify({"error":"no file uploaded"}), 400
    f = request.files['file']
    filename = secure_filename(f.filename) or (uuid.uuid4().hex + ".wav")
    out_path = os.path.join(UPLOAD_DIR, uuid.uuid4().hex + "_" + filename)
    f.save(out_path)
    try:
        result = predict_genre_from_file(out_path)
        # delete file after processing
        os.remove(out_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)

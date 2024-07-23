import os
from flask import Flask, request, jsonify
from flask_cors import CORS

FLASK_RUN_HOST = os.environ.get('FLASK_RUN_HOST', "localhost")
FLASK_RUN_PORT = os.environ.get('FLASK_RUN_PORT', "8000")

UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return "HellOOOO"

@app.route('/upload_file', methods=["POST"])
def upload_file():
    if 'files' not in request.files:
        return jsonify({"message": "No files part"}), 400
    files = request.files.getlist('files')
    file_paths = []

    for file in files:
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        file_paths.append(file_path)

    return jsonify({"message": "Files successfully uploaded", "files": file_paths}),200

if __name__ == "__main__":
    app.run(host=FLASK_RUN_HOST,port=FLASK_RUN_PORT)
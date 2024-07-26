import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from server.main_chat import generate_answer
from server.database_utils.update_database import update_database
from server.database_utils.delete_database import delete_database

FLASK_RUN_HOST = os.environ.get('FLASK_RUN_HOST', "localhost")
FLASK_RUN_PORT = os.environ.get('FLASK_RUN_PORT', "8000")

UPLOAD_FOLDER = "data"

app = Flask(__name__)
CORS(app)


# @app.route('/')
# def index():
#     return "HellOOOO"

@app.route('/upload_file', methods=["POST"])
def upload_file():
    if 'files' not in request.files:
        return jsonify({"message": "No files uploaded", "status":"False"}), 400
    files = request.files.getlist('files')
    file_paths = []
    
    for file in files:
        if file.filename == '':
            return jsonify({"message": "No selected file", "status":"False"}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        file_paths.append(file_path)
        
    intro_message = "Hello there! Please start by typing something you would like to know based on uploaded documents"
    update_database()
    return jsonify({"message": intro_message, "files":file_paths, "status":"True"}),200

@app.route('/delete_data',methods=["DELETE"])
def delete_data():
    delete_database()
    return jsonify({"message": "Database deleted"}), 200
    
    
@app.route('/chat', methods=["POST"]) 
def chat():
    message = request.json.get("message")
    bot_response = generate_answer(message)
    return jsonify({"message": bot_response}),200

@app.route('/chat_stream', methods=["POST"]) 
def chat_stream():
    message = request.json.get("message")
    return Response(generate_answer(message),content_type="text/plain"),200
    
if __name__ == "__main__":
    app.run(host=FLASK_RUN_HOST,port=FLASK_RUN_PORT)
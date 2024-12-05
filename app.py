import os
import cv2
import numpy as np
import io
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from server.main_chat import generate_answer
from server.database_utils.update_database import update_database
from server.database_utils.delete_database import delete_database
from opencv_doc_detect.document_preprocessing import main_preprocessing
from opencv_doc_detect.text_detection import text_detect
from opencv_doc_detect.text_pdf import text_to_pdf

FLASK_RUN_HOST = os.environ.get('FLASK_RUN_HOST', "localhost")
FLASK_RUN_PORT = os.environ.get('FLASK_RUN_PORT', "8000")

UPLOAD_FOLDER = "data"

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return "HellOOOO"

@app.route('/image_processing', methods=["POST"])
def image_processing():

    if 'image' not in request.files:
        return {'error': 'No image file provided'}, 400
        
    image_file = request.files['image']
    base_filename = os.path.splitext(image_file.filename)[0]
    
    # Read image file into memory
    in_memory_file = io.BytesIO()
    image_file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    processed_image = main_preprocessing(image)
    
    save_path = "opencv_doc_detect/" + base_filename + ".jpg"
    cv2.imwrite(save_path,processed_image)
    
    # Convert processed image back to bytes
    is_success, buffer = cv2.imencode(".jpg", processed_image)
    if not is_success:
        return {'error': 'Failed to encode output image'}, 500
        
    # Prepare the response
    io_buf = io.BytesIO(buffer)
    io_buf.seek(0)
    
    return send_file(
        io_buf,
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='processed_image.jpg'
    )
    
@app.route('/image_to_text', methods=["POST"])
def image_to_text():
    if 'image' not in request.files:
        return {'error': 'No image file provided'}, 400
        
    image_file = request.files['image']
    base_filename = os.path.splitext(image_file.filename)[0]
    save_path = "opencv_doc_detect/" + base_filename + ".jpg"
    
    try:
        extracted_text = text_detect(save_path)
        print("extracted text:", extracted_text)
    except Exception as e:
        return jsonify({"error": f"Text detection failed: {str(e)}"}), 500
    
    return jsonify({"extracted_text": extracted_text}), 200

@app.route('/text_to_database', methods=["POST"])
def text_to_database():
    if 'submit_text' not in request.form:
        return jsonify({'error': 'Extracted Text provided'}), 400
    
    extracted_text = request.form['submit_text']
    image_file = request.files['image']
    
    pdf_file = text_to_pdf(extracted_text)
    
    base_filename = os.path.splitext(image_file.filename)[0]
    file_path = os.path.join(UPLOAD_FOLDER, base_filename+".pdf")
    pdf_file.output(file_path)
    
    return send_file(file_path, as_attachment=True, download_name=f"{base_filename}.pdf", mimetype='application/pdf')


    
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
    app.run(host=FLASK_RUN_HOST,port=FLASK_RUN_PORT,debug=True)
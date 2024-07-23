import os
from flask import Flask
from flask_cors import CORS

FLASK_RUN_HOST = os.environ.get('FLASK_RUN_HOST', "localhost")
FLASK_RUN_PORT = os.environ.get('FLASK_RUN_PORT', "8000")

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return "HellOOOO"

if __name__ == "__main__":
    app.run(host=FLASK_RUN_HOST,port=FLASK_RUN_PORT)
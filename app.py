from flask import Flask, render_template, request, redirect, url_for
from matplotlib import pyplot as plt

from face_detection import detect_face
import os

app = Flask(__name__)

basedir = os.path.dirname(os.path.realpath(__file__))

import io
import base64

def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'jpeg', quality=100)
    img_io.seek(0)
    img = base64.b64encode(img_io.getvalue()).decode('ascii')
    img_tag = f'<img src="data:image/jpg;base64,{img}" class="img-fluid"/>'
    return img_tag


@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    faces = []
    if request.method == "POST":
        file_one = request.files['file_one']
        file_two = request.files['file_two']

        filename_one = os.path.join(basedir, file_one.filename)
        filename_two = os.path.join(basedir, file_two.filename)

        file_one.save(filename_one)
        file_two.save(filename_two)
        
        faces = detect_face(file_one.filename, file_two.filename)
        faces_as_html = []

        for _faces in faces:
            for f in _faces:
                faces_as_html.append(serve_pil_image(f))
        faces = faces_as_html

        os.remove(filename_one)
        os.remove(filename_two)

    return render_template("index.html", faces=faces, str=str, enumerate=enumerate)

if __name__ == "__main__":
    app.run(debug=True)
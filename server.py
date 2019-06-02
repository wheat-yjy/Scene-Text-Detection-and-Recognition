import os

from flask import Flask, request, json
import matplotlib.pyplot as plt
import imageio
from process import *
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/upload_img', methods=['post'])
def upload_img():
    img = request.files.get('bitmap')
    path = basedir + "/img/" + img.filename
    img.save(path)
    img2 = imageio.imread(path)
    ret = predict(img2)
    plt.imshow(img2)
    plt.show()
    # 'degree' 可选
    return json.jsonify(ret)
    # return json.jsonify([{'x0': 10, 'x1': 1000, 'y0': 1000, 'y1': 2000, 'text': 'text1'},
    #                      {'x0': 10, 'x1': 1000, 'y0': 2000, 'y1': 3000, 'text': 'text2'}])


if __name__ == '__main__':
    load_net()
    load_model()
    app.run(host='0.0.0.0')

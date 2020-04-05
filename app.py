from flask import Flask, render_template, request
import numpy as np
import svm

app = Flask(__name__)

@app.route('/')
def hi():
    return render_template('index.html')


@app.route('/dataset')
def data_set():
    return render_template('dataset.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        my_val = [
            float(request.form['ip1']),
            float(request.form['ip2']),
            float(request.form['ip3']),
            float(request.form['ip4']),
            float(request.form['ip5']),
            0
        ]

        w = svm.find_hyperplane()

        result = np.dot(my_val, w)

        if result < w[5]:
            result_diskrit = "Tidak Tepat Waktu"
        else:
            result_diskrit = "Tepat Waktu"

        result_fin = {
            "x1": w[0],
            "x2": w[1],
            "x3": w[2],
            "x4": w[3],
            "x5": w[4],
            "bias": w[5],
            "result": result,
            "result_diskrit": result_diskrit
        }

        return render_template('result.html', result=result_fin)

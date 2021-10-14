import flask
import numpy as np
import pickle
from markupsafe import escape

app = flask.Flask(__name__, template_folder='templates')

model = pickle.load(open('model/obj.pkl', 'rb'))


@app.route('/')
def index():
    return (flask.render_template('main.html'))


@app.route('/predict', methods=["POST"])
def predict():

    features = [float(x) for x in flask.request.form.values()]
    scaled = model['scaler'].transform([features[:7]])
    features = np.array(features[7:])
    features = np.concatenate((np.reshape(scaled, 7),features))
    prediction = model['model'].predict([features])

    output = {0: 'survive', 1: 'die'}

    return flask.render_template('main.html', prediction_text='The patient will {} .'.format(output[prediction[0]]))


if __name__ == '__main__':
    app.run(debug=True)

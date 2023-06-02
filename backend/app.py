from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load the model
model = tf.keras.models.load_model('./model')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('./model/label_encoder.npy', allow_pickle=True)

app = Flask(__name__)

@app.route('/')
def index():
    try:
        hello_json = {
            'status_code': 200,
            'message': 'Success testing the API!',
            'data': [],
        }
        return jsonify(hello_json)
    except Exception as e:
        error_json = {
            'status_code': 500,
            'message': 'Error occurred.',
            'error_details': str(e),
        }
        return jsonify(error_json), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get the input data from the request
        dataframe = pd.read_csv('./data/akc-data-latest-final.csv')

        new_data_dict = {}
        for key, value in data.items():
            if key != "predictions":
                new_data_dict[key] = tf.constant(value, dtype=tf.string)

        # Make predictions
        predictions = model.predict(new_data_dict)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_labels = label_encoder.inverse_transform(predicted_labels)
        predicted_labels = predicted_labels.tolist()

        result = {'predictions': []}
        for label in predicted_labels:
            breeds = dataframe.loc[dataframe['group'] == label]
            breeds = breeds.sort_values('popularity', ascending=False)
            top_breeds = breeds.head(5)
            result['predictions'].append({
                'label': label,
                'top_breeds': top_breeds.to_dict(orient='records')
            })

        return jsonify(result)

    except Exception as e:
        error_json = {
            'status_code': 500,
            'message': 'Error occurred during prediction.',
            'error_details': str(e),
        }
        return jsonify(error_json), 500

if __name__ == '__main__':
    app.run(debug=True)


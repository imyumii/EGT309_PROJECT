from flask import Blueprint, request, jsonify
import pandas as pd
from .preprocess import preprocess
from .train import train_model
from .infer import infer_model
import pickle

main = Blueprint('main', __name__)

@main.route('/preprocess', methods=['POST'])
def preprocess_data():
    preprocessed_data = preprocess()
    return preprocessed_data.to_json()

@main.route('/train', methods=['POST'])
def train():
    try:
        preprocessed_data_json = request.get_json()
        if not preprocessed_data_json:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert JSON data (which is already a dictionary) to a DataFrame
        preprocessed_data = pd.DataFrame.from_dict(preprocessed_data_json)

        model = train_model(preprocessed_data)
        model_path = 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return jsonify({'message': 'Model trained and saved', 'model_path': model_path})
    except ValueError as e:
        return jsonify({'error': 'Invalid data format: ' + str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/infer', methods=['POST'])
def infer():
    try:
        infer_data_json = request.get_json()
        if not infer_data_json:
            return jsonify({'error': 'No data provided'}), 400
        
        # Print the incoming data for debugging
        print(f"Received infer data: {infer_data_json}")

        # Convert JSON data (which is already a dictionary) to a DataFrame
        infer_data = pd.DataFrame.from_dict(infer_data_json)
        print(f"Converted infer data to DataFrame:\n{infer_data}")

        model_path = 'model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        predictions = infer_model(model, infer_data)
        return jsonify({'predictions': predictions.tolist()})
    except ValueError as e:
        return jsonify({'error': 'Invalid data format: ' + str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

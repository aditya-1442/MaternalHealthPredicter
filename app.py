import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('rf_model.pkl', 'rb') as f:  # Load the model using pickle
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    values = data.get('values')

    if values is None or len(values) != 6:
        return jsonify({'error': 'Invalid input: Please provide 6 integer values'}), 400

    try:
        values = [int(value) for value in values]  # Ensure integer conversion
        prediction = model.predict([values])  # Pass as a 2D array for model compatibility
        result_string = str(prediction[0])  # Extract the single string output
        return jsonify({'result': result_string}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production

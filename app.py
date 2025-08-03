from flask import Flask,request,jsonify
import numpy as np

app=Flask(__name__)

import joblib
#load the model
model=joblib.load('rf_model.pkl')

#website --->button --->/about (url)----> def about(design + code)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input']).reshape(1, -1)
        prediction = model.predict(input_data)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

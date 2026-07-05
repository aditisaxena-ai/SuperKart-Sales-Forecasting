import datetime
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
CURRENT_YEAR = datetime.date.today().year
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',
    'model',
    'superkart_sales_model_v1_0.joblib'
)

model = joblib.load(os.path.normpath(MODEL_PATH))

REQUIRED_FIELDS = [
    'Product_Id',
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Allocated_Area',
    'Product_Type',
    'Product_MRP',
    'Store_Establishment_Year',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]


def build_features(payload: dict) -> pd.DataFrame:
    product_id = str(payload['Product_Id'])
    product_mrp = float(payload['Product_MRP'])
    allocated_area = float(payload['Product_Allocated_Area'])
    store_year = int(payload['Store_Establishment_Year'])

    sample = {
        'Product_Weight': float(payload['Product_Weight']),
        'Product_Allocated_Area': allocated_area,
        'Product_MRP': product_mrp,
        'Store_Age': float(CURRENT_YEAR - store_year),
        'Price_Area_Interaction': float(product_mrp * allocated_area),
        'Product_Sugar_Content': payload['Product_Sugar_Content'],
        'Product_Type': payload['Product_Type'],
        'Store_Size': payload['Store_Size'],
        'Store_Location_City_Type': payload['Store_Location_City_Type'],
        'Store_Type': payload['Store_Type'],
        'Product_Category_Code': product_id[:2],
        'MRP_Category': 'Low' if product_mrp <= 100 else 'Medium' if product_mrp <= 200 else 'High'
    }

    return pd.DataFrame([sample])


@app.route('/')
def index():
    return jsonify({'message': 'SuperKart Sales Forecast API is running.'})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)

    if not payload:
        return jsonify({'error': 'Request body must be valid JSON.'}), 400

    missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing_fields:
        return jsonify({
            'error': 'Missing required fields.',
            'missing_fields': missing_fields
        }), 400

    input_df = build_features(payload)
    prediction = model.predict(input_df)[0]
    return jsonify({
        'predicted_quarterly_revenue': round(float(prediction), 2)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import argparse
import datetime
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CURRENT_YEAR = datetime.date.today().year

NUMERIC_FEATURES = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Age',
    'Price_Area_Interaction'
]

CATEGORICAL_FEATURES = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Product_Category_Code',
    'MRP_Category'
]

TARGET_COLUMN = 'Product_Store_Sales_Total'


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Store_Age'] = CURRENT_YEAR - df['Store_Establishment_Year']
    df['Product_Category_Code'] = df['Product_Id'].astype(str).str[:2]
    df['Price_Area_Interaction'] = df['Product_MRP'] * df['Product_Allocated_Area']
    df['MRP_Category'] = pd.cut(
        df['Product_MRP'],
        bins=[0, 100, 200, 500],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    df['MRP_Category'] = df['MRP_Category'].astype(str).fillna('Unknown')
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder='drop'
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=200, random_state=42))
        ]
    )
    return pipeline


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)
    r2 = r2_score(y, predictions)
    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2': float(r2)
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train the SuperKart sales forecasting model.')
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/SuperKart.csv',
        help='Path to the training CSV file.'
    )
    parser.add_argument(
        '--output-model',
        type=str,
        default='model/superkart_sales_model_v1_0.joblib',
        help='Output path for the trained model.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f'Data file not found: {args.data_path}. Please provide a valid CSV dataset path.'
        )

    df = pd.read_csv(args.data_path)
    df = create_features(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f'Expected target column {TARGET_COLUMN} in the dataset.')

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    train_metrics = evaluate_model(pipeline, X_train, y_train)
    test_metrics = evaluate_model(pipeline, X_test, y_test)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(pipeline, args.output_model)

    print('Model training complete.')
    print('Saved model to:', args.output_model)
    print('\nTraining metrics:')
    for key, value in train_metrics.items():
        print(f'  {key}: {value:.4f}')

    print('\nTest metrics:')
    for key, value in test_metrics.items():
        print(f'  {key}: {value:.4f}')


if __name__ == '__main__':
    main()

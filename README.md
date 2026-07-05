# SuperKart — Sales Forecasting Model Deployment

## Overview

SuperKart is a multi-city supermarket and food mart chain. This project delivers a predictive sales forecasting model that estimates **quarterly sales revenue per outlet**, packaged as a production-ready, API-accessible service. The goal is to give business and regional teams a reliable, data-driven tool to optimize inventory planning and sales strategy across stores.

## Problem Statement

SuperKart needed accurate, outlet-level quarterly revenue forecasts to:
- Optimize inventory allocation across multiple cities and store formats
- Inform regional sales strategy and planning cycles
- Move away from manual/heuristic forecasting toward a scalable, repeatable, data-driven process

## Approach

The project followed an end-to-end ML lifecycle, from raw data to a deployed service:

1. **Exploratory Data Analysis (EDA)** — Investigated sales trends, outlet-level patterns, seasonality, and feature distributions to understand key revenue drivers.
2. **Data Preprocessing** — Cleaned raw data, handled missing values and outliers, encoded categorical variables, and prepared data for modeling.
3. **Feature Engineering** — Derived new features (e.g., outlet attributes, time-based aggregates) to improve model predictive power.
4. **Model Building & Hyperparameter Tuning** — Built and evaluated regression models, tuning hyperparameters to optimize forecasting accuracy.
5. **Containerization** — Packaged the trained model and its dependencies into a Docker container for consistent, portable deployment.
6. **API Development** — Exposed the model through a Flask REST API, enabling other systems and teams to request forecasts programmatically.

## Tools & Skills

| Category | Tools / Skills |
|---|---|
| Analysis | Exploratory Data Analysis (EDA) |
| Data Prep | Data Preprocessing, Feature Engineering |
| Modeling | Regression Models, Hyperparameter Tuning |
| Deployment | Docker, Flask, API Development |


## Getting Started

### Prerequisites
- Python 3.9+
- Docker

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd SuperKart

# Install dependencies
pip install -r requirements.txt

# Place your dataset in data/SuperKart.csv
# Train the model
python src/train.py --data-path data/SuperKart.csv

# Run the Flask API locally
python app/app.py
```

### Running with Docker

```bash
# Build the image
docker build -t superkart-forecast:latest .

# Run the container
docker run -p 5000:5000 superkart-forecast:latest
```

## API Usage

Once running, the API exposes a prediction endpoint:

**Endpoint:** `POST /predict`

**Sample Request:**
```json
{
  "Product_Id": "FD001",
  "Product_Weight": 15.2,
  "Product_Sugar_Content": "Low",
  "Product_Allocated_Area": 0.32,
  "Product_Type": "Fruits and Vegetables",
  "Product_MRP": 150.0,
  "Store_Establishment_Year": 2014,
  "Store_Size": "Medium",
  "Store_Location_City_Type": "Tier 2",
  "Store_Type": "Supermarket Type2"
}
```

**Sample Response:**
```json
{
  "predicted_quarterly_revenue": 452300.75
}
```

> The model automatically derives `Store_Age`, `Price_Area_Interaction`, `Product_Category_Code`, and `MRP_Category` from the request body.

## Result

The project delivered a fully deployable, API-accessible sales forecasting service, enabling SuperKart's business and regional teams to make faster, data-driven decisions on inventory allocation and sales strategy at scale.

## Future Enhancements
- Add model monitoring and drift detection
- Automate retraining pipeline (CI/CD)
- Deploy to cloud infrastructure (AWS/Azure/GCP) with load balancing
- Add authentication and rate limiting to the API

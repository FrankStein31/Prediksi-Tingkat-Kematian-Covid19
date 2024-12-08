from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime

app = Flask(__name__)

def load_model():
    df = pd.read_csv('Data_COVID19_Indonesia.csv')
    
    df = df.dropna(subset=['New Cases', 'New Deaths', 'Total Cases', 'Population'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days Since Start'] = (df['Date'] - df['Date'].min()).dt.days

    features = ['Days Since Start', 'Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density', 'New Deaths', 'Growth Factor of New Cases']
    target = 'Case Fatality Rate'

    df['Case Fatality Rate'] = (df['Total Deaths'] / df['Total Cases']) * 100
    df[features] = df[features].fillna(df[features].mean())

    X = df[features]
    y = df[target]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler, features, df

MODEL, SCALER, FEATURES, FULL_DF = load_model()

def get_monthly_data(df):
    df['Bulan'] = df['Date'].dt.to_period('M')
    monthly_data = df.groupby('Bulan')['Case Fatality Rate'].mean().reset_index()
    monthly_data['Bulan'] = monthly_data['Bulan'].astype(str)
    return {
        'bulan': monthly_data['Bulan'].tolist(),
        'cfr': monthly_data['Case Fatality Rate'].round(2).tolist()
    }

def get_yearly_data(df):
    yearly_data = df.groupby(df['Date'].dt.year)['Case Fatality Rate'].mean().reset_index()
    return {
        'tahun': yearly_data['Date'].tolist(),
        'cfr': yearly_data['Case Fatality Rate'].round(2).tolist()
    }

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    input_date = None
    monthly_data = get_monthly_data(FULL_DF)
    yearly_data = get_yearly_data(FULL_DF)

    if request.method == 'POST':
        try:
            input_date = request.form['prediction_date']
            
            input_date_dt = pd.to_datetime(input_date)

            latest_features = FULL_DF.iloc[-1:].copy()
            latest_features['Days Since Start'] = (input_date_dt - FULL_DF['Date'].min()).days

            latest_features = latest_features[FEATURES] 
            latest_features = latest_features.apply(pd.to_numeric, errors='coerce')
            latest_features = latest_features.fillna(latest_features.mean())

            latest_features_scaled = SCALER.transform(latest_features)

            prediction = MODEL.predict(latest_features_scaled)[0]

        except Exception as e:
            prediction = str(e)

    return render_template('index.html', 
                            prediction=prediction, 
                            input_date=input_date, 
                            monthly_data=monthly_data,
                            yearly_data=yearly_data)

if __name__ == '__main__':
    app.run(debug=True)
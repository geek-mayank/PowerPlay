from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.pipeline.predict_pipeline import ElectricityData, PredictPipeline
from src.logger import logging
from src.exception import CustomException
import sys

application = Flask(__name__)
app = application

@app.route('/')
def index():
    """Home page route for electricity consumption prediction"""
    try:
        logging.info("Electricity consumption prediction home page accessed")
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error in index route: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Prediction route for electricity consumption"""
    try:
        if request.method == 'GET':
            logging.info("Electricity consumption prediction form accessed")
            return render_template('home.html')
        else:
            # Get form data
            timestamp = request.form.get('timestamp')
            day_of_week = request.form.get('day_of_week')
            hour_of_day = request.form.get('hour_of_day')
            is_weekend = request.form.get('is_weekend')
            temperature = request.form.get('temperature')
            is_holiday = request.form.get('is_holiday')
            solar_generation = request.form.get('solar_generation')
            month = request.form.get('month')
            season = request.form.get('season')
            is_national_holiday = request.form.get('is_national_holiday')
            is_diwali_season = request.form.get('is_diwali_season')
            lag_1h = request.form.get('lag_1h')
            lag_24h = request.form.get('lag_24h')
            rolling_mean_3h = request.form.get('rolling_mean_3h')
            rolling_mean_24h = request.form.get('rolling_mean_24h')
            
            # Validate required fields
            required_fields = [timestamp, day_of_week, hour_of_day, is_weekend, temperature,
                             is_holiday, solar_generation, month, season, is_national_holiday,
                             is_diwali_season, lag_1h, lag_24h, rolling_mean_3h, rolling_mean_24h]
            
            if not all(required_fields):
                return render_template('home.html', 
                                     error="All fields are required. Please fill in all the information.")
            
            # Validate numeric fields
            try:
                day_of_week = int(day_of_week)
                hour_of_day = int(hour_of_day)
                is_weekend = int(is_weekend)
                temperature = float(temperature)
                is_holiday = int(is_holiday)
                solar_generation = float(solar_generation)
                month = int(month)
                is_national_holiday = int(is_national_holiday)
                is_diwali_season = int(is_diwali_season)
                lag_1h = float(lag_1h)
                lag_24h = float(lag_24h)
                rolling_mean_3h = float(rolling_mean_3h)
                rolling_mean_24h = float(rolling_mean_24h)
            except ValueError:
                return render_template('home.html', 
                                     error="Invalid numeric values. Please check your inputs.")
            
            # Create electricity data object
            data = ElectricityData(
                timestamp=timestamp,
                day_of_week=day_of_week,
                hour_of_day=hour_of_day,
                is_weekend=is_weekend,
                temperature=temperature,
                is_holiday=is_holiday,
                solar_generation=solar_generation,
                month=month,
                season=season,
                is_national_holiday=is_national_holiday,
                is_diwali_season=is_diwali_season,
                lag_1h=lag_1h,
                lag_24h=lag_24h,
                rolling_mean_3h=rolling_mean_3h,
                rolling_mean_24h=rolling_mean_24h
            )
            
            # Get data as DataFrame
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Electricity prediction data: {pred_df.to_dict()}")
            
            # Make prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # Format results for display
            consumption_kwh = round(results[0], 2)
            cost_estimate = round(consumption_kwh * 0.12, 2)  # Assuming $0.12 per kWh
            
            logging.info(f"Predicted electricity consumption: {consumption_kwh} kWh")
            
            return render_template('home.html', 
                                 results=consumption_kwh, 
                                 cost=cost_estimate,
                                 timestamp=timestamp)
            
    except ValueError as e:
        logging.error(f"Value error in electricity prediction: {str(e)}")
        return render_template('home.html', 
                             error="Invalid input. Please check your values are valid numbers.")
    except Exception as e:
        logging.error(f"Error in electricity prediction route: {str(e)}")
        return render_template('home.html', 
                             error="An error occurred while making the prediction. Please try again.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for electricity consumption prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract data from JSON
        electricity_data = ElectricityData(
            timestamp=data.get('timestamp'),
            day_of_week=int(data.get('day_of_week')),
            hour_of_day=int(data.get('hour_of_day')),
            is_weekend=int(data.get('is_weekend')),
            temperature=float(data.get('temperature')),
            is_holiday=int(data.get('is_holiday')),
            solar_generation=float(data.get('solar_generation')),
            month=int(data.get('month')),
            season=data.get('season'),
            is_national_holiday=int(data.get('is_national_holiday')),
            is_diwali_season=int(data.get('is_diwali_season')),
            lag_1h=float(data.get('lag_1h')),
            lag_24h=float(data.get('lag_24h')),
            rolling_mean_3h=float(data.get('rolling_mean_3h')),
            rolling_mean_24h=float(data.get('rolling_mean_24h'))
        )
        
        pred_df = electricity_data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        consumption_kwh = round(results[0], 2)
        cost_estimate = round(consumption_kwh * 0.12, 2)
        
        return jsonify({
            "predicted_consumption_kwh": consumption_kwh,
            "estimated_cost_usd": cost_estimate,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"API prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "PowerPlay Electricity Consumption Prediction API is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/stats')
def get_stats():
    """Get prediction statistics"""
    try:
        # This would typically come from a database
        stats = {
            "total_predictions": 1250,
            "average_consumption": 45.6,
            "peak_consumption": 89.2,
            "lowest_consumption": 12.3,
            "accuracy_rate": 94.2
        }
        return jsonify(stats)
    except Exception as e:
        logging.error(f"Error getting stats: {str(e)}")
        return jsonify({"error": "Could not retrieve statistics"}), 500

if __name__ == "__main__":
    logging.info("Starting PowerPlay Electricity Consumption Prediction Flask application")
    app.run(host="0.0.0.0", debug=True, port=5000)

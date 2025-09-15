import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        """Initialize the electricity consumption prediction pipeline"""
        pass

    def predict(self, features):
        """
        Make electricity consumption predictions using the trained model and preprocessor
        
        Args:
            features: DataFrame containing input features for electricity consumption
            
        Returns:
            predictions: Array of predicted electricity consumption values in kWh
        """
        try:
            # Define model and preprocessor paths
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Electricity model file not found at {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Electricity preprocessor file not found at {preprocessor_path}")
            
            logging.info(f"Loading electricity consumption model from {model_path}")
            model = load_object(file_path=model_path)
            
            logging.info(f"Loading electricity preprocessor from {preprocessor_path}")
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transform features using preprocessor
            logging.info("Transforming electricity consumption features using preprocessor")
            data_scaled = preprocessor.transform(features)
            
            # Make predictions
            logging.info("Making electricity consumption predictions")
            preds = model.predict(data_scaled)
            
            logging.info(f"Electricity consumption predictions completed. Shape: {preds.shape}")
            return preds
            
        except Exception as e:
            logging.error(f"Error in electricity consumption prediction pipeline: {str(e)}")
            raise CustomException(e, sys)

class ElectricityData:
    """
    Custom data class to handle input data for electricity consumption predictions
    """
    def __init__(self,
                 timestamp: str,
                 day_of_week: int,
                 hour_of_day: int,
                 is_weekend: int,
                 temperature: float,
                 is_holiday: int,
                 solar_generation: float,
                 month: int,
                 season: str,
                 is_national_holiday: int,
                 is_diwali_season: int,
                 lag_1h: float,
                 lag_24h: float,
                 rolling_mean_3h: float,
                 rolling_mean_24h: float):
        """
        Initialize ElectricityData with electricity consumption features
        
        Args:
            timestamp: Date and time in YYYY-MM-DD HH:MM:SS format
            day_of_week: Day of week (0-6, where 0=Monday)
            hour_of_day: Hour of day (0-23)
            is_weekend: Binary flag for weekend (0/1)
            temperature: Temperature in Celsius
            is_holiday: Binary flag for holiday (0/1)
            solar_generation: Solar power generation in kWh
            month: Month of year (1-12)
            season: Season (Winter1, Spring, Summer, Fall)
            is_national_holiday: Binary flag for national holiday (0/1)
            is_diwali_season: Binary flag for Diwali season (0/1)
            lag_1h: Electricity demand 1 hour ago
            lag_24h: Electricity demand 24 hours ago
            rolling_mean_3h: 3-hour rolling average of demand
            rolling_mean_24h: 24-hour rolling average of demand
        """
        self.timestamp = timestamp
        self.day_of_week = day_of_week
        self.hour_of_day = hour_of_day
        self.is_weekend = is_weekend
        self.temperature = temperature
        self.is_holiday = is_holiday
        self.solar_generation = solar_generation
        self.month = month
        self.season = season
        self.is_national_holiday = is_national_holiday
        self.is_diwali_season = is_diwali_season
        self.lag_1h = lag_1h
        self.lag_24h = lag_24h
        self.rolling_mean_3h = rolling_mean_3h
        self.rolling_mean_24h = rolling_mean_24h

    def get_data_as_data_frame(self):
        """
        Convert electricity data to DataFrame format for prediction
        
        Returns:
            DataFrame: Formatted data for electricity consumption model prediction
        """
        try:
            # Calculate engineered features
            temp_squared = self.temperature ** 2
            temp_weekend = self.temperature * self.is_weekend
            
            # Cyclical features for day of year and day of week
            # Assuming day of year from month (simplified)
            day_of_year = (self.month - 1) * 30 + 15  # Approximate day of year
            sin_day = np.sin(2 * np.pi * day_of_year / 365)
            cos_day = np.cos(2 * np.pi * day_of_year / 365)
            sin_week = np.sin(2 * np.pi * self.day_of_week / 7)
            cos_week = np.cos(2 * np.pi * self.day_of_week / 7)
            
            custom_data_input_dict = {
                "day_of_week": [self.day_of_week],
                "hour_of_day": [self.hour_of_day],
                "temperature": [self.temperature],
                "solar_generation": [self.solar_generation],
                "month": [self.month],
                "lag_1h": [self.lag_1h],
                "lag_24h": [self.lag_24h],
                "rolling_mean_3h": [self.rolling_mean_3h],
                "rolling_mean_24h": [self.rolling_mean_24h],
                "temp_squared": [temp_squared],
                "temp_weekend": [temp_weekend],
                "sin_day": [sin_day],
                "cos_day": [cos_day],
                "sin_week": [sin_week],
                "cos_week": [cos_week],
                "is_weekend": [self.is_weekend],
                "is_holiday": [self.is_holiday],
                "is_national_holiday": [self.is_national_holiday],
                "is_diwali_season": [self.is_diwali_season],
                "season": [self.season]
            }

            logging.info(f"Creating electricity consumption DataFrame with data: {custom_data_input_dict}")
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logging.error(f"Error creating electricity consumption DataFrame: {str(e)}")
            raise CustomException(e, sys)

    def get_consumption_category(self, predicted_consumption):
        """
        Categorize predicted consumption into efficiency levels
        
        Args:
            predicted_consumption: Predicted electricity consumption in kWh
            
        Returns:
            str: Consumption category
        """
        try:
            if predicted_consumption < 300:
                return "Very Low"
            elif predicted_consumption < 500:
                return "Low"
            elif predicted_consumption < 700:
                return "Moderate"
            elif predicted_consumption < 900:
                return "High"
            else:
                return "Very High"
                
        except Exception as e:
            logging.error(f"Error categorizing consumption: {str(e)}")
            return "Unknown"

    def get_peak_hour_analysis(self):
        """
        Analyze if current hour is peak consumption time
        
        Returns:
            dict: Peak hour analysis
        """
        try:
            is_peak_hour = 1 if 8 <= self.hour_of_day <= 20 else 0
            is_night = 1 if 22 <= self.hour_of_day or self.hour_of_day <= 6 else 0
            
            return {
                "is_peak_hour": is_peak_hour,
                "is_night": is_night,
                "hour_category": "Peak" if is_peak_hour else ("Night" if is_night else "Off-Peak")
            }
            
        except Exception as e:
            logging.error(f"Error in peak hour analysis: {str(e)}")
            return {"is_peak_hour": 0, "is_night": 0, "hour_category": "Unknown"}

    def get_weather_impact_score(self):
        """
        Calculate a weather impact score based on temperature and solar generation
        
        Returns:
            float: Weather impact score (0-100)
        """
        try:
            # Temperature impact (optimal around 20-25°C)
            temp_score = 100 - abs(self.temperature - 22.5) * 2
            temp_score = max(0, min(100, temp_score))
            
            # Solar generation impact (higher solar = lower demand)
            solar_score = min(100, self.solar_generation * 10)  # Scale solar generation
            
            # Combined weather impact
            weather_impact = (temp_score * 0.7 + solar_score * 0.3)
            
            return round(weather_impact, 2)
            
        except Exception as e:
            logging.error(f"Error calculating weather impact score: {str(e)}")
            return 50.0  # Default neutral score

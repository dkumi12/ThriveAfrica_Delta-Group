"""
Shared utility functions for ThriveAfrica Delta Group diabetes prediction.
This module consolidates feature engineering logic to eliminate code duplication.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import hashlib
import json
from typing import Dict, Any, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_derived_features(glucose: float, insulin: float, skin_thickness: float, 
                             bmi: float, age: int) -> Tuple[float, float, float, str]:
    """
    Calculate derived features for a single prediction.
    
    Args:
        glucose: Glucose level
        insulin: Insulin level  
        skin_thickness: Skin thickness measurement
        bmi: Body Mass Index
        age: Patient age
    
    Returns:
        Tuple of (glucose_insulin_ratio, skin_thickness_bmi_ratio, glucose_bmi, age_group)
    """
    try:        # Avoid division by zero
        glucose_insulin_ratio = glucose / max(insulin, 1e-6)
        glucose_insulin_ratio = min(glucose_insulin_ratio, 1e6)
        
        skin_thickness_bmi_ratio = skin_thickness / max(bmi, 1e-6)
        skin_thickness_bmi_ratio = min(skin_thickness_bmi_ratio, 1e6)
        
        glucose_bmi = np.log1p(min(glucose * bmi, 1e6))
        
        # Age group categorization
        if age < 25:
            age_group = "<25"
        elif age <= 40:
            age_group = "25-40"
        elif age <= 60:
            age_group = "40-60"
        else:
            age_group = ">60"
        
        return glucose_insulin_ratio, skin_thickness_bmi_ratio, glucose_bmi, age_group
        
    except Exception as e:
        logger.error(f"Error in feature calculation: {str(e)}")
        # Return safe defaults
        return 0.0, 0.0, 0.0, "25-40"


def validate_input_data(input_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate input data for predictions with comprehensive error checking.
    
    Args:
        input_dict: Dictionary of input features
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    try:
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in input_dict]
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        # Validate data types and ranges
        validations = [
            ('Pregnancies', lambda x: isinstance(x, (int, float)) and 0 <= x <= 20),
            ('Glucose', lambda x: isinstance(x, (int, float)) and 0 <= x <= 400),
            ('BloodPressure', lambda x: isinstance(x, (int, float)) and 0 <= x <= 200),
            ('SkinThickness', lambda x: isinstance(x, (int, float)) and 0 <= x <= 100),
            ('Insulin', lambda x: isinstance(x, (int, float)) and 0 <= x <= 1000),
            ('BMI', lambda x: isinstance(x, (int, float)) and 10 <= x <= 70),
            ('DiabetesPedigreeFunction', lambda x: isinstance(x, (int, float)) and 0 <= x <= 5),
            ('Age', lambda x: isinstance(x, (int, float)) and 1 <= x <= 120),
        ]
        
        for field, validator in validations:
            if not validator(input_dict[field]):
                return False, f"Invalid value for {field}: {input_dict[field]}"
        
        return True, "Valid input data"
        
    except Exception as e:
        logger.error(f"Error validating input data: {str(e)}")
        return False, f"Validation error: {str(e)}"
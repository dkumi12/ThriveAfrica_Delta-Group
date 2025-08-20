"""
Test suite for utility functions
"""
import pytest
import numpy as np
from src.utils import calculate_derived_features, validate_input_data

def test_calculate_derived_features():
    """Test feature calculation with normal inputs"""
    glucose, insulin, skin_thickness, bmi, age = 120, 30, 25, 25.0, 30
    
    ratio1, ratio2, glucose_bmi, age_group = calculate_derived_features(
        glucose, insulin, skin_thickness, bmi, age
    )
    
    assert ratio1 > 0  # Glucose/Insulin ratio should be positive
    assert ratio2 > 0  # SkinThickness/BMI ratio should be positive
    assert glucose_bmi > 0  # Glucose*BMI should be positive
    assert age_group == "25-40"  # Age 30 should be in 25-40 group

def test_validate_input_data_valid():
    """Test validation with valid input"""
    valid_data = {
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 80,
        'SkinThickness': 25,
        'Insulin': 30,
        'BMI': 25.0,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 30
    }
    
    is_valid, message = validate_input_data(valid_data)
    assert is_valid == True
    assert "Valid input data" in message

def test_validate_input_data_invalid():
    """Test validation with invalid input"""
    invalid_data = {
        'Pregnancies': 2,
        'Glucose': 500,  # Invalid: too high
        'BloodPressure': 80,
        'SkinThickness': 25,
        'Insulin': 30,
        'BMI': 25.0,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 30
    }
    
    is_valid, message = validate_input_data(invalid_data)
    assert is_valid == False
    assert "Invalid value" in message

def test_calculate_derived_features_edge_cases():
    """Test feature calculation with edge cases"""
    # Test with zero insulin (division by zero protection)
    glucose, insulin, skin_thickness, bmi, age = 120, 0, 25, 25.0, 30
    
    ratio1, ratio2, glucose_bmi, age_group = calculate_derived_features(
        glucose, insulin, skin_thickness, bmi, age
    )
    
    # Should handle division by zero gracefully
    assert ratio1 > 0
    assert not np.isinf(ratio1)
    assert not np.isnan(ratio1)
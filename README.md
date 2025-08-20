# 🏥 ThriveAfrica Delta Group - Diabetes Prediction System

[![CI/CD Pipeline](https://github.com/dkumi12/ThriveAfrica_Delta-Group/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/dkumi12/ThriveAfrica_Delta-Group/actions/workflows/ci-cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **AI-powered diabetes risk assessment platform with professional engineering practices and comprehensive validation.**

## 🚀 Features

### Core Functionality
- **🤖 ML-Powered Predictions**: Logistic regression model for diabetes risk assessment
- **📊 Feature Engineering**: Automated derivation of clinical risk indicators  
- **🎯 Validated Model**: Professional ML pipeline with preprocessing
- **💻 Web Interface**: User-friendly Streamlit application

### Professional Engineering
- **✅ Shared Utilities**: Eliminated code duplication with reusable functions
- **✅ Input Validation**: Comprehensive data sanitization and validation
- **🔄 CI/CD Pipeline**: Automated testing, linting, and security scanning
- **📈 Test Coverage**: Comprehensive test suite with pytest
- **🏗️ Clean Architecture**: Organized codebase with separation of concerns

### Security & Compliance
- **🔒 Error Handling**: Robust error management throughout application
- **📋 Audit Logging**: Basic prediction tracking for compliance
- **🚫 No Data Persistence**: Patient data not stored permanently
- **🛡️ Security Scanning**: Automated vulnerability detection

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/dkumi12/ThriveAfrica_Delta-Group.git
cd ThriveAfrica_Delta-Group

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```
## 📖 Usage

### Web Application

1. **Start the application**: 
   ```bash
   streamlit run app.py
   ```

2. **Access the interface**: Open http://localhost:8501 in your browser

3. **Enter patient data**: Use the sliders to input health metrics:
   - Pregnancies, Glucose, Blood Pressure, Skin Thickness
   - Insulin, BMI, Diabetes Pedigree Function, Age

4. **Get assessment**: Click "Assess Diabetes Risk" for AI-powered analysis

5. **Review results**: View risk level, probability, and recommendations

### Model Information

- **Algorithm**: Logistic Regression
- **Features**: 8 original + 3 engineered features
- **Performance**: Validated on diabetes dataset
- **Preprocessing**: Standardized features using scikit-learn

## 🏗️ Project Structure

```
ThriveAfrica_Delta-Group/
├── app.py                    # Main Streamlit application (enhanced)
├── app_original.py          # Original application (backup)
├── logistic_regression_model.pkl  # Trained ML model
├── preprocessor.pkl         # Feature preprocessing pipeline
├── diabetes.csv            # Training dataset
├── src/                    # Source code modules
│   └── utils.py           # Shared utilities and validation
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_utils.py      # Unit tests for utilities
├── .github/workflows/      # CI/CD automation
│   └── ci-cd.yml         # GitHub Actions pipeline
├── logs/                   # Application logging
├── requirements.txt        # Python dependencies
├── LICENSE                # MIT License
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🧪 Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run test suite
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Code Quality
```bash
# Install linting tools
pip install flake8 black isort

# Lint code
flake8 src/ tests/

# Format code
black src/ tests/

# Sort imports
isort src/ tests/
```

## 📊 Model Performance

- **Training Data**: Pima Indians Diabetes Database
- **Model Type**: Logistic Regression with feature engineering
- **Features Used**: 
  - Original: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age
  - Engineered: Glucose_Insulin_Ratio, SkinThickness_BMI_Ratio, Glucose_BMI, Age_Group
- **Preprocessing**: StandardScaler normalization
## 🔧 Technical Improvements

### What's New in This Version

✅ **Eliminated Code Duplication**: Moved feature engineering to shared `src/utils.py`  
✅ **Enhanced Error Handling**: Comprehensive exception handling throughout  
✅ **Input Validation**: Robust validation of all user inputs  
✅ **Professional Structure**: Organized codebase with proper separation  
✅ **CI/CD Pipeline**: Automated testing and security scanning  
✅ **Test Coverage**: Unit tests for all utility functions  
✅ **Security Scanning**: Bandit and Safety integration  
✅ **Documentation**: Professional README and code documentation  

### Security Features

- **Input Sanitization**: All user inputs validated for type and range
- **Error Handling**: Graceful handling of model loading and prediction errors  
- **Audit Trail**: Basic prediction logging with unique identifiers
- **No Data Persistence**: Patient information never stored permanently
- **Secure Defaults**: Safe fallbacks for edge cases and errors

## 🚀 Deployment

### Local Development
```bash
# Run locally with live reload
streamlit run app.py --server.runOnSave true
```

### Production Considerations
- Use environment variables for sensitive configuration
- Implement proper logging in production environment
- Consider containerization with Docker
- Set up monitoring and health checks
- Implement proper authentication if needed

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure CI/CD pipeline passes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Pima Indians Diabetes Database
- **Framework**: Streamlit for web interface
- **ML Library**: Scikit-learn for model implementation
- **Team**: ThriveAfrica Delta Group

## 📞 Support

For questions or issues:
- Open an [issue](https://github.com/dkumi12/ThriveAfrica_Delta-Group/issues)
- Contact: ThriveAfrica Delta Group

---

**Developed with ❤️ by ThriveAfrica Delta Group**
#!/bin/bash

# Secure Diabetes Prediction System - Setup Script
# ================================================

echo "🏥 Setting up Secure Diabetes Prediction System..."

# Check Python version
python_version=$(python --version 2>&1)
echo "📋 Detected: $python_version"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p models
mkdir -p visualizations
mkdir -p docs
mkdir -p tests

# Train the model
echo "🧠 Training the diabetes prediction model..."
cd src
python train_model.py
cd ..

echo "✅ Setup complete!"
echo ""
echo "🚀 To run the application:"
echo "   streamlit run src/app.py"
echo ""
echo "🔒 Security features enabled:"
echo "   ✅ Input validation"
echo "   ✅ Zero data persistence"
echo "   ✅ Secure model serving"
echo ""
echo "⚠️  Remember: This is for educational purposes only."
echo "   Always consult healthcare professionals for medical advice."

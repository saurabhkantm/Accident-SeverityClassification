# 🚗 Accident Severity Classification System

A professional Streamlit-based web application for classifying accident severity levels using Deep Learning.

---

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Integration](#model-integration)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

- 🖼 *Image Upload*: Support for JPG, PNG, JPEG formats
- 🤖 *AI Classification*: 3-class severity prediction (Minor, Moderate, Severe)
- 📊 *Confidence Score*: Visual confidence indicators with progress bars
- 💡 *Smart Recommendations*: Action items based on severity level
- 📱 *Responsive Design*: Works on desktop and mobile devices
- 🎨 *Modern UI*: Clean interface with color-coded results
- ⚡ *Fast Processing*: Real-time image analysis
- 📈 *Statistics Dashboard*: Model performance metrics

---

## 📁 Project Structure


accident_severity_app/
│
├── app.py                 # Main Streamlit application
├── model.py               # ML model logic and predictions
├── utils.py               # Image preprocessing utilities
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
└── models/               # (Create this folder)
    └── accident_model.h5  # Trained model file (add later)


---

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download Project
bash
mkdir accident_severity_app
cd accident_severity_app


### Step 2: Create Virtual Environment (Optional but Recommended)
bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate


### Step 3: Install Dependencies
bash
pip install -r requirements.txt


### Step 4: Verify Installation
bash
streamlit --version


---

## 🚀 Usage

### Run the Application
bash
streamlit run app.py


The app will automatically open in your default browser at:

http://localhost:8501


### Using the App
1. *Upload Image*: Click "Browse files" and select an accident photo
2. *Wait for Analysis*: The model processes the image (2-3 seconds)
3. *View Results*: See severity classification and confidence score
4. *Follow Recommendations*: Read suggested actions based on severity

---

## 🧠 Model Integration

Currently, the app uses a *dummy model* for demonstration. To integrate your trained model:

### Step 1: Train Your Model
python
# Example training code (separate script)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# Build and train your model
model = Sequential([...])
model.compile(...)
model.fit(X_train, y_train, epochs=50)

# Save the model
model.save('models/accident_model.h5')


### Step 2: Update model.py
Replace the dummy prediction code:

python
# In model.py
from tensorflow.keras.models import load_model

def load_model():
    model = load_model('models/accident_model.h5')
    return model

MODEL = load_model()

def predict_severity(image_array):
    predictions = MODEL.predict(image_array)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx] * 100)
    severity_class = SEVERITY_CLASSES[class_idx]
    return severity_class, confidence


### Step 3: Update Class Names
Modify SEVERITY_CLASSES in model.py to match your training labels:

python
SEVERITY_CLASSES = [
    "🟢 Minor Damage",
    "🟡 Moderate Damage",
    "🔴 Severe Crash"
]


---

## 📸 Screenshots

### Main Interface

[Upload section with drag-and-drop area]


### Results Display

┌─────────────────────────────────────┐
│ Uploaded Image    │  Analysis Results│
│                   │  Severity: Minor │
│   [Car Image]     │  Confidence: 92% │
│                   │  [Progress Bar]  │
└─────────────────────────────────────┘


---

## 🔮 Future Enhancements

### Phase 1: Core Features
- [ ] Integrate real trained ML model
- [ ] Add batch processing for multiple images
- [ ] Implement image augmentation for better accuracy
- [ ] Add model performance metrics dashboard

### Phase 2: Advanced Features
- [ ] *Grad-CAM Visualization*: Show heatmaps of model attention
- [ ] *Real-time Camera Input*: Capture images directly
- [ ] *Export Reports*: Generate PDF reports with analysis
- [ ] *Multi-model Comparison*: Compare different model architectures

### Phase 3: Production Ready
- [ ] *REST API*: Flask/FastAPI backend
- [ ] *Database Integration*: Store predictions and user data
- [ ] *User Authentication*: Secure login system
- [ ] *Cloud Deployment*: AWS/GCP/Azure hosting
- [ ] *Docker Container*: Containerized deployment
- [ ] *CI/CD Pipeline*: Automated testing and deployment

---

## 🛠 Troubleshooting

### Common Issues

*Issue*: Module not found error
bash
# Solution
pip install --upgrade -r requirements.txt


*Issue*: Port already in use
bash
# Solution: Run on different port
streamlit run app.py --server.port 8502


*Issue*: Image upload fails
bash
# Solution: Check image size (max 10MB)
# Supported formats: JPG, PNG, JPEG


---

## 📚 Documentation

### Code Structure

*app.py* - Frontend application
- Handles user interface
- File upload management
- Results display

*model.py* - ML model logic
- Model loading
- Prediction generation
- Confidence calculation

*utils.py* - Helper functions
- Image preprocessing
- Validation checks
- Metadata extraction

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

*Gaurav*
- Project: Accident Severity Classification System
- Technology Stack: Python, Streamlit, TensorFlow, Pillow
- Year: 2024

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com
- Documentation: [Link to detailed docs]

---

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- TensorFlow community for ML resources
- Open-source contributors

---

*⭐ If you find this project helpful, please give it a star!*

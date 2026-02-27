# ASL Recognition

Real-time American Sign Language (ASL) alphabet recognition using deep learning and MediaPipe hand tracking.

## Overview

This project uses a 1D CNN with Residual blocks to recognize ASL alphabet gestures (A-Z) from hand landmarks extracted via MediaPipe. The model processes 21 hand joint coordinates (x, y, z) in real-time from webcam feed.

## Features

- **Real-time hand tracking** using MediaPipe
- **Deep learning model** (ResNet-style 1D CNN) for gesture classification
- **29 classes**: A-Z, space, delete, and nothing
- **Stability buffer** - requires 10 consecutive same predictions to confirm a letter
- **Time-based confirmation** - hold a sign for 2 seconds to confirm it
- **Word building** with backspace support
- **Visual feedback** with hand landmark overlay and confidence display

## Project Structure

```
ASL-recognition/
├── app.py                      # Main real-time recognition application
├── asl_recognition.ipynb      # Jupyter notebook for model training
├── asl_resnet_model.keras     # Trained model (29 classes) - NOT included in git
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ASL-recognition.git
   cd ASL-recognition
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Model**
   
   Train your own model (see Training section below) 

## Usage

Run the real-time recognition:

```bash
python app.py
```

### Controls

| Key | Action |
|-----|--------|
| Hold sign for 2 seconds | Confirm a letter |
| Sign 'del' | Delete last character |
| Sign 'space' | Add a space |
| Press `q` | Quit the application |

### How It Works

1. **Hand Detection**: MediaPipe detects 21 hand landmarks (wrists, knuckles, fingertips)
2. **Preprocessing**: 
   - Landmarks are centered at the wrist
   - Scaled by maximum 3D distance for size invariance
3. **Prediction**: The CNN model classifies the normalized landmarks
4. **Confirmation**: 
   - A stability buffer requires 10 consecutive same predictions
   - A 2-second hold timer confirms the letter
5. **Display**: The recognized word is shown on screen with confidence score

## Training Your Own Model

To retrain the model:

1. Open `asl_recognition.ipynb` in Google Colab
2. Attach the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle
3. Run through the notebook cells:
   - Phase 1: Environment setup
   - Phase 2: Feature extraction (images → landmarks)
   - Phase 3: Data preprocessing and model training
4. Download the trained model and replace `asl_resnet_model.keras`

## Requirements

- Python 3.8+
- Webcam
- TensorFlow 2.10+
- MediaPipe
- OpenCV
- NumPy

See `requirements.txt` for exact versions.

## License

This project is for educational purposes.

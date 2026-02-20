# Real-Time ASL Recognition with Stability Buffer and Word Reconstruction

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from collections import Counter
from tensorflow import keras

# Load model
model = keras.models.load_model("asl_resnet_model.keras")
print("✔ Model loaded")

# Initialize TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)


class StabilityBuffer:
    """Keeps track of last N predictions, confirms when one dominates."""
    
    def __init__(self, buffer_size=10, confirm_threshold=10):
        self.buffer_size = buffer_size
        self.confirm_threshold = confirm_threshold
        self.buffer = []
    
    def add(self, prediction):
        """Add a prediction (class index) to buffer."""
        self.buffer.append(prediction)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def get_confirmed(self):
        """Return confirmed character if threshold met, else None."""
        if len(self.buffer) < self.confirm_threshold:
            return None
        
        counts = Counter(self.buffer)
        most_common = counts.most_common(1)[0]
        
        if most_common[1] >= self.confirm_threshold:
            return most_common[0]
        return None
    
    def reset(self):
        self.buffer = []


def process_frame(frame, hands, model, buffer, label_map):
    """Process single frame and return display info."""
    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    current_word = ""
    confidence = 0.0
    hand_detected = True
    
    if results.multi_hand_landmarks:
        # Extract landmarks (21 joints x 3 coords)
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_values = []
        for landmark in hand_landmarks.landmark:
            landmark_values.extend([landmark.x, landmark.y, landmark.z])
        
        # Convert to numpy array (21, 3)
        landmarks_np = np.array(landmark_values, dtype=np.float32).reshape(21, 3)
        
        # Wrist centering: subtract wrist (landmark 0) from all points
        wrist = landmarks_np[0]
        centered = landmarks_np - wrist
        
        # 3D scaling: divide by max 3D Euclidean distance
        distances = np.linalg.norm(centered[1:], axis=1)
        max_distance = np.max(distances) + 1e-6  # epsilon to prevent zero-div
        
        # Normalize
        landmarks_normalized = centered / max_distance
        
        # Reshape to (1, 21, 3) for model input
        landmarks_input = landmarks_normalized.reshape(1, 21, 3)
        
        # Predict
        pred = model.predict(landmarks_input, verbose=0)[0]
        pred_class = np.argmax(pred)
        confidence = float(np.max(pred))
        
        # Add to stability buffer
        buffer.add(pred_class)
        confirmed = buffer.get_confirmed()
        
        if confirmed is not None and confirmed < len(label_map):
            char = label_map[confirmed]
            if char == 'space':
                char = ' '
            current_word = char
    
    else:
        hand_detected = False
    
    return current_word, confidence, hand_detected


def run_asl_recognition():
    """Main loop for real-time ASL recognition."""
    
    # Label map (29 classes)
    label_map = {i: chr(ord('A') + i) for i in range(26)}
    label_map[26] = 'del'
    label_map[27] = 'nothing'
    label_map[28] = 'space'
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    
    # Initialize buffer and word
    buffer = StabilityBuffer(buffer_size=10, confirm_threshold=10)
    current_word = ""
    last_hand_time = cv2.getTickCount()
    
    cap = cv2.VideoCapture(0)
    
    print("Starting ASL recognition... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror
        
        # Process frame
        char, confidence, hand_detected = process_frame(
            frame, hands, model, buffer, label_map
        )
        
        # Update word
        if char and char != current_word:
            if char == ' ' and current_word and current_word[-1] != ' ':
                current_word += char
            elif char != ' ':
                current_word += char
        
        # Track hand detection for space/clear
        if hand_detected:
            last_hand_time = cv2.getTickCount()
        else:
            # No hand for 2 seconds = word finished
            elapsed = (cv2.getTickCount() - last_hand_time) / cv2.getTickFrequency()
            if elapsed > 2.0 and current_word.strip():
                print(f"Word: {current_word.strip()}")
                tts_engine.say(current_word.strip())
                tts_engine.runAndWait()
                current_word = ""
                buffer.reset()
        
        # Display on frame
        cv2.putText(frame, f"Word: {current_word}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("ASL Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_asl_recognition()

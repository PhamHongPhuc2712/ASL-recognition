# Real-Time ASL Recognition with Stability Buffer and Word Reconstruction

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import Counter
from tensorflow import keras

# Load model
model = keras.models.load_model("asl_resnet_model.keras")
print("✔ Model loaded")

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles



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
    
    pending_char = ""  # Character waiting for time-based confirmation
    confidence = 0.0
    hand_detected = True
    hand_landmarks = None  # For visualization
    
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
            pending_char = char  # This is now pending confirmation by time
    
    else:
        hand_detected = False
    
    return pending_char, confidence, hand_detected, hand_landmarks


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
    
    # Time-based letter confirmation
    CHAR_CONFIRM_INTERVAL = 2.0  # Seconds to hold sign before confirming
    pending_char = None  # Current character waiting to be confirmed
    
    cap = cv2.VideoCapture(0)
    
    print("Starting ASL recognition... Press 'q' to quit.")

    last_pending_char = None
    char_start_time = None

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror
        
        # Process frame
        pending_char, confidence, hand_detected, hand_landmarks = process_frame(
            frame, hands, model, buffer, label_map
        )
        
        # Draw hand landmarks visualization
        if hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Update word with time-based confirmation
        current_time = time.time()

        if pending_char and pending_char not in ('nothing',):
            if pending_char != last_pending_char:
                # Sign changed - restart timer
                char_start_time = current_time
                last_pending_char = pending_char
            elif current_time - char_start_time > CHAR_CONFIRM_INTERVAL:
                # 2 seconds held - confirm it
                if pending_char == ' ':
                    if current_word and current_word[-1] != ' ':
                        current_word += pending_char
                elif pending_char == 'del':
                    current_word = current_word[:-1]
                else:
                    current_word += pending_char
                
                # Force user to briefly change sign before next letter
                char_start_time = current_time  # reset timer
                last_pending_char = None        # ← forget last char, so next L restarts cleanly
                buffer.reset()
        
        # Display on frame
        cv2.putText(frame, f"Word: {current_word}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Show confirmation timer if a letter is pending
        # if pending_char and last_letter_confirmed_time is not None:
        #     elapsed = current_time - last_letter_confirmed_time
        #     remaining = max(0, CHAR_CONFIRM_INTERVAL - elapsed)
        if pending_char and pending_char != 'nothing' and char_start_time is not None:
            elapsed = current_time - char_start_time
            remaining = max(0, CHAR_CONFIRM_INTERVAL - elapsed)
            color = (0, 255, 0) if remaining == 0 else (0, 165, 255)
            cv2.putText(frame, f"Confirming '{pending_char}' in: {remaining:.1f}s", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("ASL Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_asl_recognition()

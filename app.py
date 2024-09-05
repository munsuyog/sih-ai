import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Configuration
IMG_SIZE = 224
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    return frame_resized / 255.0

def process_video_sequential(model, label_encoder, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    predictions = []
    frame_count = 0
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            frame_processed = preprocess_frame(frame)
            frame_buffer.append(frame_processed)
            
            if len(frame_buffer) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(frame_buffer, axis=0)
                prediction = model.predict(input_data)
                predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
                confidence = np.max(prediction)
                
                predictions.append((predicted_class[0], confidence, frame_count - SEQUENCE_LENGTH + 1, frame_count))
                
                cv2.putText(frame, f"Sign: {predicted_class[0]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                frame_buffer = []  # Clear buffer for next sequence
            
            cv2.imshow('Indian Sign Language Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return predictions

def post_process_predictions(predictions, threshold=0.5, min_sequence_length=10):
    processed_predictions = []
    current_sign = None
    start_frame = None
    for sign, confidence, start, end in predictions:
        if confidence >= threshold:
            if current_sign is None:
                current_sign = sign
                start_frame = start
            elif sign != current_sign:
                if end - start_frame >= min_sequence_length:
                    processed_predictions.append((current_sign, start_frame, end - 1))
                current_sign = sign
                start_frame = start
        elif current_sign is not None:
            if end - start_frame >= min_sequence_length:
                processed_predictions.append((current_sign, start_frame, end - 1))
            current_sign = None
            start_frame = None
    
    if current_sign is not None and end - start_frame >= min_sequence_length:
        processed_predictions.append((current_sign, start_frame, end))
    
    return processed_predictions

if __name__ == "__main__":
    try:
        # Load pre-trained model
        model = load_model('isl_model.h5')
        label_encoder_classes = np.load('label_encoder.npy', allow_pickle=True)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_encoder_classes
        
        print("Starting Indian Sign Language detection on input video...")
        print(f"Available signs: {label_encoder.classes_}")
        
        input_video_path = 'input_video.mp4'  # Update this to your input video path
        predictions = process_video_sequential(model, label_encoder, input_video_path)
        
        processed_predictions = post_process_predictions(predictions)
        
        print("\nDetected Signs:")
        for sign, start, end in processed_predictions:
            print(f"Sign: {sign}, Frames: {start} to {end}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
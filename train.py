import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuration
IMG_SIZE = 224  # MobileNetV2 input size
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    return frame_resized / 255.0

def extract_hand_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened() and len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            frame_processed = preprocess_frame(frame)
            frames.append(frame_processed)
    
    cap.release()
    
    if len(frames) < SEQUENCE_LENGTH:
        return None
    
    return np.array(frames[:SEQUENCE_LENGTH])

def load_data(data_dir):
    X = []
    y = []
    for word in os.listdir(data_dir):
        word_dir = os.path.join(data_dir, word)
        if os.path.isdir(word_dir):
            for video_file in os.listdir(word_dir):
                if video_file.endswith('.MOV'):
                    video_path = os.path.join(word_dir, video_file)
                    frames = extract_hand_frames(video_path)
                    if frames is not None:
                        X.append(frames)
                        y.append(word)
    return np.array(X), np.array(y)

def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.TimeDistributed(base_model)(inputs)
    x = tf.keras.layers.TimeDistributed(GlobalAveragePooling2D())(x)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(data_dir):
    X, y = load_data(data_dir)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    model = create_model(len(label_encoder.classes_))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")
    
    model.save('isl_model.h5')
    np.save('label_encoder.npy', label_encoder.classes_)
    
    return model, label_encoder, history

def process_video(model, label_encoder, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    predictions = []
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
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
                
                predictions.append((predicted_class[0], confidence))
                
                cv2.putText(frame, f"Sign: {predicted_class[0]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                frame_buffer.pop(0)
            
            cv2.imshow('Indian Sign Language Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return predictions

if __name__ == "__main__":
    try:
        data_dir = 'data'  # Update this to the path of your data directory
        
        # Uncomment these lines if you want to train a new model
        model, label_encoder, history = train_model(data_dir)
        
        # Load a pre-trained model
        # model = tf.keras.models.load_model('isl_model.h5')
        # label_encoder_classes = np.load('label_encoder.npy', allow_pickle=True)
        # label_encoder = LabelEncoder()
        # label_encoder.classes_ = label_encoder_classes
        
        print("Starting Indian Sign Language detection on input video...")
        print(f"Available signs: {label_encoder.classes_}")
        
        input_video_path = 'path/to/your/input_video.mp4'  # Update this to your input video path
        predictions = process_video(model, label_encoder, input_video_path)
        
        print("\nPredictions:")
        for i, (sign, confidence) in enumerate(predictions):
            print(f"Frame {i*SEQUENCE_LENGTH}: Sign: {sign}, Confidence: {confidence:.2f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
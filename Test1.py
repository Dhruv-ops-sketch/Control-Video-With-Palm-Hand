import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Variables to store previous palm positions
prev_xs = []
window_size = 5  # Size of the smoothing window
threshold = 30  # Movement threshold
last_move_time = time.time()
move_cooldown = 0.5  # seconds
last_gesture_time = time.time()
gesture_cooldown = 1.5  # seconds

def smooth_positions(positions, window_size):
    if len(positions) < window_size:
        return np.mean(positions)
    else:
        return np.mean(positions[-window_size:])

def is_fist(hand_landmarks):
    """Detect if the hand is a fist"""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    thumb_ip = hand_landmarks.landmark[2]
    wrist = hand_landmarks.landmark[0]

    # Check if the tips of all fingers are close to the palm
    return all([
        thumb_tip.y > thumb_ip.y,
        index_tip.y > wrist.y,
        middle_tip.y > wrist.y,
        ring_tip.y > wrist.y,
        pinky_tip.y > wrist.y,
    ])

def is_open_hand(hand_landmarks):
    """Detect if the hand is open"""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # Check if the tips of all fingers are away from the palm
    return all([
        thumb_tip.y < wrist.y,
        index_tip.y < wrist.y,
        middle_tip.y < wrist.y,
        ring_tip.y < wrist.y,
        pinky_tip.y < wrist.y,
    ])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    result = hands.process(rgb_frame)
    
    current_time = time.time()
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the x-coordinate of the wrist (landmark 0)
            wrist_x = hand_landmarks.landmark[0].x * frame.shape[1]
            prev_xs.append(wrist_x)
            smooth_x = smooth_positions(prev_xs, window_size)
            
            if len(prev_xs) > 1:
                dx = smooth_x - smooth_positions(prev_xs[:-1], window_size)
                if current_time - last_move_time > move_cooldown:
                    if dx > threshold:
                        # Move the video forward (simulate right arrow key press)
                        pyautogui.press('right')
                        last_move_time = current_time
                    elif dx < -threshold:
                        # Move the video backward (simulate left arrow key press)
                        pyautogui.press('left')
                        last_move_time = current_time
            
            # Detect stop/play gesture
            if current_time - last_gesture_time > gesture_cooldown:
                if is_fist(hand_landmarks):
                    print("Fist detected: Pausing/Playing video")
                    pyautogui.press('space')  # Pause/Play toggle
                    last_gesture_time = current_time
                elif is_open_hand(hand_landmarks):
                    print("Open hand detected: Ignored")
    
    # Display the image
    cv2.imshow('Palm Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
    
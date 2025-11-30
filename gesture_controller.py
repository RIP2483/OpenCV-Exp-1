import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.7,
                       static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

# --- Constants and State Management ---
TIP_OF_THUMB = 4
TIP_OF_INDEX_FINGER = 8
TIP_OF_MIDDLE_FINGER = 12
TIP_OF_RING_FINGER = 16
TIP_OF_PINKY = 20
# will add more later

# Action cooldowns
last_action_time = 0
action_cooldown_ms = 500 # milliseconds
CONTINUOUS_ACTION_COOLDOWN_MS = 150 # milliseconds

# for determining whether the pause gesture is active
is_paused_gesture_active = False

# System Keys
PAUSE_KEY = 'space'
VOLUME_UP_KEY = 'volumeup'
VOLUME_DOWN_KEY = 'volumedown'

# --- Helper Functions ---
def calculate_distance(p1, p2):
    """
    Calculates the Euclidean distance between two 3D points (landmarks).
    The coordinates (x, y, z) are normalized (0 to 1), making the distance 
    gesture-independent of screen size.
    """
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2) 

def is_finger_extended(hand_landmarks, tip_id, pip_id):
    """
    Determines if a finger is extended based on the positions of its tip and pip landmarks.
    basically if the tip's y coordinate is less than the pip's y coordinate the finger is considered extended.
    """
    tip = hand_landmarks.landmark[tip_id]
    pip = hand_landmarks.landmark[pip_id]
    return tip.y < pip.y  # Assuming y=0 is at the top of the image

def get_landmark_coords(hand_landmarks, landmark_id, width, height):
    """
    Converts normalized landmark coordinates (0-1) to pixel coordinates.
    This is necessary for logic that depends on the hand's position relative to the screen, 
    like determining if the hand is in the 'UP' or 'DOWN' zone for volume control.
    """
    lm = hand_landmarks.landmark[landmark_id]
    x = int(lm.x * width)
    y = int(lm.y * height)
    return x, y, lm.z # Returns pixel X, pixel Y, and the normalized Z depth

def is_all_fingers_extended(hand_landmarks):
    """
    Checks if all fingers are extended and by doing so also determining the open palm gesture.
    """
    index_extended = is_finger_extended(hand_landmarks, TIP_OF_INDEX_FINGER, TIP_OF_INDEX_FINGER - 2)
    middle_extended = is_finger_extended(hand_landmarks, TIP_OF_MIDDLE_FINGER, TIP_OF_MIDDLE_FINGER - 2)
    ring_extended = is_finger_extended(hand_landmarks, TIP_OF_RING_FINGER, TIP_OF_RING_FINGER - 2)
    pinky_extended = is_finger_extended(hand_landmarks, TIP_OF_PINKY, TIP_OF_PINKY - 2)
    # Simple check for thumb extension (tip is outside of the PIP joint's shadow)
    thumb_extended = hand_landmarks.landmark[TIP_OF_THUMB].x < hand_landmarks.landmark[TIP_OF_THUMB - 1].x # assumes hand is facing palm-to-camera

    return index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended

def handle_gestures(frame, hand_landmarks, width, height, current_time):
    """
    Main function for gesture recognition and system output.
    This is the core business logic of the application and also the most confusing lol.
    """
    global last_action_time, PAUSE_KEY, VOLUME_UP_KEY, VOLUME_DOWN_KEY, action_cooldown_ms, CONTINUOUS_ACTION_COOLDOWN_MS, is_paused_gesture_active
    
    # this detects the pinch
    index_tip = hand_landmarks.landmark[TIP_OF_INDEX_FINGER]
    thumb_tip = hand_landmarks.landmark[TIP_OF_THUMB]
    dist = calculate_distance(index_tip, thumb_tip)

    pinch_threshold = 0.05 # a set distnace to consider a pinch gesture
    
    if dist < pinch_threshold:
        # Pinch is active. Check Y-position to determine volume UP or DOWN.
        
        # Only allow volume commands every 150ms for smoother, continuous control
        if current_time - last_action_time > CONTINUOUS_ACTION_COOLDOWN_MS: 
            
            # Get the Index Finger's pixel Y position
            index_x, index_y, _ = get_landmark_coords(hand_landmarks, TIP_OF_INDEX_FINGER, width, height)
            
            # Divide the screen into vertical zones based on the frame height
            screen_mid_y = height / 2

            if index_y < screen_mid_y - 50:
                # Hand is in the top zone (Higher volume)
                pyautogui.press(VOLUME_UP_KEY)
                cv2.putText(frame, "Volume Up (Pinch High)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif index_y > screen_mid_y + 50:
                # Hand is in the bottom zone (Lower volume)
                pyautogui.press(VOLUME_DOWN_KEY)
                cv2.putText(frame, "Volume Down (Pinch Low)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            last_action_time = current_time
        is_paused_gesture_active = False

    # Detect Open Palm (Discrete Play/Pause)
    else:
        palm_open = is_all_fingers_extended(hand_landmarks)
        
        if palm_open and not is_paused_gesture_active:
            # Gesture is active AND it was NOT active in the previous cycle.
            
            # Check cooldown only when the gesture first appears
            if current_time - last_action_time > action_cooldown_ms:
                pyautogui.press(PAUSE_KEY)
                cv2.putText(frame, "Play/Pause Toggle (Open Palm)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                last_action_time = current_time
                print(f"Action triggered at {time.strftime('%H:%M:%S')}: Play/Pause Toggled")
            
            # Set flag to True to prevent re-triggering until hand is closed
            is_paused_gesture_active = True 
            
        elif not palm_open and is_paused_gesture_active:
            # RESET CONDITION: Gesture is inactive AND it was previously active.
            # This is the "release" event, preparing the system for the next Play/Pause command.
            is_paused_gesture_active = False

# --- Main Loop ---
cap = cv2.VideoCapture(0) # 0 means to use the defaulkt webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open video stream. Check camera ID or if another app is using it.")
    exit()

try:
    while cap.isOpened():
        success, frame = cap.read() # Read a new frame from the camera
        if not success:
            continue
        
        # Pre-processing: Flip the frame and convert color space
        frame = cv2.flip(frame, 1) # Mirror image for intuitive control
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe requires RGB input
        
        # Inference: Pass the frame to the MediaPipe model where the main hand tracking happens
        results = hands.process(rgb_frame)
        
        current_time = int(round(time.time() * 1000)) # Get current time for cooldown checks
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # this draws the tracking overlay
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                # Gesture Logic: Run the recognition and command execution
                handle_gestures(frame, hand_landmarks, width, height, current_time)

        # Show the processed frame
        cv2.imshow('Python Hand Gesture Controller', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

print("Application stopped.")
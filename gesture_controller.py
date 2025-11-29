import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

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
action_cooldown = 500 # milliseconds
CONTINUOUS_ACTION_COOLDOWN = 150 # milliseconds

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
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
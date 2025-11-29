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

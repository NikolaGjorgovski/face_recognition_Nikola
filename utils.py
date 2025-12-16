"""
Utility functions for face recognition operations.
"""

import os
import pickle
import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Optional, Dict
from config import ENCODINGS_FILE, TOLERANCE, CONFIDENCE_THRESHOLD


def load_encodings() -> Tuple[List[np.ndarray], List[str]]:
    """
    Load face encodings and names from the pickle file.
    
    Returns:
        Tuple of (encodings list, names list)
    """
    if not os.path.exists(ENCODINGS_FILE):
        return [], []
    
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            encodings = data.get('encodings', [])
            names = data.get('names', [])
        print(f"Loaded {len(encodings)} face encodings from {ENCODINGS_FILE}")
        return encodings, names
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return [], []


def save_encodings(encodings: List[np.ndarray], names: List[str]) -> bool:
    """
    Save face encodings and names to a pickle file.
    
    Args:
        encodings: List of face encodings
        names: List of corresponding names
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(ENCODINGS_FILE), exist_ok=True)
        
        data = {
            'encodings': encodings,
            'names': names
        }
        
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(encodings)} face encodings to {ENCODINGS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving encodings: {e}")
        return False


def encode_face(image_path: str) -> Optional[np.ndarray]:
    """
    Generate face encoding from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Face encoding array or None if no face detected
    """
    try:
        # Load image using face_recognition (handles RGB conversion)
        image = face_recognition.load_image_file(image_path)
        
        # Find face locations
        face_locations = face_recognition.face_locations(image, model='hog')
        
        if len(face_locations) == 0:
            print(f"No face detected in {image_path}")
            return None
        
        if len(face_locations) > 1:
            print(f"Warning: Multiple faces detected in {image_path}. Using the first one.")
        
        # Generate encoding for the first face
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        return face_encoding
        
    except Exception as e:
        print(f"Error encoding face from {image_path}: {e}")
        return None


def calculate_confidence(distance: float) -> float:
    """
    Calculate confidence score from face distance.
    
    The face_recognition library returns distances (lower = more similar).
    We convert this to a confidence score (0-1, higher = more confident).
    
    Args:
        distance: Face distance from face_recognition.compare_faces
        
    Returns:
        Confidence score between 0 and 1
    """
    # Convert distance to confidence using exponential decay
    # Distance of 0 = 100% confidence, distance of tolerance = ~50% confidence
    confidence = max(0.0, min(1.0, 1.0 - (distance / TOLERANCE)))
    return confidence


def recognize_face(
        face_encoding: np.ndarray,
        known_encodings: List[np.ndarray],
        known_names: List[str]
) -> Tuple[Optional[str], float]:

    if len(known_encodings) == 0:
        return None, 0.0

    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)

    if not any(matches):
        return None, 0.0

    matched_indices = [i for i, match in enumerate(matches) if match]
    matched_encodings = [known_encodings[i] for i in matched_indices]

    face_distances = face_recognition.face_distance(matched_encodings, face_encoding)

    best_match_idx = np.argmin(face_distances)
    best_distance = face_distances[best_match_idx]
    original_index = matched_indices[best_match_idx]

    confidence = calculate_confidence(best_distance)

    if confidence >= CONFIDENCE_THRESHOLD:
        return known_names[original_index], confidence

    return None, confidence


def draw_face_info(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    name: Optional[str],
    confidence: float,
    color: Tuple[int, int, int]
) -> None:

    top, right, bottom, left = face_location
    
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    if name:
        label = f"{name} ({confidence:.2%})"
    else:
        label = f"Unknown ({confidence:.2%})"
    
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    
    cv2.rectangle(
        frame,
        (left, top - text_height - 10),
        (left + text_width, top),
        color,
        -1
    )
    
    cv2.putText(
        frame,
        label,
        (left, top - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


def resize_frame(frame: np.ndarray, max_width: int = 800) -> np.ndarray:

    height, width = frame.shape[:2]
    
    if width <= max_width:
        return frame
    
    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


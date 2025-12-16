import cv2
import face_recognition
import numpy as np

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    COLOR_KNOWN,
    COLOR_UNKNOWN,
    FACE_DETECTION_MODEL,
    DETECTION_SKIP
)
from utils import (
    load_encodings,
    recognize_face,
    draw_face_info
)


class FaceRecognizer:
    
    def __init__(self):
        self.known_encodings, self.known_names = load_encodings()
        self.video_capture = None
        self.latest_faces = []
        self.running = False
        
        if len(self.known_encodings) == 0:
            print("Warning: No face encodings found!")
            print("Please run 'python train.py' first to train the model.")
    
    def initialize_camera(self):
        backends = []
        try:
            if hasattr(cv2, 'CAP_DSHOW'):
                backends.append(cv2.CAP_DSHOW)
        except:
            pass
        
        backends.append(cv2.CAP_ANY)
        
        self.video_capture = None
        for backend in backends:
            try:
                self.video_capture = cv2.VideoCapture(CAMERA_INDEX, backend)
                if self.video_capture.isOpened():
                    break
            except:
                continue
        
        if self.video_capture is None or not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(CAMERA_INDEX)
            if not self.video_capture.isOpened():
                raise RuntimeError(f"Unable to open camera {CAMERA_INDEX}")
        
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            self.video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except:
            pass
        
        print(f"Camera initialized: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    
    def detect_faces(self, frame: np.ndarray):
        scale_factor = 0.25
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            model=FACE_DETECTION_MODEL
        )

        if len(face_locations) == 0:
            self.latest_faces = []
            return

        scale_back = int(1 / scale_factor)
        face_locations_scaled = [
            (top * scale_back, right * scale_back, bottom * scale_back, left * scale_back)
            for (top, right, bottom, left) in face_locations
        ]

        face_encodings = face_recognition.face_encodings(
            rgb_small_frame,
            face_locations,
            num_jitters=0
        )

        detected_faces = []
        for i, face_location in enumerate(face_locations_scaled):
            if len(face_encodings) > i and len(self.known_encodings) > 0:
                name, confidence = recognize_face(
                    face_encodings[i],
                    self.known_encodings,
                    self.known_names
                )
                color = COLOR_KNOWN if name else COLOR_UNKNOWN
            else:
                name = None
                confidence = 0.0
                color = COLOR_UNKNOWN

            detected_faces.append((face_location, name, confidence, color))

        self.latest_faces = detected_faces
    
    
    def run(self):
        try:
            self.initialize_camera()
            self.running = True
            
            print("\nStarting face recognition...")
            print("Press 'q' to quit")
            print("-" * 50)
            
            frame_index = 0
            while self.running:
                ret, frame = self.video_capture.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break
                
                if frame_index % max(DETECTION_SKIP, 1) == 0:
                    self.detect_faces(frame)
                frame_index += 1

                for face_location, name, confidence, color in self.latest_faces:
                    draw_face_info(frame, face_location, name, confidence, color)
                
                cv2.putText(
                    frame,
                    "Press 'q' to quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                cv2.imshow('Face Recognition', frame)
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"Error during recognition: {e}")
        
        finally:
            self.running = False
            self.cleanup()
    
    def cleanup(self):
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()
        print("\nCamera released. Goodbye!")


def main():
    print("=" * 50)
    print("Real-Time Face Recognition System")
    print("=" * 50)
    
    recognizer = FaceRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()


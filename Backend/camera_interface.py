# Camera Interface Module for Face Recognition
# Handles camera operations, image capture, and live recognition display

import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple
from face_recognition_core import FaceRecognitionCore

class CameraInterface:
    def __init__(self, face_core: FaceRecognitionCore):
        """
        Initialize Camera Interface
        
        Args:
            face_core (FaceRecognitionCore): Face recognition core instance
        """
        self.face_core = face_core
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Camera settings - adaptive based on laptop capability
        self.camera_settings = {
            "width": 640,  # Start with standard resolution
            "height": 480,
            "fps": 30
        }
        
        # Recognition settings
        self.recognition_enabled = True
        self.last_recognition_time = 0
        self.recognition_interval = 1.5  # Reduced from 2.0 to 1.5 seconds
        self.last_recognition_result = None  # Initialize this
        self.recognition_in_progress = False  # Track if recognition is running
        
        # Face tracking
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Capture settings
        self.capture_mode = False
        self.captured_images = []
        self.max_captures = 3
        
        # Display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        
        # Colors (BGR format)
        self.colors = {
            "green": (0, 255, 0),      # Success/Real face
            "red": (0, 0, 255),        # Error/Fake face
            "blue": (255, 0, 0),       # Info
            "yellow": (0, 255, 255),   # Warning
            "white": (255, 255, 255),  # Text
            "black": (0, 0, 0)         # Background
        }
    
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """
        Initialize camera with optimal settings for the laptop
        
        Args:
            camera_id (int): Camera device ID (default: 0)
        
        Returns:
            bool: True if camera initialized successfully
        """
        try:
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                print("❌ Error: Could not open camera")
                return False
            
            # Try to set optimal resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_settings["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_settings["height"])
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_settings["fps"])
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Camera initialized:")
            print(f"   📐 Resolution: {actual_width}x{actual_height}")
            print(f"   🎬 FPS: {actual_fps}")
            
            # Update settings with actual values
            self.camera_settings.update({
                "width": actual_width,
                "height": actual_height,
                "fps": actual_fps
            })
            
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def start_registration_mode(self, username: str) -> Dict:
        """
        Start camera in registration mode to capture user images
        
        Args:
            username (str): Username for registration
        
        Returns:
            Dict: Registration result
        """
        if not self.cap or not self.cap.isOpened():
            return {
                "success": False,
                "message": "Camera not initialized"
            }
        
        print(f"📸 Starting registration for: {username}")
        print("📋 Instructions:")
        print("   • Position your face in the center")
        print("   • Press SPACEBAR to capture image")
        print("   • Capture 3 different angles/expressions")
        print("   • Press 'q' to quit")
        
        self.capture_mode = True
        self.captured_images = []
        
        while self.capture_mode and len(self.captured_images) < self.max_captures:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add registration overlay
            self._draw_registration_overlay(frame, username, len(self.captured_images))
            
            # Display frame
            cv2.imshow('Face Registration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar to capture
                captured_image = frame.copy()
                self.captured_images.append(captured_image)
                print(f"✅ Captured image {len(self.captured_images)}/{self.max_captures}")
                
                # Brief feedback
                cv2.putText(frame, f"CAPTURED {len(self.captured_images)}", 
                           (50, 100), self.font, 1, self.colors["green"], 3)
                cv2.imshow('Face Registration', frame)
                cv2.waitKey(200)  # Show feedback for 200ms
                
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyWindow('Face Registration')
        self.capture_mode = False
        
        # Process captured images
        if len(self.captured_images) > 0:
            print(f"🔄 Processing {len(self.captured_images)} captured images...")
            result = self.face_core.register_user(username, self.captured_images)
            return result
        else:
            return {
                "success": False,
                "message": "No images captured"
            }
    
    def start_recognition_mode(self):
        """
        Start camera in live recognition mode
        """
        if not self.cap or not self.cap.isOpened():
            print("❌ Camera not initialized")
            return
        
        print("🔍 Starting live face recognition...")
        print("📋 Instructions:")
        print("   • Position your face in the camera")
        print("   • Recognition will happen automatically")
        print("   • Press 'q' to quit")
        print("   • Press 'r' to toggle recognition on/off")
        
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Store current frame
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Perform recognition (throttled for performance and prevent backlog)
            current_time = time.time()
            if (self.recognition_enabled and 
                not self.recognition_in_progress and  # Don't start if already in progress
                current_time - self.last_recognition_time > self.recognition_interval):
                
                # Mark recognition as in progress
                self.recognition_in_progress = True
                
                # Run recognition in separate thread to avoid blocking
                recognition_thread = threading.Thread(
                    target=self._process_recognition,
                    args=(frame.copy(),)
                )
                recognition_thread.daemon = True
                recognition_thread.start()
                
                self.last_recognition_time = current_time
            
            # Add recognition overlay
            self._draw_recognition_overlay(frame)
            
            # Display frame
            cv2.imshow('Live Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('r'):  # Toggle recognition
                self.recognition_enabled = not self.recognition_enabled
                status = "ENABLED" if self.recognition_enabled else "DISABLED"
                print(f"🔄 Recognition {status}")
        
        self.is_running = False
        cv2.destroyWindow('Live Face Recognition')
    
    def _process_recognition(self, frame: np.ndarray):
        """
        Process face recognition in background thread
        
        Args:
            frame (np.ndarray): Frame to process
        """
        try:
            result = self.face_core.recognize_face(frame)
            
            # Store result for display
            with self.frame_lock:
                self.last_recognition_result = result
                
        except Exception as e:
            with self.frame_lock:
                self.last_recognition_result = {
                    "success": False,
                    "message": f"Error: {str(e)}",
                    "username": "Error",
                    "confidence": 0.0,
                    "is_real": None
                }
        finally:
            # Mark recognition as completed
            self.recognition_in_progress = False
    
    def _draw_registration_overlay(self, frame: np.ndarray, username: str, captured_count: int):
        """
        Draw overlay for registration mode with face tracking
        
        Args:
            frame (np.ndarray): Frame to draw on
            username (str): Username being registered
            captured_count (int): Number of images already captured
        """
        height, width = frame.shape[:2]
        
        # Title
        title = f"REGISTERING: {username}"
        cv2.putText(frame, title, (10, 30), self.font, self.font_scale, 
                   self.colors["blue"], self.thickness)
        
        # Progress
        progress = f"Images: {captured_count}/{self.max_captures}"
        cv2.putText(frame, progress, (10, 60), self.font, self.font_scale, 
                   self.colors["white"], self.thickness)
        
        # Real-time face detection for tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
        
        # Draw rectangles around detected faces
        face_detected = len(faces) > 0
        for (x, y, w, h) in faces:
            color = self.colors["green"] if face_detected else self.colors["red"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "Face Detected - Ready to Capture", (x, y - 10), 
                       self.font, 0.5, color, 1)
        
        # Instructions
        if captured_count < self.max_captures:
            if face_detected:
                instruction = "Press SPACEBAR to capture"
                color = self.colors["green"]
            else:
                instruction = "Position your face in view"
                color = self.colors["yellow"]
            cv2.putText(frame, instruction, (10, height - 60), self.font, 
                       self.font_scale, color, self.thickness)
        else:
            instruction = "Press 'q' to finish"
            cv2.putText(frame, instruction, (10, height - 60), self.font, 
                       self.font_scale, self.colors["green"], self.thickness)
        
        # Face guide rectangle (center guide)
        center_x, center_y = width // 2, height // 2
        rect_size = min(width, height) // 3
        guide_color = self.colors["green"] if face_detected else self.colors["white"]
        cv2.rectangle(frame, 
                     (center_x - rect_size//2, center_y - rect_size//2),
                     (center_x + rect_size//2, center_y + rect_size//2),
                     guide_color, 2)
        
        # Status text
        status = "Face Detected!" if face_detected else "No Face Detected"
        status_color = self.colors["green"] if face_detected else self.colors["red"]
        cv2.putText(frame, status, (10, 90), self.font, 0.6, status_color, 1)
    
    def _draw_recognition_overlay(self, frame: np.ndarray):
        """
        Draw overlay for recognition mode with face tracking
        
        Args:
            frame (np.ndarray): Frame to draw on
        """
        height, width = frame.shape[:2]
        
        # Title
        title = "LIVE FACE RECOGNITION"
        cv2.putText(frame, title, (10, 30), self.font, self.font_scale, 
                   self.colors["blue"], self.thickness)
        
        # Recognition status
        if not self.recognition_enabled:
            status = "RECOGNITION DISABLED"
            cv2.putText(frame, status, (10, 60), self.font, self.font_scale, 
                       self.colors["red"], self.thickness)
            return
        
        # Real-time face detection for tracking (using OpenCV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors["white"], 2)
            # Move "Face Detected" text below the box to avoid overlap
            cv2.putText(frame, "Face Detected", (x, y + h + 20), 
                       self.font, 0.5, self.colors["white"], 1)
        
        # Show recognition result or processing status
        try:
            with self.frame_lock:
                if hasattr(self, 'last_recognition_result') and self.last_recognition_result:
                    result = self.last_recognition_result
                    
                    # Username and confidence
                    username = result.get('username', 'Unknown')
                    confidence = result.get('confidence', 0.0)
                    is_real = result.get('is_real', None)
                    
                    # Choose color based on result
                    if result.get('success', False):
                        text_color = self.colors["green"]
                        status_text = f"Recognized: {username}"
                    elif username == "SPOOFING DETECTED":
                        text_color = self.colors["red"]
                        status_text = f"SPOOFING DETECTED"
                    else:
                        text_color = self.colors["yellow"]
                        status_text = f"Unknown User"
                    
                    # Display username
                    cv2.putText(frame, status_text, (10, 60), self.font, 
                               self.font_scale, text_color, self.thickness)
                    
                    # Display confidence
                    confidence_text = f"Confidence: {confidence:.1%}"
                    cv2.putText(frame, confidence_text, (10, 90), self.font, 
                               0.6, self.colors["white"], 1)
                    
                    # Display anti-spoofing status
                    if is_real is not None:
                        spoof_text = f"Anti-Spoofing: {'Real' if is_real else 'Fake'}"
                        spoof_color = self.colors["green"] if is_real else self.colors["red"]
                        cv2.putText(frame, spoof_text, (10, 120), self.font, 
                                   0.6, spoof_color, 1)
                    else:
                        spoof_text = "Anti-Spoofing: Unknown"
                        cv2.putText(frame, spoof_text, (10, 120), self.font, 
                                   0.6, self.colors["yellow"], 1)
                    
                    # Enhanced face box from DeepFace result
                    facial_area = result.get('facial_area', {})
                    if facial_area and len(faces) > 0:
                        # Use the first detected face for labeling
                        x, y, w, h = faces[0]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), text_color, 3)
                        
                        # Label below the rectangle to avoid overlap
                        label_y = y + h + 45  # Position below the box with some spacing
                        cv2.putText(frame, username, (x, label_y), 
                                   self.font, 0.7, text_color, 2)
                else:
                    # Show default status when no recognition result yet
                    cv2.putText(frame, "Processing...", (10, 60), self.font, 
                               self.font_scale, self.colors["white"], self.thickness)
                    cv2.putText(frame, "Waiting for face recognition", (10, 90), self.font, 
                               0.6, self.colors["white"], 1)
        except Exception as e:
            print(f"Overlay error: {e}")
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to toggle recognition", 
                   (10, height - 20), self.font, 0.5, self.colors["white"], 1)
    
    def cleanup(self):
        """Clean up camera resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera resources cleaned up")

# Test the camera interface
if __name__ == "__main__":
    print("🎥 Testing Camera Interface...")
    
    # Initialize face recognition core
    from face_recognition_core import FaceRecognitionCore
    face_core = FaceRecognitionCore()
    
    # Initialize camera interface
    camera = CameraInterface(face_core)
    
    if camera.initialize_camera():
        print("✅ Camera interface ready!")
        print("🔧 Available test modes:")
        print("   1. Registration mode")
        print("   2. Recognition mode")
    else:
        print("❌ Camera initialization failed!")

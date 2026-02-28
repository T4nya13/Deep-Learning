# Face Recognition and Anti-Spoofing Core Module
# High Accuracy Configuration for Banking Application

import os
import cv2
import numpy as np
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from deepface import DeepFace

class FaceRecognitionCore:
    def __init__(self, db_path: str = "face_recognition_db"):
        """
        Initialize Face Recognition Core with high accuracy configuration
        """
        self.db_path = db_path
        self.users_path = os.path.join(db_path, "users")
        self.embeddings_cache = {}
        
        # High Accuracy Configuration - Fixed for compatibility
        self.config = {
            "model_name": "Facenet512",  # Match existing users' model
            "detector_backend": "opencv",  # More reliable than retinaface
            "distance_metric": "cosine",
            "anti_spoofing": True,  # Enable anti-spoofing
            "confidence_threshold": 0.70,  # Lower threshold for better detection
            "enforce_detection": False,  # Don't enforce to avoid crashes
            "align": True,
            "expand_percentage": 10  # Add some padding
        }
        
        # Create database structure
        self._initialize_database()
        
        # Load existing user embeddings
        self._load_user_embeddings()
    
    def _initialize_database(self):
        """Create database directory structure"""
        os.makedirs(self.users_path, exist_ok=True)
        
        # Create logs directory
        logs_path = os.path.join(self.db_path, "logs")
        os.makedirs(logs_path, exist_ok=True)
        
        print(f"✅ Database initialized at: {self.db_path}")
    
    def _load_user_embeddings(self):
        """Load all user embeddings into memory for fast recognition"""
        self.embeddings_cache = {}
        
        if not os.path.exists(self.users_path):
            return
            
        for username in os.listdir(self.users_path):
            user_dir = os.path.join(self.users_path, username)
            if os.path.isdir(user_dir):
                profile_path = os.path.join(user_dir, "profile.json")
                if os.path.exists(profile_path):
                    try:
                        with open(profile_path, 'r') as f:
                            profile = json.load(f)
                            embeddings = profile.get('embeddings', [])
                            
                            # Check model compatibility
                            stored_model = profile.get('model_config', {}).get('model_name', 'Unknown')
                            if stored_model != self.config['model_name'] and stored_model != 'Unknown':
                                pass  # Silent model mismatch handling
                            
                            self.embeddings_cache[username] = embeddings
                    except Exception as e:
                        pass  # Silent error handling
        
        # Cache loaded silently
    
    def register_user(self, username: str, images: List[np.ndarray]) -> Dict:
        """
        Register a new user with multiple face images
        
        Args:
            username (str): Username for registration
            images (List[np.ndarray]): List of face images (max 3)
        
        Returns:
            Dict: Registration result with success status and details
        """
        try:
            # Create user directory
            user_dir = os.path.join(self.users_path, username)
            os.makedirs(user_dir, exist_ok=True)
            
            embeddings = []
            valid_images = 0
            anti_spoofing_results = []
            
            for i, image in enumerate(images[:3]):  # Max 3 images
                try:
                    # Extract faces with anti-spoofing using DeepFace
                    face_objs = DeepFace.extract_faces(
                        img_path=image,
                        detector_backend=self.config["detector_backend"],
                        enforce_detection=False,  # Don't enforce to avoid crashes
                        align=self.config["align"],
                        anti_spoofing=self.config["anti_spoofing"]  # Use DeepFace anti-spoofing
                    )
                    
                    if not face_objs:
                        continue
                    
                    # Check anti-spoofing result from DeepFace
                    face_obj = face_objs[0]  # Take first detected face
                    is_real = face_obj.get("is_real", False)  # Default to False for security
                    
                    anti_spoofing_results.append(is_real)
                    
                    # STRICT: Skip this image if it's detected as fake
                    if not is_real:
                        print(f"⚠️ Image {i+1} failed anti-spoofing check - REJECTED")
                        continue
                    
                    # Generate embedding
                    embedding_obj = DeepFace.represent(
                        img_path=image,
                        model_name=self.config["model_name"],
                        detector_backend=self.config["detector_backend"],
                        enforce_detection=False,  # Changed to False
                        align=self.config["align"]
                    )
                    
                    if embedding_obj:
                        embeddings.append(embedding_obj[0]["embedding"])
                        
                        # Save image
                        image_path = os.path.join(user_dir, f"image_{i+1}.jpg")
                        cv2.imwrite(image_path, image)
                        valid_images += 1
                
                except Exception as e:
                    # STRICT: NO FALLBACK - If anti-spoofing fails, reject the image
                    print(f"⚠️ Image {i+1} anti-spoofing failed: {e} - REJECTED")
                    anti_spoofing_results.append(False)
                    continue
            
            if valid_images == 0:
                return {
                    "success": False,
                    "message": "No valid face images found or all failed anti-spoofing",
                    "details": {
                        "valid_images": valid_images,
                        "anti_spoofing_results": anti_spoofing_results
                    }
                }
            
            # Save user profile
            profile = {
                "username": username,
                "registration_date": datetime.now().isoformat(),
                "total_images": valid_images,
                "embeddings": embeddings,
                "model_config": self.config,
                "anti_spoofing_results": anti_spoofing_results
            }
            
            profile_path = os.path.join(user_dir, "profile.json")
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            # Update cache
            self.embeddings_cache[username] = embeddings
            
            # Log registration
            self._log_activity(username, "USER_REGISTERED", True, 
                             f"Registered with {valid_images} valid images")
            
            return {
                "success": True,
                "message": f"User {username} registered successfully",
                "details": {
                    "valid_images": valid_images,
                    "total_embeddings": len(embeddings),
                    "anti_spoofing_passed": sum(anti_spoofing_results),
                    "anti_spoofing_failed": len(anti_spoofing_results) - sum(anti_spoofing_results)
                }
            }
            
        except Exception as e:
            self._log_activity(username, "USER_REGISTRATION_FAILED", False, str(e))
            return {
                "success": False,
                "message": f"Registration failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def recognize_face(self, image: np.ndarray) -> Dict:
        """
        Recognize face in the given image with DeepFace anti-spoofing
        
        Args:
            image (np.ndarray): Input image for recognition
        
        Returns:
            Dict: Recognition result with username, confidence, and anti-spoofing status
        """
        try:
            # Extract faces with DeepFace anti-spoofing
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.config["detector_backend"],
                enforce_detection=False,  # Don't raise exception if no face
                align=self.config["align"],
                anti_spoofing=self.config["anti_spoofing"]  # Use DeepFace anti-spoofing
            )
            
            if not face_objs:
                return {
                    "success": False,
                    "message": "No face detected",
                    "username": None,
                    "confidence": 0.0,
                    "is_real": None,
                    "facial_area": None
                }
            
            # Get face information and anti-spoofing result
            face_obj = face_objs[0]
            is_real = face_obj.get("is_real", False)  # Default to False for security
            facial_area = face_obj.get("facial_area", {})
            
            # STRICT ANTI-SPOOFING: Reject immediately if fake detected
            if not is_real:
                print(f"🚫 ANTI-SPOOFING FAILED: Fake face detected!")
                self._log_activity("UNKNOWN", "SPOOFING_DETECTED", False, "DeepFace anti-spoofing detected fake face")
                return {
                    "success": False,
                    "message": "SECURITY ALERT: Fake face detected by anti-spoofing",
                    "username": "SPOOFING_DETECTED",
                    "confidence": 0.0,
                    "is_real": False,
                    "facial_area": facial_area
                }
            
            # Generate embedding for recognition
            try:
                embedding_obj = DeepFace.represent(
                    img_path=image,
                    model_name=self.config["model_name"],
                    detector_backend=self.config["detector_backend"],
                    enforce_detection=False,
                    align=self.config["align"]
                )
            except Exception as embed_error:
                return {
                    "success": False,
                    "message": "Could not generate face embedding",
                    "username": None,
                    "confidence": 0.0,
                    "is_real": is_real,
                    "facial_area": facial_area
                }
            
            if not embedding_obj:
                return {
                    "success": False,
                    "message": "Could not generate face embedding",
                    "username": None,
                    "confidence": 0.0,
                    "is_real": is_real,
                    "facial_area": facial_area
                }
            
            query_embedding = embedding_obj[0]["embedding"]
            
            # Find best match among registered users
            best_match = None
            best_distance = float('inf')
            best_username = None
            
            for username, user_embeddings in self.embeddings_cache.items():
                for user_embedding in user_embeddings:
                    # Calculate cosine distance
                    distance = self._calculate_cosine_distance(query_embedding, user_embedding)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_username = username
                        best_match = distance
            
            # Convert distance to confidence (0-1 scale)
            if best_match is not None:
                confidence = max(0.0, 1.0 - best_distance)
                
                # Check if confidence meets threshold
                if confidence >= self.config["confidence_threshold"]:
                    self._log_activity(best_username, "FACE_RECOGNIZED", True, 
                                     f"Confidence: {confidence:.2f}, DeepFace anti-spoofing: {'Pass' if is_real else 'Fail'}")
                    
                    return {
                        "success": True,
                        "message": "Face recognized successfully",
                        "username": best_username,
                        "confidence": confidence,
                        "is_real": is_real,
                        "facial_area": facial_area
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Low confidence recognition ({confidence:.2f})",
                        "username": f"Unknown (best: {best_username})",
                        "confidence": confidence,
                        "is_real": is_real,
                        "facial_area": facial_area
                    }
            else:
                return {
                    "success": False,
                    "message": "No matching user found",
                    "username": "Unknown",
                    "confidence": 0.0,
                    "is_real": is_real,
                    "facial_area": facial_area
                }
                
        except Exception as e:
            # STRICT: NO FALLBACK - If anti-spoofing fails, authentication fails
            print(f"🚫 ANTI-SPOOFING ERROR: {e}")
            self._log_activity("UNKNOWN", "ANTI_SPOOFING_ERROR", False, f"Anti-spoofing failed: {str(e)}")
            return {
                "success": False,
                "message": f"SECURITY ERROR: Anti-spoofing check failed - {str(e)}",
                "username": "ANTI_SPOOFING_ERROR",
                "confidence": 0.0,
                "is_real": False,
                "facial_area": None
            }
    
    def _calculate_cosine_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine distance between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Check dimension compatibility
            if vec1.shape != vec2.shape:
                return 1.0  # Maximum distance for incompatible embeddings
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance
            
            cosine_similarity = dot_product / (norm1 * norm2)
            cosine_distance = 1.0 - cosine_similarity
            
            return max(0.0, cosine_distance)  # Ensure non-negative
            
        except Exception as e:
            return 1.0  # Maximum distance on error
    
    def _log_activity(self, username: str, activity: str, success: bool, details: str = ""):
        """Log user activities"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "username": username,
                "activity": activity,
                "success": success,
                "details": details
            }
            
            # Log to daily file
            log_date = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(self.db_path, "logs", f"{log_date}.log")
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            pass  # Silent logging error
    
    def get_registered_users(self) -> List[str]:
        """Get list of all registered users"""
        return list(self.embeddings_cache.keys())
    
    def delete_user(self, username: str) -> bool:
        """Delete a registered user"""
        try:
            user_dir = os.path.join(self.users_path, username)
            if os.path.exists(user_dir):
                import shutil
                shutil.rmtree(user_dir)
                
                # Remove from cache
                if username in self.embeddings_cache:
                    del self.embeddings_cache[username]
                
                self._log_activity(username, "USER_DELETED", True, "User profile deleted")
                return True
            return False
        except Exception as e:
            self._log_activity(username, "USER_DELETE_FAILED", False, str(e))
            return False
    
    def check_model_compatibility(self) -> Dict:
        """Check if all users are compatible with current model"""
        compatibility_report = {
            "current_model": self.config["model_name"],
            "compatible_users": [],
            "incompatible_users": [],
            "recommendations": []
        }
        
        for username in self.get_registered_users():
            user_dir = os.path.join(self.users_path, username)
            profile_path = os.path.join(user_dir, "profile.json")
            
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r') as f:
                        profile = json.load(f)
                        stored_model = profile.get('model_config', {}).get('model_name', 'Unknown')
                        
                        if stored_model == self.config['model_name'] or stored_model == 'Unknown':
                            compatibility_report["compatible_users"].append(username)
                        else:
                            compatibility_report["incompatible_users"].append({
                                "username": username,
                                "stored_model": stored_model,
                                "current_model": self.config["model_name"]
                            })
                except Exception as e:
                    pass  # Silent error handling
        
        # Generate recommendations
        if compatibility_report["incompatible_users"]:
            compatibility_report["recommendations"].append(
                "Some users have incompatible embeddings. Consider:"
            )
            compatibility_report["recommendations"].append(
                "1. Re-register incompatible users with current model"
            )
            compatibility_report["recommendations"].append(
                "2. Or change current model to match majority of users"
            )
        
        return compatibility_report

# Test the core functionality
if __name__ == "__main__":
    print("🔧 Testing Face Recognition Core...")
    
    # Initialize the core system
    face_core = FaceRecognitionCore()
    
    print("✅ Face Recognition Core initialized successfully!")
    print(f"📁 Database path: {face_core.db_path}")
    print(f"⚙️ Configuration: {face_core.config}")
    print(f"👥 Registered users: {len(face_core.get_registered_users())}")

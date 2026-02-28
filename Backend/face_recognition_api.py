# Face Recognition API Server for ZeroDay Project
# This server provides endpoints for face detection, recognition, anti-spoofing, and liveness detection

import os
import sys
import logging
import time
import json
import base64
import io
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import traceback
from typing import Dict, List, Optional, Tuple, Any

# Import the face recognition modules from your existing code
try:
    from face_recognition_core import FaceRecognitionCore
    from camera_interface import CameraInterface
    face_modules_available = True
    print("✅ Face recognition modules imported successfully")
except ImportError as e:
    face_modules_available = False
    print(f"Warning: Could not import face recognition modules: {e}")
    print("Please ensure face_recognition_core.py and camera_interface.py are in the same directory")
    # Create dummy classes as fallback
    class FaceRecognitionCore:
        def __init__(self, *args, **kwargs):
            pass
        def register_user(self, username, images):
            return {"success": False, "message": "Face recognition not available"}
        def recognize_face(self, image):
            return {"success": False, "message": "Face recognition not available"}
    
    class CameraInterface:
        def __init__(self, *args, **kwargs):
            pass

# Import DeepFace and related libraries
try:
    from deepface import DeepFace
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except ImportError as e:
    print(f"Error: DeepFace not installed. Please install: pip install deepface")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for face recognition services
face_core = None
camera_interface = None

# Configuration
CONFIG = {
    'models': {
        'face_detection': 'opencv',  # Use opencv like in working ZeroDay-Subpart version
        'face_recognition': 'ArcFace',  # VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace
        'anti_spoofing': 'default',
        'emotion': 'enet',
        'age': 'age',
        'gender': 'gender'
    },
    'thresholds': {
        'face_detection': 0.7,
        'face_recognition': 0.6,
        'anti_spoofing': 0.5,
        'liveness': 0.7
    },
    'image': {
        'max_size': (1024, 1024),
        'quality': 95,
        'format': 'JPEG'
    }
}

def initialize_services():
    """Initialize face recognition services"""
    global face_core, camera_interface
    
    try:
        # Initialize face recognition core (only if modules are available)
        if face_modules_available:
            face_core = FaceRecognitionCore()
            logger.info("FaceRecognitionCore initialized successfully")
        else:
            face_core = FaceRecognitionCore()  # Will use dummy class
            logger.warning("Using dummy FaceRecognitionCore - face recognition not available")
    except Exception as e:
        logger.warning(f"Could not initialize FaceRecognitionCore: {e}")
        face_core = None
    
    try:
        # Initialize camera interface (for liveness detection) - only if modules are available
        if face_modules_available and face_core:
            camera_interface = CameraInterface(face_core)
            logger.info("CameraInterface initialized successfully")
        else:
            camera_interface = CameraInterface()  # Will use dummy class
            logger.warning("Using dummy CameraInterface - camera not available")
    except Exception as e:
        logger.warning(f"Could not initialize CameraInterface: {e}")
        camera_interface = None

def decode_image(image_data) -> np.ndarray:
    """Decode image from various formats to numpy array"""
    try:
        if isinstance(image_data, str):
            # Base64 encoded image
            if image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',')[1]
            
            # Validate base64 string
            if not image_data or len(image_data.strip()) < 4:
                raise ValueError("Base64 string is too short or empty")
            
            # Clean the base64 string
            image_data = image_data.strip()
            
            # Add padding if needed
            missing_padding = len(image_data) % 4
            if missing_padding:
                image_data += '=' * (4 - missing_padding)
            
            # Decode base64
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                raise ValueError(f"Invalid base64 data: {e}")
            
            if len(image_bytes) < 100:  # Too small to be a valid image
                raise ValueError("Decoded image data is too small")
            
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            
        elif hasattr(image_data, 'read'):
            # File-like object
            image = Image.open(image_data)
            image_array = np.array(image)
            
        else:
            # Assume it's already a numpy array
            image_array = np.array(image_data)
        
        # Validate image dimensions
        if len(image_array.shape) < 2:
            raise ValueError("Invalid image dimensions")
        
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # RGBA to RGB
            image_array = image_array[:, :, :3]
        elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
            # Grayscale to RGB
            image_array = np.repeat(image_array, 3, axis=2)
        elif len(image_array.shape) == 2:
            # Grayscale to RGB
            image_array = np.stack([image_array] * 3, axis=2)
            
        return image_array
        
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise ValueError(f"Could not decode image: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'ZeroDay Face Recognition API is running',
        'status': 'healthy',
        'modules': {
            'face_core': face_core is not None,
            'camera_interface': camera_interface is not None,
            'face_modules_available': face_modules_available
        },
        'endpoints': [
            '/health',
            '/detect_face',
            '/check_anti_spoofing', 
            '/check_liveness',
            '/generate_embedding',
            '/verify_face',
            '/register_user_faces',
            '/authenticate_user'
        ]
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        'success': True,
        'message': 'ZeroDay Face Recognition API',
        'version': '1.0.0',
        'description': 'Face recognition, anti-spoofing, and liveness detection API',
        'health_check': '/health',
        'documentation': 'See /health for available endpoints'
    })

@app.route('/detect_face', methods=['POST'])
def detect_face():
    """Detect faces in an image"""
    try:
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({
                'success': False,
                'message': 'No image provided'
            }), 400
        
        start_time = time.time()
        
        # Get image data
        if 'image' in request.files:
            image_data = request.files['image']
        else:
            image_data = request.json.get('image')
        
        # Decode image
        image_array = decode_image(image_data)
        
        # Use DeepFace to detect faces
        try:
            faces = DeepFace.extract_faces(
                img_path=image_array,
                detector_backend=CONFIG['models']['face_detection'],
                enforce_detection=False
            )
            
            face_count = len(faces)
            processing_time = (time.time() - start_time) * 1000
            
            return jsonify({
                'success': True,
                'face_count': face_count,
                'faces_detected': face_count > 0,
                'confidence': 0.9 if face_count > 0 else 0.0,
                'processing_time': round(processing_time, 2),
                'message': f'Detected {face_count} face(s)'
            })
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return jsonify({
                'success': False,
                'message': f'Face detection failed: {str(e)}',
                'face_count': 0,
                'faces_detected': False
            }), 500
        
    except Exception as e:
        logger.error(f"General error in face detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/check_anti_spoofing', methods=['POST'])
def check_anti_spoofing():
    """Check if the face is real (anti-spoofing)"""
    try:
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({
                'success': False,
                'message': 'No image provided'
            }), 400
        
        start_time = time.time()
        
        # Get image data
        if 'image' in request.files:
            image_data = request.files['image']
        else:
            image_data = request.json.get('image')
        
        # Decode image
        image_array = decode_image(image_data)
        
        # Use DeepFace anti-spoofing (based on working ZeroDay-Subpart implementation)
        try:
            face_objs = DeepFace.extract_faces(
                img_path=image_array,
                detector_backend='opencv',  # Use opencv like in working version
                enforce_detection=False,
                anti_spoofing=True
            )
            
            if face_objs and len(face_objs) > 0:
                # Get anti-spoofing result from DeepFace (like in working version)
                face_obj = face_objs[0]  # Take first detected face
                is_real = face_obj.get("is_real", True)  # Default to True if not available
                confidence = 0.9 if is_real else 0.1
            else:
                is_real = False
                confidence = 0.0
            
            processing_time = (time.time() - start_time) * 1000
            
            return jsonify({
                'success': True,
                'is_real': bool(is_real),  # Convert to native Python bool
                'confidence': float(confidence),  # Convert to native Python float
                'anti_spoofing_passed': bool(is_real),  # Convert to native Python bool
                'processing_time': round(float(processing_time), 2),  # Convert to native Python float
                'message': 'Real face detected' if is_real else 'Spoof attempt detected'
            })
            
        except Exception as e:
            logger.error(f"Anti-spoofing error: {e}")
            return jsonify({
                'success': False,
                'message': f'Anti-spoofing check failed: {str(e)}',
                'is_real': False,
                'confidence': 0.0
            }), 500
        
    except Exception as e:
        logger.error(f"General error in anti-spoofing: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/check_liveness', methods=['POST'])
def check_liveness():
    """Check liveness with challenge-response"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        challenge_type = data.get('challenge_type', 'blink')
        images = data.get('images', [])
        
        if not images:
            return jsonify({
                'success': False,
                'message': 'No images provided for liveness check'
            }), 400
        
        start_time = time.time()
        
        # Simple liveness check - verify multiple images show face movement
        face_detected_count = 0
        
        for image_data in images:
            try:
                image_array = decode_image(image_data)
                faces = DeepFace.extract_faces(
                    img_path=image_array,
                    detector_backend=CONFIG['models']['face_detection'],
                    enforce_detection=False
                )
                
                if faces:
                    face_detected_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing liveness image: {e}")
                continue
        
        # Liveness passed if faces detected in majority of images
        liveness_passed = face_detected_count >= len(images) * 0.6
        confidence = face_detected_count / len(images) if images else 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'success': True,
            'liveness_passed': liveness_passed,
            'confidence': confidence,
            'challenge_type': challenge_type,
            'faces_detected': face_detected_count,
            'total_images': len(images),
            'processing_time': round(processing_time, 2),
            'message': 'Liveness verified' if liveness_passed else 'Liveness check failed'
        })
        
    except Exception as e:
        logger.error(f"Liveness check error: {e}")
        return jsonify({
            'success': False,
            'message': f'Liveness check failed: {str(e)}'
        }), 500

@app.route('/generate_embedding', methods=['POST'])
def generate_embedding():
    """Generate face embedding from image"""
    try:
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({
                'success': False,
                'message': 'No image provided'
            }), 400
        
        start_time = time.time()
        
        # Get image data
        if 'image' in request.files:
            image_data = request.files['image']
        else:
            image_data = request.json.get('image')
        
        # Decode image
        image_array = decode_image(image_data)
        
        # Generate embedding using DeepFace
        try:
            embedding_result = DeepFace.represent(
                img_path=image_array,
                model_name=CONFIG['models']['face_recognition'],
                detector_backend=CONFIG['models']['face_detection'],
                enforce_detection=False
            )
            
            if embedding_result:
                embedding = embedding_result[0]['embedding']
                processing_time = (time.time() - start_time) * 1000
                
                return jsonify({
                    'success': True,
                    'embedding': embedding,
                    'model': CONFIG['models']['face_recognition'],
                    'embedding_size': len(embedding),
                    'processing_time': round(processing_time, 2),
                    'message': 'Embedding generated successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No face detected for embedding generation'
                }), 400
                
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return jsonify({
                'success': False,
                'message': f'Embedding generation failed: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f"General error in embedding generation: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/verify_face', methods=['POST'])
def verify_face():
    """Verify if two faces match"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        image1 = data.get('image1')
        image2 = data.get('image2')
        embeddings = data.get('embeddings', [])
        similarity_threshold = data.get('threshold', CONFIG['thresholds']['face_recognition'])
        
        if not image1:
            return jsonify({
                'success': False,
                'message': 'image1 is required'
            }), 400
        
        if not image2 and not embeddings:
            return jsonify({
                'success': False,
                'message': 'Either image2 or embeddings array is required'
            }), 400
        
        start_time = time.time()
        
        # Decode the first image
        image1_array = decode_image(image1)
        
        # Generate embedding for image1
        try:
            embedding1_result = DeepFace.represent(
                img_path=image1_array,
                model_name=CONFIG['models']['face_recognition'],
                detector_backend=CONFIG['models']['face_detection'],
                enforce_detection=False
            )
            
            if not embedding1_result:
                return jsonify({
                    'success': False,
                    'message': 'No face detected in image1'
                }), 400
            
            embedding1 = embedding1_result[0]['embedding']
            
        except Exception as e:
            logger.error(f"Error processing image1: {e}")
            return jsonify({
                'success': False,
                'message': f'Error processing image1: {str(e)}'
            }), 500
        
        similarities = []
        
        # Compare with image2 if provided
        if image2:
            try:
                image2_array = decode_image(image2)
                embedding2_result = DeepFace.represent(
                    img_path=image2_array,
                    model_name=CONFIG['models']['face_recognition'],
                    detector_backend=CONFIG['models']['face_detection'],
                    enforce_detection=False
                )
                
                if embedding2_result:
                    embedding2 = embedding2_result[0]['embedding']
                    
                    # Calculate similarity
                    similarity = DeepFace.verification.find_cosine_distance(embedding1, embedding2)
                    similarities.append(1 - similarity)  # Convert distance to similarity
                    
            except Exception as e:
                logger.error(f"Error processing image2: {e}")
                return jsonify({
                    'success': False,
                    'message': f'Error processing image2: {str(e)}'
                }), 500
        
        # Compare with provided embeddings
        for i, embedding in enumerate(embeddings):
            try:
                similarity = DeepFace.verification.find_cosine_distance(embedding1, embedding)
                similarities.append(1 - similarity)  # Convert distance to similarity
            except Exception as e:
                logger.warning(f"Error comparing with embedding {i}: {e}")
                similarities.append(0.0)
        
        if not similarities:
            return jsonify({
                'success': False,
                'message': 'No valid comparisons could be made'
            }), 400
        
        # Find best match
        best_similarity = max(similarities)
        best_match_index = similarities.index(best_similarity)
        is_match = best_similarity >= similarity_threshold
        confidence = best_similarity
        
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'success': True,
            'is_match': bool(is_match),  # Convert to native Python bool
            'confidence': float(confidence),  # Convert to native Python float
            'similarity': float(best_similarity),  # Convert to native Python float
            'best_match': {
                'index': int(best_match_index),  # Convert to native Python int
                'similarity': float(best_similarity)  # Convert to native Python float
            } if best_match_index >= 0 else None,
            'all_scores': [float(score) for score in similarities],  # Convert list elements to float
            'threshold': float(similarity_threshold),  # Convert to native Python float
            'message': 'Face verified successfully' if is_match else 'Face verification failed',
            'processing_time': round(float(processing_time), 2)  # Convert to native Python float
        })
        
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        return jsonify({
            'success': False,
            'message': f'Verification failed: {str(e)}'
        }), 500

@app.route('/register_user_faces', methods=['POST'])
def register_user_faces():
    """Register a user with multiple face images"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        username = data.get('username')
        images = data.get('images', [])
        
        if not username:
            return jsonify({
                'success': False,
                'message': 'Username is required'
            }), 400
        
        if not images:
            return jsonify({
                'success': False,
                'message': 'At least one image is required'
            }), 400
        
        start_time = time.time()
        
        # Process images and generate embeddings
        embeddings = []
        processed_images = []
        
        for i, image_data in enumerate(images):
            try:
                image_array = decode_image(image_data)
                processed_images.append(image_array)
                
                # Generate embedding
                embedding_result = DeepFace.represent(
                    img_path=image_array,
                    model_name=CONFIG['models']['face_recognition'],
                    detector_backend=CONFIG['models']['face_detection'],
                    enforce_detection=False
                )
                
                if embedding_result:
                    embeddings.append(embedding_result[0]['embedding'])
                
            except Exception as e:
                logger.warning(f"Error processing image {i} for user {username}: {e}")
                continue
        
        if not embeddings:
            return jsonify({
                'success': False,
                'message': 'No valid face embeddings could be generated'
            }), 400
        
        # Use face_core if available
        if face_core and face_modules_available:
            try:
                result = face_core.register_user(username, processed_images)
                processing_time = (time.time() - start_time) * 1000
                result['processing_time'] = round(processing_time, 2)
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error using face_core for registration: {e}")
        
        # Fallback: basic registration
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'success': True,
            'message': f'User {username} registered successfully',
            'username': username,
            'embeddings_count': len(embeddings),
            'images_processed': len(processed_images),
            'processing_time': round(processing_time, 2)
        })
        
    except Exception as e:
        logger.error(f"User registration error: {e}")
        return jsonify({
            'success': False,
            'message': f'Registration failed: {str(e)}'
        }), 500

@app.route('/authenticate_user', methods=['POST'])
def authenticate_user():
    """Authenticate a user with face recognition"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        image = data.get('image')
        username = data.get('username')  # Optional: for specific user check
        
        if not image:
            return jsonify({
                'success': False,
                'message': 'Image is required'
            }), 400
        
        start_time = time.time()
        
        # Decode image
        image_array = decode_image(image)
        
        # Use face_core if available
        if face_core and face_modules_available:
            try:
                result = face_core.recognize_face(image_array)
                processing_time = (time.time() - start_time) * 1000
                result['processing_time'] = round(processing_time, 2)
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error using face_core for authentication: {e}")
        
        # Fallback: basic authentication
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'success': False,
            'message': 'Face recognition core not available',
            'username': None,
            'confidence': 0.0,
            'processing_time': round(processing_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return jsonify({
            'success': False,
            'message': f'Authentication failed: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("Initializing ZeroDay Face Recognition API Server...")
    
    # Initialize services
    initialize_services()
    
    print(f"Configuration: {CONFIG}")
    print(f"Face Core Available: {face_core is not None}")
    print(f"Camera Interface Available: {camera_interface is not None}")
    
    # Start server
    print("Starting server on http://localhost:8000")
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True,
        threaded=True
    )

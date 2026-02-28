from flask import Flask, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# API endpoints only - React will handle the frontend
@app.route('/api/hello', methods=['GET'])
def hello_world():
    return jsonify(message="Hello from Flask!")

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # Add your authentication logic here
    # For now, basic validation
    if email and password:
        # Simulate successful login
        return jsonify(success=True, message="Login successful", user={"email": email})
    else:
        return jsonify(success=False, message="Invalid credentials"), 400

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('fullName')
    
    # Add your registration logic here
    # For now, basic validation
    if email and password and full_name:
        # Simulate successful registration
        return jsonify(success=True, message="Registration successful", user={"email": email, "name": full_name})
    else:
        return jsonify(success=False, message="Missing required fields"), 400

@app.route('/api/user/profile', methods=['GET'])
def get_profile():
    # Mock user profile data
    return jsonify({
        "success": True,
        "user": {
            "name": "John Doe",
            "email": "john@example.com",
            "securityFeatures": {
                "faceAuth": True,
                "behaviorAnalysis": True,
                "fraudDetection": True
            }
        }
    })

@app.route('/api/security/status', methods=['GET'])
def security_status():
    # Mock security status
    return jsonify({
        "success": True,
        "status": {
            "threatLevel": "LOW",
            "activeSessions": 1,
            "recentActivity": [
                {"action": "Face authentication verified", "time": "12:34 PM", "status": "success"},
                {"action": "Behavior pattern normal", "time": "12:33 PM", "status": "success"},
                {"action": "New device detected - blocked", "time": "12:30 PM", "status": "warning"}
            ]
        }
    })

if __name__ == '__main__':
    app.run(debug=True)


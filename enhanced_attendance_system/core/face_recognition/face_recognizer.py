"""
Face Recognizer Module
This module provides face recognition functionality for the Enhanced Attendance System.
"""

import os
import cv2
import numpy as np
import logging
import time

class FaceRecognizer:
    """
    A class for face recognition.
    """
    
    def __init__(self, model_path):
        """
        Initialize the face recognizer.
        
        Args:
            model_path (str): Path to the trained model
        """
        self.model_path = model_path
        self.recognizer = None
        self.logger = logging.getLogger("AttendanceSystem")
        self.face_module_available = self._check_face_module()
        self.model_loaded = False
        
        # Try to initialize recognizer
        if self.face_module_available:
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                
                # Try to load model if it exists
                if os.path.exists(model_path):
                    self.load_model()
            except Exception as e:
                self.logger.error(f"Error initializing face recognizer: {e}")
    
    def _check_face_module(self):
        """
        Check if OpenCV face module is available.
        
        Returns:
            bool: True if face module is available, False otherwise
        """
        try:
            # Try to access face module
            face_module = cv2.face
            return True
        except AttributeError:
            self.logger.warning("OpenCV face module not available. Install opencv-contrib-python for face recognition.")
            return False
    
    def is_face_module_available(self):
        """
        Check if face module is available.
        
        Returns:
            bool: True if face module is available, False otherwise
        """
        return self.face_module_available
    
    def is_model_loaded(self):
        """
        Check if model is loaded.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.model_loaded
    
    def load_model(self):
        """
        Load the trained model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not self.face_module_available:
            return False
        
        try:
            if os.path.exists(self.model_path):
                self.recognizer.read(self.model_path)
                self.model_loaded = True
                self.logger.info(f"Model loaded from {self.model_path}")
                return True
            else:
                self.logger.warning(f"Model file not found: {self.model_path}")
                self.model_loaded = False
                return False
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def save_model(self):
        """
        Save the trained model.
        
        Returns:
            bool: True if model saved successfully, False otherwise
        """
        if not self.face_module_available or not self.model_loaded:
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            self.recognizer.write(self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def train(self, faces, ids):
        """
        Train the face recognizer.
        
        Args:
            faces (list): List of face images
            ids (list): List of corresponding IDs
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if not self.face_module_available:
            return False
        
        try:
            # Ensure faces and ids are not empty
            if not faces or not ids:
                self.logger.error("No faces or IDs provided for training")
                return False
            
            # Ensure faces and ids have the same length
            if len(faces) != len(ids):
                self.logger.error(f"Number of faces ({len(faces)}) does not match number of IDs ({len(ids)})")
                return False
            
            # Preprocess faces
            processed_faces = []
            processed_ids = []
            
            for i, face in enumerate(faces):
                try:
                    # Ensure face is grayscale
                    if len(face.shape) > 2:
                        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = face
                    
                    # Resize to standard size
                    resized = cv2.resize(gray, (100, 100))
                    
                    processed_faces.append(resized)
                    processed_ids.append(ids[i])
                except Exception as e:
                    self.logger.warning(f"Error preprocessing face {i}: {e}")
            
            # Ensure we still have faces after preprocessing
            if not processed_faces:
                self.logger.error("No faces left after preprocessing")
                return False
            
            # Convert to numpy arrays
            np_faces = np.array(processed_faces)
            np_ids = np.array(processed_ids)
            
            # Train recognizer
            self.recognizer.train(np_faces, np_ids)
            self.model_loaded = True
            
            # Save model
            self.save_model()
            
            return True
        except Exception as e:
            self.logger.error(f"Error training face recognizer: {e}")
            return False
    
    def recognize_face(self, face_img):
        """
        Recognize a face.
        
        Args:
            face_img: Face image
            
        Returns:
            tuple: (ID, confidence) where ID is the recognized ID or -1 if not recognized
        """
        if not self.face_module_available or not self.model_loaded:
            return -1, 100.0
        
        try:
            # Ensure face is grayscale
            if len(face_img.shape) > 2:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            # Resize to standard size
            resized = cv2.resize(gray, (100, 100))
            
            # Recognize
            id, confidence = self.recognizer.predict(resized)
            
            return id, confidence
        except Exception as e:
            self.logger.error(f"Error recognizing face: {e}")
            return -1, 100.0
    
    def update_model(self, faces, ids):
        """
        Update the existing model with new faces.
        
        Args:
            faces (list): List of face images
            ids (list): List of corresponding IDs
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.face_module_available:
            return False
        
        try:
            # If model is not loaded, just train from scratch
            if not self.model_loaded:
                return self.train(faces, ids)
            
            # Update model
            self.recognizer.update(faces, np.array(ids))
            
            # Save model
            self.save_model()
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating face recognizer: {e}")
            return False

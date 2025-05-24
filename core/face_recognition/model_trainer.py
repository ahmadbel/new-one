"""
Model Trainer Module
This module provides functionality for training face recognition models with improved accuracy.
"""

import os
import cv2
import numpy as np
import pickle
from PIL import Image
import time
import threading
import logging

class ModelTrainer:
    """
    A class for training face recognition models with enhanced performance.
    """
    
    def __init__(self, cascade_path=None):
        """
        Initialize the model trainer.
        
        Args:
            cascade_path (str): Path to the Haar cascade XML file for face detection
        """
        # Set up logging
        self.logger = logging.getLogger("AttendanceSystem")
        
        # Set standard face size for consistent training
        self.face_width = 100
        self.face_height = 100
        
        if cascade_path is None:
            # Use the default cascade file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cascade_path = os.path.join(current_dir, '..', '..', 'data', 'haarcascades', 'haarcascade_frontalface_default.xml')
        
        self.detector = cv2.CascadeClassifier(cascade_path)
        
        # Verify that the cascade file was loaded
        if self.detector.empty():
            raise ValueError(f"Error: Could not load cascade classifier from {cascade_path}")
    
    def preprocess_face(self, face_img):
        """
        Preprocess a face image for consistent training.
        
        Args:
            face_img: Face image to preprocess
            
        Returns:
            numpy.ndarray: Preprocessed face image
        """
        try:
            # Convert to grayscale if needed
            if len(face_img.shape) > 2 and face_img.shape[2] > 1:
                gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = face_img
            
            # Resize to standard size
            resized_img = cv2.resize(gray_img, (self.face_width, self.face_height))
            
            # Normalize lighting (optional)
            # normalized_img = cv2.equalizeHist(resized_img)
            
            return resized_img
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {e}")
            return None
    
    def extract_faces_from_directory(self, directory, callback=None):
        """
        Extract faces from all images in the specified directory structure.
        
        Args:
            directory (str): Path to the directory containing student subdirectories
            callback (function, optional): Callback function to report progress
            
        Returns:
            tuple: (faces, labels, names) where:
                  faces is a list of face images,
                  labels is a list of corresponding labels,
                  names is a dictionary mapping labels to names
        """
        faces = []
        labels = []
        names = {}
        
        # Get all subdirectories (one per student)
        try:
            student_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        except Exception as e:
            self.logger.error(f"Error accessing directory {directory}: {e}")
            if callback:
                callback(-1, f"Error: {e}")
            return [], [], {}
            
        total_dirs = len(student_dirs)
        
        for i, student_dir in enumerate(student_dirs):
            # Try to extract student ID from directory name
            try:
                student_id = int(student_dir)
                student_name = f"Student {student_id}"
            except ValueError:
                # If directory name is not a number, try to parse as "ID_Name"
                try:
                    parts = student_dir.split('_', 1)
                    if len(parts) > 1:
                        student_id = int(parts[0])
                        student_name = parts[1]
                    else:
                        self.logger.warning(f"Directory name '{student_dir}' does not follow expected format")
                        continue
                except ValueError:
                    self.logger.warning(f"Could not extract student ID from directory name '{student_dir}'")
                    continue
            
            # Store the name mapping
            names[student_id] = student_name
            
            # Process all images in the student directory
            student_path = os.path.join(directory, student_dir)
            try:
                image_files = [f for f in os.listdir(student_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            except Exception as e:
                self.logger.error(f"Error accessing student directory {student_path}: {e}")
                continue
            
            student_faces = 0
            for image_file in image_files:
                try:
                    # Load the image
                    image_path = os.path.join(student_path, image_file)
                    img = cv2.imread(image_path)
                    
                    if img is None:
                        self.logger.warning(f"Could not read image {image_path}")
                        continue
                    
                    # Preprocess the face image
                    processed_face = self.preprocess_face(img)
                    
                    if processed_face is None:
                        self.logger.warning(f"Failed to preprocess image {image_path}")
                        continue
                    
                    # Add to the training set
                    faces.append(processed_face)
                    labels.append(student_id)
                    student_faces += 1
                except Exception as e:
                    self.logger.error(f"Error processing image {image_file}: {e}")
            
            self.logger.info(f"Processed {student_faces} images for student {student_id}")
            
            # Report progress if callback is provided
            if callback:
                progress = (i + 1) / total_dirs * 100
                callback(progress, f"Processed {i+1}/{total_dirs} students")
        
        return faces, labels, names
    
    def train_model(self, faces, labels, names, model_path, callback=None):
        """
        Train a face recognition model with the provided faces and labels.
        
        Args:
            faces (list): List of face images
            labels (list): List of corresponding labels
            names (dict): Dictionary mapping labels to names
            model_path (str): Path where the model should be saved
            callback (function, optional): Callback function to report progress
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Check if we have OpenCV face module
            if not hasattr(cv2, 'face'):
                self.logger.error("OpenCV face module not available")
                if callback:
                    callback(-1, "Error: OpenCV face module not available. Please install opencv-contrib-python.")
                return False
            
            # Check if we have enough data
            if len(faces) == 0 or len(labels) == 0:
                self.logger.error("No training data available")
                if callback:
                    callback(-1, "Error: No training data available")
                return False
            
            # Create and train the recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Report progress if callback is provided
            if callback:
                callback(0, "Starting training...")
            
            # Ensure all faces are the same size and format
            processed_faces = []
            processed_labels = []
            
            for i, (face, label) in enumerate(zip(faces, labels)):
                try:
                    # Double-check that each face is properly preprocessed
                    if face.shape != (self.face_height, self.face_width):
                        face = cv2.resize(face, (self.face_width, self.face_height))
                    
                    processed_faces.append(face)
                    processed_labels.append(label)
                except Exception as e:
                    self.logger.error(f"Error processing face {i}: {e}")
            
            if len(processed_faces) == 0:
                self.logger.error("No valid faces after processing")
                if callback:
                    callback(-1, "Error: No valid faces after processing")
                return False
            
            # Convert lists to numpy arrays
            np_faces = np.array(processed_faces)
            np_labels = np.array(processed_labels)
            
            # Log shape information for debugging
            self.logger.info(f"Training with {len(np_faces)} faces, shape: {np_faces.shape}")
            
            # Train the model
            recognizer.train(np_faces, np_labels)
            
            # Report progress if callback is provided
            if callback:
                callback(50, "Model trained, saving...")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            recognizer.write(model_path)
            
            # Save the labels info
            labels_info_path = os.path.splitext(model_path)[0] + "_labels.pkl"
            with open(labels_info_path, 'wb') as f:
                pickle.dump(names, f)
            
            # Report progress if callback is provided
            if callback:
                callback(100, "Training completed successfully")
            
            return True
        except Exception as e:
            self.logger.error(f"Error training face recognizer: {e}")
            if callback:
                callback(-1, f"Error: {e}")
            return False
    
    def train_from_directory(self, directory, model_path, callback=None):
        """
        Extract faces from a directory and train a model in one step.
        
        Args:
            directory (str): Path to the directory containing student subdirectories
            model_path (str): Path where the model should be saved
            callback (function, optional): Callback function to report progress
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        # Extract faces
        if callback:
            callback(0, "Starting face extraction...")
        
        faces, labels, names = self.extract_faces_from_directory(directory, callback)
        
        if len(faces) == 0:
            self.logger.error("No faces found in the directory")
            if callback:
                callback(-1, "No faces found in the directory")
            return False
        
        # Train the model
        if callback:
            callback(50, f"Extracted {len(faces)} faces. Starting training...")
        
        return self.train_model(faces, labels, names, model_path, callback)
    
    def train_async(self, directory, model_path, progress_callback=None, completion_callback=None):
        """
        Train a model asynchronously in a separate thread.
        
        Args:
            directory (str): Path to the directory containing student subdirectories
            model_path (str): Path where the model should be saved
            progress_callback (function, optional): Function to call with progress updates
            completion_callback (function, optional): Function to call when training completes
            
        Returns:
            threading.Thread: The thread object performing the training
        """
        def training_thread():
            success = self.train_from_directory(directory, model_path, progress_callback)
            if completion_callback:
                completion_callback(success)
        
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
        
        return thread

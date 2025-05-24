"""
Test Suite for Enhanced Attendance System
This module provides comprehensive testing for all system components.
"""

import os
import sys
import unittest
import tkinter as tk
import cv2
import numpy as np
import tempfile
import shutil
import time
import threading

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_recognition.face_detector import FaceDetector
from core.face_recognition.face_recognizer import FaceRecognizer
from core.face_recognition.model_trainer import ModelTrainer
from core.data_management.database import DatabaseManager
from core.data_management.config import ConfigManager
from core.data_management.attendance_logger import AttendanceLogger
from core.data_management.alert_system import AlertSystem
from utils.ui_components import ModernUI, CameraFeed
from utils.validators import *
from utils.logger import Logger
from utils.image_processing import *
from utils.theme_manager import ThemeManager
from utils.icon_manager import IconManager
from utils.modern_ui import *

class TestFaceDetector(unittest.TestCase):
    """Test cases for the FaceDetector class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test image with a face
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (255, 255, 255), -1)  # White square as a "face"
        
        # Initialize face detector
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(current_dir, '..', 'Attendance-Management-system-using-face-recognition', 'haarcascade_frontalface_default.xml')
        self.detector = FaceDetector(cascade_path)
    
    def test_detect_faces(self):
        """Test face detection functionality."""
        # This is a basic test - in a real scenario, we would use actual face images
        faces = self.detector.detect_faces(self.test_image)
        self.assertIsInstance(faces, np.ndarray, "detect_faces should return a numpy array")
    
    def test_detect_and_draw(self):
        """Test face detection with drawing functionality."""
        result_image, faces = self.detector.detect_and_draw(self.test_image)
        self.assertIsInstance(result_image, np.ndarray, "detect_and_draw should return a numpy array as the first element")
        self.assertIsInstance(faces, np.ndarray, "detect_and_draw should return a numpy array as the second element")
        
        # Check that the result image is different from the input (has drawings)
        self.assertFalse(np.array_equal(self.test_image, result_image), "Result image should have drawings")

class TestDatabaseManager(unittest.TestCase):
    """Test cases for the DatabaseManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize database manager with test directory
        self.db = DatabaseManager(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_add_student(self):
        """Test adding a student to the database."""
        # Add a test student
        result = self.db.add_student("12345", "Test Student")
        self.assertTrue(result, "add_student should return True on success")
        
        # Check if student exists
        self.assertTrue(self.db.student_exists("12345"), "Student should exist after adding")
    
    def test_get_student_name(self):
        """Test retrieving a student's name."""
        # Add a test student
        self.db.add_student("12345", "Test Student")
        
        # Get the student's name
        name = self.db.get_student_name("12345")
        self.assertEqual(name, "Test Student", "get_student_name should return the correct name")
    
    def test_mark_attendance(self):
        """Test marking attendance for a student."""
        # Add a test student
        self.db.add_student("12345", "Test Student")
        
        # Mark attendance
        result = self.db.mark_attendance("12345", "Test Subject")
        self.assertTrue(result, "mark_attendance should return True on success")
        
        # Get attendance records
        attendance = self.db.get_attendance("Test Subject")
        self.assertFalse(attendance.empty, "Attendance records should not be empty")
        self.assertEqual(len(attendance), 1, "There should be one attendance record")
        self.assertEqual(attendance.iloc[0]['ID'], "12345", "Attendance record should have the correct ID")

class TestValidators(unittest.TestCase):
    """Test cases for validator functions."""
    
    def test_validate_student_id(self):
        """Test student ID validation."""
        self.assertTrue(validate_student_id("12345"), "Valid student ID should pass validation")
        self.assertFalse(validate_student_id(""), "Empty student ID should fail validation")
        self.assertFalse(validate_student_id("abc"), "Non-numeric student ID should fail validation")
        self.assertFalse(validate_student_id("1"), "Too short student ID should fail validation")
    
    def test_validate_name(self):
        """Test name validation."""
        self.assertTrue(validate_name("John Doe"), "Valid name should pass validation")
        self.assertFalse(validate_name(""), "Empty name should fail validation")
        self.assertFalse(validate_name("J"), "Too short name should fail validation")
        self.assertFalse(validate_name("John123"), "Name with numbers should fail validation")
    
    def test_validate_subject(self):
        """Test subject validation."""
        self.assertTrue(validate_subject("Computer Science"), "Valid subject should pass validation")
        self.assertFalse(validate_subject(""), "Empty subject should fail validation")
        self.assertTrue(validate_subject("CS101"), "Subject with numbers should pass validation")
    
    def test_validate_date(self):
        """Test date validation."""
        self.assertTrue(validate_date("2023-01-01"), "Valid date should pass validation")
        self.assertFalse(validate_date(""), "Empty date should fail validation")
        self.assertFalse(validate_date("01-01-2023"), "Incorrectly formatted date should fail validation")
        self.assertFalse(validate_date("2023-13-01"), "Invalid month should fail validation")

class TestImageProcessing(unittest.TestCase):
    """Test cases for image processing functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test image
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (255, 255, 255), -1)  # White square
    
    def test_resize_image(self):
        """Test image resizing functionality."""
        # Resize to specific dimensions
        resized = resize_image(self.test_image, width=150, height=150)
        self.assertEqual(resized.shape[:2], (150, 150), "Resized image should have the specified dimensions")
        
        # Resize with only width specified
        resized = resize_image(self.test_image, width=150)
        self.assertEqual(resized.shape[1], 150, "Resized image should have the specified width")
        
        # Resize with only height specified
        resized = resize_image(self.test_image, height=150)
        self.assertEqual(resized.shape[0], 150, "Resized image should have the specified height")
    
    def test_draw_face_box(self):
        """Test drawing a box around a face."""
        # Draw a box
        face_location = (100, 100, 100, 100)  # x, y, w, h
        result = draw_face_box(self.test_image, face_location)
        
        # Check that the result is different from the input (has drawings)
        self.assertFalse(np.array_equal(self.test_image, result), "Result image should have drawings")

class TestThemeManager(unittest.TestCase):
    """Test cases for the ThemeManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test config file path
        self.config_file = os.path.join(self.test_dir, "theme_config.json")
        
        # Initialize theme manager with test config file
        self.theme_manager = ThemeManager(self.config_file)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_get_theme_colors(self):
        """Test getting theme colors."""
        # Get dark theme colors
        dark_colors = self.theme_manager.get_theme_colors("dark")
        self.assertIsInstance(dark_colors, dict, "get_theme_colors should return a dictionary")
        self.assertIn("bg", dark_colors, "Theme colors should include 'bg'")
        
        # Get light theme colors
        light_colors = self.theme_manager.get_theme_colors("light")
        self.assertIsInstance(light_colors, dict, "get_theme_colors should return a dictionary")
        self.assertIn("bg", light_colors, "Theme colors should include 'bg'")
        
        # Dark and light themes should be different
        self.assertNotEqual(dark_colors["bg"], light_colors["bg"], "Dark and light themes should have different background colors")
    
    def test_toggle_theme(self):
        """Test toggling between themes."""
        # Get initial theme
        initial_theme = self.theme_manager.current_theme
        
        # Toggle theme
        new_theme = self.theme_manager.toggle_theme()
        
        # Check that the theme changed
        self.assertNotEqual(initial_theme, new_theme, "Theme should change after toggling")
        self.assertEqual(self.theme_manager.current_theme, new_theme, "current_theme should be updated after toggling")
        
        # Toggle again
        final_theme = self.theme_manager.toggle_theme()
        
        # Check that we're back to the initial theme
        self.assertEqual(initial_theme, final_theme, "Theme should toggle back to initial theme")

class TestIconManager(unittest.TestCase):
    """Test cases for the IconManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test icons
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize icon manager with test directory
        self.icon_manager = IconManager(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_get_icon_path(self):
        """Test getting icon paths."""
        # Get path for an icon
        path = self.icon_manager.get_icon_path("user", "dark")
        
        # Path should be in the test directory
        self.assertTrue(path.startswith(self.test_dir), "Icon path should be in the test directory")
        
        # Path should be for a file that exists
        self.assertTrue(os.path.exists(path), "Icon file should exist")

class TestUIComponents(unittest.TestCase):
    """Test cases for UI components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a root window for testing
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window
    
    def tearDown(self):
        """Clean up test environment."""
        # Destroy the root window
        self.root.destroy()
    
    def test_modern_button(self):
        """Test ModernButton class."""
        # Create a button
        button = ModernButton(self.root, text="Test Button")
        
        # Check that the button was created
        self.assertIsInstance(button, ModernButton, "ModernButton should be created successfully")
        
        # Check that the button has the expected attributes
        self.assertEqual(button["text"], "Test Button", "Button should have the specified text")
    
    def test_animated_progress_bar(self):
        """Test AnimatedProgressBar class."""
        # Create a progress bar
        progress_bar = AnimatedProgressBar(self.root, width=200, height=20, value=50)
        
        # Check that the progress bar was created
        self.assertIsInstance(progress_bar, AnimatedProgressBar, "AnimatedProgressBar should be created successfully")
        
        # Check that the progress bar has the expected attributes
        self.assertEqual(progress_bar.value, 50, "Progress bar should have the specified value")
        
        # Update the progress
        progress_bar.update_progress(75)
        self.assertEqual(progress_bar.value, 75, "Progress bar value should be updated")

def run_tests():
    """Run all tests."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestFaceDetector))
    test_suite.addTest(unittest.makeSuite(TestDatabaseManager))
    test_suite.addTest(unittest.makeSuite(TestValidators))
    test_suite.addTest(unittest.makeSuite(TestImageProcessing))
    test_suite.addTest(unittest.makeSuite(TestThemeManager))
    test_suite.addTest(unittest.makeSuite(TestIconManager))
    test_suite.addTest(unittest.makeSuite(TestUIComponents))
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    return test_result

if __name__ == "__main__":
    run_tests()

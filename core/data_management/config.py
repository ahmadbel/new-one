"""
Configuration Manager Module
This module provides functionality for managing application configuration.
"""

import os
import json
import platform

class ConfigManager:
    """
    A class for managing application configuration.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file (str): Path to the configuration file
        """
        # Set default configuration file path if not provided
        if config_file is None:
            # Get the base directory of the application
            base_dir = self._get_base_dir()
            config_file = os.path.join(base_dir, 'data', 'config.json')
        
        self.config_file = config_file
        self.config = self._load_default_config()
        
        # Create configuration directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        # Load configuration from file if it exists
        if os.path.exists(self.config_file):
            self._load_config()
        else:
            # Save default configuration
            self._save_config()
    
    def _get_base_dir(self):
        """
        Get the base directory of the application.
        
        Returns:
            str: Base directory path
        """
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go up two levels to get the base directory
        base_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        return base_dir
    
    def _load_default_config(self):
        """
        Load default configuration.
        
        Returns:
            dict: Default configuration
        """
        base_dir = self._get_base_dir()
        
        # Use os.path.join for cross-platform compatibility
        return {
            'General': {
                'ApplicationName': 'Enhanced Attendance System',
                'Version': '1.0.0',
                'Theme': 'dark',
                'LogLevel': 'INFO'
            },
            'Database': {
                'StudentsDirectory': os.path.join(base_dir, 'data', 'students'),
                'AttendanceDirectory': os.path.join(base_dir, 'data', 'attendance')
            },
            'FaceRecognition': {
                'CascadePath': os.path.join(base_dir, 'data', 'haarcascades', 'haarcascade_frontalface_default.xml'),
                'ModelPath': os.path.join(base_dir, 'data', 'models', 'face_recognition_model.yml'),
                'ImagesDirectory': os.path.join(base_dir, 'data', 'images'),
                'MinFaceSize': '30',
                'ScaleFactor': '1.1',
                'MinNeighbors': '5',
                'ConfidenceThreshold': '80'
            },
            'AlertSystem': {
                'AlertImagesDirectory': os.path.join(base_dir, 'data', 'alerts'),
                'AlertDuration': '30',
                'AlertCooldown': '10',
                'SoundEnabled': 'True'
            },
            'UI': {
                'AdminTitle': 'Admin Interface - Enhanced Attendance System',
                'ScannerTitle': 'Scanner Interface - Enhanced Attendance System',
                'FontSize': '10',
                'ButtonWidth': '15',
                'ButtonHeight': '2'
            }
        }
    
    def _load_config(self):
        """
        Load configuration from file.
        
        Returns:
            bool: True if configuration was loaded successfully, False otherwise
        """
        try:
            with open(self.config_file, 'r') as f:
                loaded_config = json.load(f)
            
            # Update configuration with loaded values
            for section, values in loaded_config.items():
                if section in self.config:
                    for key, value in values.items():
                        if key in self.config[section]:
                            self.config[section][key] = value
            
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def _save_config(self):
        """
        Save configuration to file.
        
        Returns:
            bool: True if configuration was saved successfully, False otherwise
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get_value(self, section, key, default=None):
        """
        Get a configuration value.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            default: Default value to return if the key doesn't exist
            
        Returns:
            Configuration value or default if not found
        """
        if section in self.config and key in self.config[section]:
            return self.config[section][key]
        return default
    
    def set_value(self, section, key, value):
        """
        Set a configuration value.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            value: Value to set
            
        Returns:
            bool: True if value was set successfully, False otherwise
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        return self._save_config()
    
    def get_path(self, section, key, default=None):
        """
        Get a path from configuration, ensuring it uses the correct path separator.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            default: Default path to return if the key doesn't exist
            
        Returns:
            str: Path with correct separators for the current OS
        """
        path = self.get_value(section, key, default)
        
        if path:
            # Normalize path for the current OS
            return os.path.normpath(path)
        
        return default
    
    def ensure_directories_exist(self):
        """
        Ensure all configured directories exist.
        
        Returns:
            bool: True if all directories were created successfully, False otherwise
        """
        try:
            # Create database directories
            os.makedirs(self.get_path('Database', 'StudentsDirectory'), exist_ok=True)
            os.makedirs(self.get_path('Database', 'AttendanceDirectory'), exist_ok=True)
            
            # Create face recognition directories
            os.makedirs(os.path.dirname(self.get_path('FaceRecognition', 'CascadePath')), exist_ok=True)
            os.makedirs(os.path.dirname(self.get_path('FaceRecognition', 'ModelPath')), exist_ok=True)
            os.makedirs(self.get_path('FaceRecognition', 'ImagesDirectory'), exist_ok=True)
            
            # Create alert system directories
            os.makedirs(self.get_path('AlertSystem', 'AlertImagesDirectory'), exist_ok=True)
            
            return True
        except Exception as e:
            print(f"Error creating directories: {e}")
            return False
    
    def get_os_name(self):
        """
        Get the name of the current operating system.
        
        Returns:
            str: Operating system name ('Windows', 'Linux', 'Darwin', etc.)
        """
        return platform.system()
    
    def is_windows(self):
        """
        Check if the current operating system is Windows.
        
        Returns:
            bool: True if Windows, False otherwise
        """
        return platform.system() == 'Windows'
    
    def is_linux(self):
        """
        Check if the current operating system is Linux.
        
        Returns:
            bool: True if Linux, False otherwise
        """
        return platform.system() == 'Linux'
    
    def is_mac(self):
        """
        Check if the current operating system is macOS.
        
        Returns:
            bool: True if macOS, False otherwise
        """
        return platform.system() == 'Darwin'

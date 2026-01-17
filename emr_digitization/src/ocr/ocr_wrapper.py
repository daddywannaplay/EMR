import logging
import os
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class OCRModelWrapper:
    """Wrapper for trained OCR model"""
    
    def __init__(self, model_path: str):
        """
        Initialize OCR model wrapper
        
        Args:
            model_path: Path to trained OCR model
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained OCR model"""
        try:
            # Depending on your OCR model type (TensorFlow, PyTorch, etc.)
            # Adjust this accordingly
            
            if self.model_path.endswith('.pt'):
                # PyTorch model
                import torch
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()
                logger.info(f"Loaded PyTorch OCR model from {self.model_path}")
            
            elif self.model_path.endswith('.h5') or self.model_path.endswith('.pb'):
                # TensorFlow model
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded TensorFlow OCR model from {self.model_path}")
            
            elif self.model_path.endswith('.pkl'):
                # Pickle model
                import pickle
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded pickled OCR model from {self.model_path}")
            
            elif os.path.isdir(self.model_path):
                # ONNX or other directory-based models
                logger.info(f"OCR model directory found at {self.model_path}")
                # Load ONNX or other formats
                import onnxruntime as rt
                model_files = list(Path(self.model_path).glob('*.onnx'))
                if model_files:
                    self.model = rt.InferenceSession(str(model_files[0]))
                    logger.info(f"Loaded ONNX OCR model")
            
            else:
                logger.warning(f"Unknown model format: {self.model_path}")
        
        except Exception as e:
            logger.error(f"Failed to load OCR model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for OCR"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if necessary
            max_size = 2048
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Normalize
            img_array = img_array.astype(np.float32) / 255.0
            
            return img_array
        
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image using OCR model
        
        Args:
            image_path: Path to medical document image
            
        Returns:
            Dictionary with extracted text and confidence scores
        """
        if not self.model:
            logger.error("OCR model not loaded")
            return {'text': '', 'confidence': 0.0, 'error': 'Model not loaded'}
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                return {'text': '', 'confidence': 0.0, 'error': 'Image preprocessing failed'}
            
            # Run inference
            if hasattr(self.model, 'predict'):
                # Keras/TensorFlow
                predictions = self.model.predict(np.expand_dims(img_array, axis=0))
            
            elif hasattr(self.model, 'run'):
                # ONNX
                input_name = self.model.get_inputs()[0].name
                predictions = self.model.run(None, {input_name: np.expand_dims(img_array, axis=0)})
            
            elif hasattr(self.model, '__call__'):
                # PyTorch
                import torch
                with torch.no_grad():
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                    predictions = self.model(img_tensor)
            
            else:
                logger.error("Unknown model type")
                return {'text': '', 'confidence': 0.0, 'error': 'Unknown model type'}
            
            # Parse predictions (adjust based on your model output format)
            extracted_text = self._parse_predictions(predictions)
            confidence = self._calculate_confidence(predictions)
            
            return {
                'text': extracted_text,
                'confidence': confidence,
                'image_path': image_path,
                'raw_predictions': predictions
            }
        
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return {'text': '', 'confidence': 0.0, 'error': str(e)}
    
    def _parse_predictions(self, predictions) -> str:
        """Parse model predictions into text"""
        # This depends on your specific OCR model output format
        # Example implementations below:
        
        if isinstance(predictions, np.ndarray):
            if predictions.ndim == 1:
                # If predictions are class indices
                return str(predictions.tolist())
            elif predictions.ndim == 2:
                # If predictions are confidence scores per character
                return self._decode_character_predictions(predictions)
        
        elif isinstance(predictions, list):
            if predictions and isinstance(predictions[0], np.ndarray):
                return self._decode_character_predictions(predictions[0])
        
        elif isinstance(predictions, str):
            return predictions
        
        return str(predictions)
    
    @staticmethod
    def _decode_character_predictions(char_predictions) -> str:
        """Decode character-level predictions to text"""
        # Assuming character predictions are confidence scores
        if isinstance(char_predictions, list):
            char_predictions = np.array(char_predictions)
        
        # Get character indices with highest confidence
        indices = np.argmax(char_predictions, axis=-1) if char_predictions.ndim > 1 else char_predictions
        
        # Map indices to characters (adjust based on your character set)
        charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?\' ()-'
        text = ''.join([charset[i] if i < len(charset) else '' for i in indices])
        return text
    
    @staticmethod
    def _calculate_confidence(predictions) -> float:
        """Calculate overall confidence score"""
        if isinstance(predictions, np.ndarray):
            if predictions.ndim == 1:
                return float(np.max(predictions))
            elif predictions.ndim == 2:
                return float(np.mean(np.max(predictions, axis=-1)))
        return 0.5
    
    def batch_extract_text(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Extract text from multiple images"""
        results = []
        for image_path in image_paths:
            result = self.extract_text(image_path)
            results.append(result)
        return results

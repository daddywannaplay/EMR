import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class SpellCorrector:
    """Correct spelling errors in OCR text using medical dictionaries"""
    
    def __init__(self, medical_dict_path: str = None):
        self.medical_dict = {}
        self.common_abbreviations = {}
        self.medication_corrections = {}
        self.common_misspellings = {}
        
        if medical_dict_path:
            self.load_medical_dictionary(medical_dict_path)
    
    def load_medical_dictionary(self, dict_path: str):
        """Load medical dictionary from JSON file"""
        try:
            with open(dict_path, 'r') as f:
                data = json.load(f)
            self.common_abbreviations = data.get('common_abbreviations', {})
            self.medication_corrections = data.get('medication_corrections', {})
            self.common_misspellings = data.get('common_misspellings', {})
            logger.info(f"Medical dictionary loaded from {dict_path}")
        except Exception as e:
            logger.error(f"Failed to load medical dictionary: {e}")
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations"""
        for abbr, expansion in self.common_abbreviations.items():
            text = text.replace(f" {abbr} ", f" {expansion} ")
            text = text.replace(f" {abbr}.", f" {expansion}.")
        return text
    
    def correct_medication_names(self, text: str) -> str:
        """Correct common medication name misspellings"""
        for correct_name, variations in self.medication_corrections.items():
            for variation in variations:
                text = text.replace(variation, correct_name)
        return text
    
    def correct_common_misspellings(self, text: str) -> str:
        """Correct common medical misspellings"""
        for misspelled, correct in self.common_misspellings.items():
            text = text.replace(misspelled, correct)
        return text
    
    def similarity_ratio(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def fuzzy_correct(self, word: str, candidates: List[str], threshold: float = 0.8) -> str:
        """Find closest match from candidates using fuzzy matching"""
        best_match = word
        best_score = 0.0
        
        for candidate in candidates:
            score = self.similarity_ratio(word, candidate)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
        
        return best_match
    
    def correct_text(self, text: str) -> str:
        """Apply all correction methods to text"""
        text = self.expand_abbreviations(text)
        text = self.correct_medication_names(text)
        text = self.correct_common_misspellings(text)
        return text


class StructuredFieldExtractor:
    """Extract and normalize structured medical fields"""
    
    FIELD_PATTERNS = {
        'demographics': {
            'patient_name': r'(?:patient\s*name|name)[:\s]+([^\n]+)',
            'patient_id': r'(?:patient\s*id|mrn|medical\s*record)[:\s]+([A-Z0-9-]+)',
            'dob': r'(?:dob|date\s*of\s*birth)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            'gender': r'(?:gender|sex)[:\s]+(M|F|Male|Female)',
            'age': r'(?:age)[:\s]+(\d+)'
        },
        'vitals': {
            'blood_pressure': r'(?:bp|blood\s*pressure)[:\s]+(\d{2,3}[/\\]\d{2,3})',
            'heart_rate': r'(?:hr|heart\s*rate|pulse)[:\s]+(\d{2,3})',
            'temperature': r'(?:temp|temperature)[:\s]+(\d{2,3}\.?\d*)',
            'respiratory_rate': r'(?:rr|respiratory\s*rate)[:\s]+(\d{1,2})',
            'oxygen_saturation': r'(?:o2|spo2|oxygen)[:\s]+(\d{1,3}%?)'
        },
        'lab_values': {
            'glucose': r'(?:glucose|blood\s*sugar)[:\s]+(\d{1,3})',
            'hemoglobin': r'(?:hb|hemoglobin)[:\s]+(\d{1,2}\.?\d*)',
            'wbc': r'(?:wbc|white\s*blood\s*cell)[:\s]+(\d{1,2}\.?\d*)',
            'platelet': r'(?:plt|platelet)[:\s]+(\d{3,4})'
        }
    }
    
    @staticmethod
    def extract_fields(text: str) -> Dict[str, Dict[str, str]]:
        """Extract structured fields from medical text"""
        import re
        extracted = {}
        
        for category, patterns in StructuredFieldExtractor.FIELD_PATTERNS.items():
            extracted[category] = {}
            for field_name, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted[category][field_name] = match.group(1)
        
        return extracted

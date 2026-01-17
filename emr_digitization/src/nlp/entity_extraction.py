import torch
import logging
from typing import List, Dict, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re

logger = logging.getLogger(__name__)

class ClinicalNLPExtractor:
    """Extract clinical entities and relationships using transformer models"""
    
    def __init__(self, ner_model_name: str = "allenai/scibert_scivocab_uncased"):
        """
        Initialize NLP extractor with pre-trained models
        
        Args:
            ner_model_name: Named Entity Recognition model name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Initialize NER pipeline
            self.ner_pipeline = pipeline(
                "token-classification",
                model="DistilBERT-clinical-NER",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("NER pipeline initialized")
        except Exception as e:
            logger.warning(f"Could not load clinical NER model: {e}. Using basic pattern matching.")
            self.ner_pipeline = None
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text
        
        Args:
            text: Input clinical text
            
        Returns:
            List of extracted entities with their types and positions
        """
        if not text or len(text.strip()) == 0:
            return []
        
        try:
            if self.ner_pipeline:
                entities = self.ner_pipeline(text[:512])  # Truncate to token limit
                return entities
            else:
                return self._pattern_based_extraction(text)
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return self._pattern_based_extraction(text)
    
    def _pattern_based_extraction(self, text: str) -> List[Dict[str, str]]:
        """Fallback pattern-based entity extraction"""
        entities = []
        
        # Medical condition patterns
        condition_patterns = [
            r'\b(diabetes|hypertension|pneumonia|sepsis|asthma|arthritis)\b',
            r'\b(chest pain|shortness of breath|fever|cough)\b'
        ]
        
        for pattern in condition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'entity': 'CONDITION',
                    'score': 0.95,
                    'index': match.start(),
                    'word': match.group(0)
                })
        
        # Medication patterns
        med_pattern = r'\b([A-Z][a-z]+(?:illin|ifene|olol|pril|azole|statin|ine))\b'
        matches = re.finditer(med_pattern, text)
        for match in matches:
            entities.append({
                'entity': 'MEDICATION',
                'score': 0.9,
                'index': match.start(),
                'word': match.group(0)
            })
        
        return entities
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        Extract relationships between entities (e.g., medication-dosage)
        
        Args:
            text: Input text
            entities: Extracted entities
            
        Returns:
            List of relationships
        """
        relationships = []
        
        # Pattern: medication followed by dosage
        med_dosage_pattern = r'(\w+(?:illin|ifene|olol|pril|azole|statin|ine))\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|units|tablets?|capsules?)'
        
        matches = re.finditer(med_dosage_pattern, text, re.IGNORECASE)
        for match in matches:
            relationships.append({
                'subject': match.group(1),
                'predicate': 'has_dosage',
                'object': f"{match.group(2)} {match.group(3)}",
                'confidence': 0.9
            })
        
        # Pattern: condition with value
        value_pattern = r'(blood\s*pressure|glucose|temperature|heart\s*rate)\s*[:\-]?\s*(\d{2,3}[/\\]?\d{0,3}|\d+\.?\d*)'
        
        matches = re.finditer(value_pattern, text, re.IGNORECASE)
        for match in matches:
            relationships.append({
                'subject': match.group(1),
                'predicate': 'has_value',
                'object': match.group(2),
                'confidence': 0.85
            })
        
        return relationships
    
    def normalize_entities(self, entities: List[Dict], ontology_mappings: Dict = None) -> List[Dict]:
        """
        Normalize extracted entities to standard medical codes
        
        Args:
            entities: Raw extracted entities
            ontology_mappings: Mapping to standard ontologies (SNOMED, ICD-10, LOINC)
            
        Returns:
            Normalized entities with standard codes
        """
        if not ontology_mappings:
            ontology_mappings = self._get_default_mappings()
        
        normalized = []
        for entity in entities:
            entity_text = entity.get('word', '').lower()
            entity_type = entity.get('entity', '')
            
            # Look up in ontology mappings
            key = f"{entity_type}:{entity_text}"
            standard_code = ontology_mappings.get(key, {})
            
            normalized.append({
                **entity,
                'normalized': {
                    'snomed_ct': standard_code.get('snomed_ct'),
                    'icd10': standard_code.get('icd10'),
                    'loinc': standard_code.get('loinc'),
                    'standard_name': standard_code.get('standard_name', entity_text)
                }
            })
        
        return normalized
    
    @staticmethod
    def _get_default_mappings() -> Dict:
        """Get default medical ontology mappings"""
        return {
            'CONDITION:diabetes': {
                'snomed_ct': '73211009',
                'icd10': 'E11',
                'standard_name': 'Type 2 Diabetes Mellitus'
            },
            'CONDITION:hypertension': {
                'snomed_ct': '59621000',
                'icd10': 'I10',
                'standard_name': 'Essential Hypertension'
            },
            'CONDITION:pneumonia': {
                'snomed_ct': '233604007',
                'icd10': 'J15',
                'standard_name': 'Pneumonia'
            },
            'MEDICATION:amoxicillin': {
                'snomed_ct': '27061005',
                'loinc': 'LA21600-7',
                'standard_name': 'Amoxicillin'
            },
            'MEDICATION:ibuprofen': {
                'snomed_ct': '5002',
                'loinc': 'LA18373-7',
                'standard_name': 'Ibuprofen'
            }
        }


class RelationshipMapper:
    """Map extracted relationships to FHIR-compatible structures"""
    
    @staticmethod
    def map_to_observation(text: str, entity: Dict) -> Dict:
        """Map entity to FHIR Observation resource"""
        return {
            'resourceType': 'Observation',
            'status': 'final',
            'code': {
                'coding': [{
                    'system': 'http://snomed.info/sct',
                    'code': entity.get('normalized', {}).get('snomed_ct', ''),
                    'display': entity.get('normalized', {}).get('standard_name', '')
                }]
            },
            'valueString': text,
            'issued': None  # Will be filled by FHIR converter
        }
    
    @staticmethod
    def map_to_condition(text: str, entity: Dict) -> Dict:
        """Map entity to FHIR Condition resource"""
        return {
            'resourceType': 'Condition',
            'code': {
                'coding': [{
                    'system': 'http://hl7.org/fhir/sid/icd-10-cm',
                    'code': entity.get('normalized', {}).get('icd10', ''),
                    'display': entity.get('normalized', {}).get('standard_name', '')
                }]
            },
            'subject': None,  # Will be filled with patient reference
            'recordedDate': None  # Will be filled by FHIR converter
        }
    
    @staticmethod
    def map_to_medication(entity: Dict) -> Dict:
        """Map entity to FHIR Medication resource"""
        return {
            'resourceType': 'Medication',
            'code': {
                'coding': [{
                    'system': 'http://www.nlm.nih.gov/research/umls/rxnorm',
                    'code': '',
                    'display': entity.get('normalized', {}).get('standard_name', '')
                }]
            }
        }

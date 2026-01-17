"""
EMR Digitization Pipeline - Main Package
Convert physical medical records to interoperable FHIR-compliant EMR format
"""

__version__ = "1.0.0"
__author__ = "EMR Digitization Team"

from src.pipeline import EMRDigitizationPipeline
from src.utils.config import ConfigManager
from src.ocr.ocr_wrapper import OCRModelWrapper
from src.post_ocr.text_correction import SpellCorrector, StructuredFieldExtractor
from src.nlp.entity_extraction import ClinicalNLPExtractor
from src.fhir.converter import FHIRValidator, HL7FIRConverter
from src.validation.human_validation import HumanInLoopValidator
from src.security.encryption import DataEncryption, AccessControl

__all__ = [
    'EMRDigitizationPipeline',
    'ConfigManager',
    'OCRModelWrapper',
    'SpellCorrector',
    'StructuredFieldExtractor',
    'ClinicalNLPExtractor',
    'FHIRValidator',
    'HL7FIRConverter',
    'HumanInLoopValidator',
    'DataEncryption',
    'AccessControl'
]

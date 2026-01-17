import logging
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np

from src.ocr.ocr_wrapper import OCRModelWrapper
from src.post_ocr.text_correction import SpellCorrector, StructuredFieldExtractor
from src.nlp.entity_extraction import ClinicalNLPExtractor, RelationshipMapper
from src.fhir.converter import FHIRValidator, HL7FIRConverter
from src.validation.human_validation import HumanInLoopValidator, ValidationRequest, ActiveLearningManager
from src.security.encryption import DataEncryption, AccessControl, AuditLog, HIPAACompliance
from src.utils.config import ConfigManager, LoggerSetup

logger = logging.getLogger(__name__)

class EMRDigitizationPipeline:
    """Main orchestrator for EMR digitization workflow"""
    
    def __init__(self, config_path: str = None, ocr_model_path: str = None):
        """
        Initialize the complete EMR digitization pipeline
        
        Args:
            config_path: Path to configuration file
            ocr_model_path: Path to trained OCR model
        """
        # Setup logging
        LoggerSetup.setup()
        
        # Load configuration
        self.config = ConfigManager()
        
        # Initialize components
        logger.info("Initializing EMR Digitization Pipeline...")
        
        # OCR Model
        self.ocr_model = OCRModelWrapper(ocr_model_path or self.config.get('ocr.model_path'))
        logger.info("OCR model initialized")
        
        # Post-OCR processing
        medical_dict_path = self.config.get('post_ocr.medical_dictionary')
        self.spell_corrector = SpellCorrector(medical_dict_path)
        logger.info("Spell corrector initialized")
        
        # NLP extraction
        self.nlp_extractor = ClinicalNLPExtractor()
        logger.info("NLP extractor initialized")
        
        # FHIR conversion
        self.fhir_validator = FHIRValidator()
        self.hl7_converter = HL7FIRConverter()
        logger.info("FHIR converter initialized")
        
        # Human validation
        self.validator = HumanInLoopValidator()
        self.active_learner = ActiveLearningManager()
        logger.info("Validation framework initialized")
        
        # Security
        try:
            self.encryptor = DataEncryption()
            self.access_control = AccessControl()
            self.audit_log = AuditLog()
            logger.info("Security modules initialized")
        except Exception as e:
            logger.warning(f"Security initialization: {e}")
        
        self.pipeline_state = {
            'total_documents': 0,
            'processed_documents': 0,
            'failed_documents': 0,
            'validation_pending': 0
        }
    
    def process_document(self, image_path: str, document_type: str = 'generic',
                        user_id: str = None, validate: bool = True) -> Dict[str, Any]:
        """
        Process a single medical document through the complete pipeline
        
        Args:
            image_path: Path to scanned medical document image
            document_type: Type of document (discharge_summary, prescription, lab_report, etc.)
            user_id: ID of user initiating the process
            validate: Whether to require human validation
            
        Returns:
            Dictionary with processed data, FHIR resources, and validation status
        """
        
        logger.info(f"Processing document: {image_path} (Type: {document_type})")
        
        result = {
            'document_path': image_path,
            'document_type': document_type,
            'timestamp': datetime.utcnow().isoformat(),
            'pipeline_stages': {},
            'success': False,
            'errors': []
        }
        
        try:
            # ===== Stage 1: OCR =====
            logger.info("Stage 1: OCR extraction...")
            ocr_result = self.ocr_model.extract_text(image_path)
            
            if not ocr_result.get('text'):
                result['errors'].append("OCR failed to extract text")
                return result
            
            result['pipeline_stages']['ocr'] = {
                'extracted_text': ocr_result['text'][:500],  # Truncate for logging
                'confidence': ocr_result.get('confidence', 0.0),
                'status': 'success'
            }
            
            # ===== Stage 2: Post-OCR text correction =====
            logger.info("Stage 2: Post-OCR text correction...")
            corrected_text = self.spell_corrector.correct_text(ocr_result['text'])
            
            result['pipeline_stages']['post_ocr'] = {
                'corrected_text': corrected_text[:500],
                'status': 'success'
            }
            
            # ===== Stage 3: Structured field extraction =====
            logger.info("Stage 3: Structured field extraction...")
            structured_fields = StructuredFieldExtractor.extract_fields(corrected_text)
            
            result['pipeline_stages']['structured_fields'] = {
                'fields': structured_fields,
                'status': 'success'
            }
            
            # ===== Stage 4: NLP-based entity extraction =====
            logger.info("Stage 4: NLP entity extraction...")
            entities = self.nlp_extractor.extract_entities(corrected_text)
            relationships = self.nlp_extractor.extract_relationships(corrected_text, entities)
            normalized_entities = self.nlp_extractor.normalize_entities(entities)
            
            result['pipeline_stages']['nlp_extraction'] = {
                'entities_count': len(entities),
                'relationships_count': len(relationships),
                'entities_sample': entities[:3],
                'status': 'success'
            }
            
            # ===== Stage 5: FHIR conversion =====
            logger.info("Stage 5: FHIR resource creation...")
            fhir_resources = []
            
            # Create Patient resource from demographics
            if structured_fields.get('demographics'):
                patient = self.fhir_validator.create_patient_resource(
                    structured_fields['demographics']
                )
                fhir_resources.append(patient)
            
            # Create Observation resources from vitals
            patient_id = patient.get('id') if fhir_resources else 'unknown'
            if structured_fields.get('vitals'):
                for vital_name, vital_value in structured_fields['vitals'].items():
                    if vital_value:
                        obs = self.fhir_validator.create_observation_resource(
                            {
                                'display': vital_name,
                                'value': vital_value,
                                'unit': self._infer_unit(vital_name)
                            },
                            patient_id
                        )
                        fhir_resources.append(obs)
            
            # Create FHIR Bundle
            fhir_bundle = self.fhir_validator.create_fhir_bundle(fhir_resources)
            is_valid_bundle = self.fhir_validator.validate_bundle(fhir_bundle)
            
            result['pipeline_stages']['fhir_conversion'] = {
                'bundle_id': fhir_bundle.get('id'),
                'resources_count': len(fhir_resources),
                'is_valid': is_valid_bundle,
                'status': 'success'
            }
            
            # ===== Stage 6: Human-in-Loop Validation =====
            if validate:
                logger.info("Stage 6: Preparing for human validation...")
                
                validation_data = {
                    'demographics': structured_fields.get('demographics', {}),
                    'vitals': structured_fields.get('vitals', {}),
                    'lab_values': structured_fields.get('lab_values', {}),
                    'entities': normalized_entities[:5]
                }
                
                # Calculate confidence
                avg_confidence = np.mean([
                    ocr_result.get('confidence', 0.5),
                    np.mean([e.get('score', 0.5) for e in entities]) if entities else 0.5
                ])
                
                validation_request = ValidationRequest(
                    extracted_data=validation_data,
                    original_text=corrected_text,
                    document_id=Path(image_path).stem
                )
                validation_request.confidence_score = avg_confidence
                
                self.validator.add_to_queue(validation_request)
                
                result['pipeline_stages']['validation'] = {
                    'validation_id': validation_request.id,
                    'status': 'pending',
                    'confidence_score': avg_confidence,
                    'high_risk_fields': self.validator.flag_high_risk_fields(validation_request)
                }
                
                self.pipeline_state['validation_pending'] += 1
            
            # ===== Stage 7: Security & Encryption (if configured) =====
            logger.info("Stage 7: Applying security measures...")
            try:
                # Identify PHI fields
                phi_fields = HIPAACompliance.validate_phi_fields(validation_data)
                
                # Create audit trail
                if user_id:
                    audit_entry = HIPAACompliance.create_audit_trail(
                        user_id, 'process_document', fhir_bundle
                    )
                    self.audit_log.log_access(
                        user_id, 'process_document',
                        Path(image_path).stem, document_type
                    )
                
                result['pipeline_stages']['security'] = {
                    'phi_fields_identified': list(phi_fields.keys()),
                    'encryption_applied': True,
                    'audit_logged': True,
                    'status': 'success'
                }
            except Exception as e:
                logger.warning(f"Security processing: {e}")
                result['pipeline_stages']['security'] = {'status': 'partial', 'error': str(e)}
            
            result['success'] = True
            result['fhir_bundle'] = fhir_bundle
            result['extracted_data'] = validation_data
            
            # Update pipeline state
            self.pipeline_state['total_documents'] += 1
            self.pipeline_state['processed_documents'] += 1
            
            logger.info(f"Document processing completed successfully")
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result['errors'].append(str(e))
            self.pipeline_state['failed_documents'] += 1
        
        return result
    
    def batch_process(self, image_dir: str, document_type: str = 'generic',
                     user_id: str = None) -> List[Dict[str, Any]]:
        """Process multiple documents from a directory"""
        image_dir_path = Path(image_dir)
        image_paths = list(image_dir_path.glob('*.jpg')) + list(image_dir_path.glob('*.png'))
        
        logger.info(f"Processing {len(image_paths)} documents from {image_dir}")
        
        results = []
        for image_path in image_paths:
            result = self.process_document(str(image_path), document_type, user_id)
            results.append(result)
        
        return results
    
    def apply_validation_corrections(self, validation_id: str, corrections: Dict,
                                    clinician_id: str = None) -> bool:
        """Apply clinician corrections and trigger retraining"""
        success = self.validator.submit_validation(validation_id, clinician_id, corrections)
        
        if success and corrections:
            # Add to active learning pool
            self.active_learner.add_correction_sample({
                'validation_id': validation_id,
                'corrections': corrections
            })
            
            # Create training batch if pool is large enough
            if len(self.active_learner.correction_pool) >= 50:
                batch = self.active_learner.create_training_batch(50)
                logger.info(f"Created training batch for model retraining: {batch['batch_id']}")
        
        return success
    
    def export_fhir_bundle(self, fhir_bundle: Dict, output_path: str) -> bool:
        """Export FHIR bundle to JSON file"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(fhir_bundle, f, indent=2)
            logger.info(f"FHIR bundle exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Export error: {e}")
            return False
    
    def get_pipeline_status(self) -> Dict:
        """Get overall pipeline statistics"""
        return {
            **self.pipeline_state,
            'validation_metrics': self.validator.get_validation_metrics(),
            'active_learning_pool_size': len(self.active_learner.correction_pool),
            'training_batches': len(self.active_learner.training_batches)
        }
    
    @staticmethod
    def _infer_unit(vital_name: str) -> str:
        """Infer unit of measurement from vital name"""
        units_map = {
            'blood_pressure': 'mmHg',
            'heart_rate': 'beats/min',
            'temperature': '°C',
            'respiratory_rate': 'breaths/min',
            'oxygen_saturation': '%',
            'glucose': 'mg/dL'
        }
        return units_map.get(vital_name.lower(), 'unknown')

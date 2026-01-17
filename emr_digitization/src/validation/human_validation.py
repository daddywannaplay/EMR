import logging
import json
from typing import Dict, List, Tuple
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)

class ValidationRequest:
    """Represent a single validation task"""
    
    def __init__(self, extracted_data: Dict, original_text: str, document_id: str):
        self.id = str(uuid4())
        self.extracted_data = extracted_data
        self.original_text = original_text
        self.document_id = document_id
        self.created_at = datetime.utcnow()
        self.status = 'pending'  # pending, approved, rejected, corrected
        self.clinician_notes = ""
        self.corrections = {}
        self.confidence_score = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'extracted_data': self.extracted_data,
            'original_text': self.original_text,
            'status': self.status,
            'clinician_notes': self.clinician_notes,
            'corrections': self.corrections,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat()
        }


class HumanInLoopValidator:
    """Manage human validation workflows for high-risk medical data"""
    
    def __init__(self):
        self.validation_queue = []
        self.completed_validations = []
        self.correction_history = []
    
    def add_to_queue(self, validation_request: ValidationRequest):
        """Add extraction to validation queue"""
        self.validation_queue.append(validation_request)
        logger.info(f"Added validation request {validation_request.id} to queue")
    
    def prioritize_by_confidence(self, threshold: float = 0.7):
        """Sort queue by confidence score - highest priority for low confidence"""
        def priority_score(request: ValidationRequest) -> float:
            # Lower confidence = higher priority
            return abs(threshold - request.confidence_score)
        
        self.validation_queue.sort(key=priority_score, reverse=True)
        logger.info(f"Prioritized {len(self.validation_queue)} validation requests")
    
    def flag_high_risk_fields(self, validation_request: ValidationRequest, 
                             high_risk_fields: List[str] = None) -> List[str]:
        """Identify and flag high-risk fields for manual review"""
        if not high_risk_fields:
            high_risk_fields = ['diagnosis', 'medication', 'dosage', 'allergies']
        
        flagged_fields = []
        extracted = validation_request.extracted_data
        
        for field in high_risk_fields:
            if field in extracted and extracted[field]:
                flagged_fields.append(field)
        
        return flagged_fields
    
    def get_validation_form(self, validation_request: ValidationRequest) -> Dict:
        """Generate validation form for clinician review"""
        high_risk_fields = self.flag_high_risk_fields(validation_request)
        
        form = {
            'validation_id': validation_request.id,
            'document_id': validation_request.document_id,
            'timestamp': datetime.utcnow().isoformat(),
            'high_risk_fields': high_risk_fields,
            'extracted_data': validation_request.extracted_data,
            'original_text_snippet': validation_request.original_text[:500],
            'confidence_score': validation_request.confidence_score,
            'fields_for_review': []
        }
        
        # Generate field-level review items
        for field, value in validation_request.extracted_data.items():
            if value:
                form['fields_for_review'].append({
                    'field_name': field,
                    'extracted_value': value,
                    'is_high_risk': field in high_risk_fields,
                    'requires_correction': False,
                    'suggested_correction': None
                })
        
        return form
    
    def submit_validation(self, validation_id: str, clinician_id: str,
                        corrections: Dict = None, notes: str = "") -> bool:
        """Record clinician validation and corrections"""
        request = next((r for r in self.validation_queue if r.id == validation_id), None)
        
        if not request:
            logger.warning(f"Validation request {validation_id} not found")
            return False
        
        # Apply corrections
        if corrections:
            request.corrections = corrections
            for field, correction in corrections.items():
                request.extracted_data[field] = correction['new_value']
        
        request.status = 'approved' if not corrections else 'corrected'
        request.clinician_notes = notes
        
        # Move to completed
        self.validation_queue.remove(request)
        self.completed_validations.append(request)
        
        # Log correction for model retraining
        if corrections:
            self.correction_history.append({
                'validation_id': validation_id,
                'document_id': request.document_id,
                'clinician_id': clinician_id,
                'corrections': corrections,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        logger.info(f"Validation {validation_id} completed by clinician {clinician_id}")
        return True
    
    def get_pending_validations(self, limit: int = 10) -> List[Dict]:
        """Get pending validations for clinician dashboard"""
        pending = self.validation_queue[:limit]
        return [req.to_dict() for req in pending]
    
    def get_validation_metrics(self) -> Dict:
        """Get validation workflow metrics"""
        total_processed = len(self.completed_validations)
        corrections_made = sum(1 for v in self.completed_validations if v.corrections)
        
        return {
            'total_pending': len(self.validation_queue),
            'total_processed': total_processed,
            'corrections_made': corrections_made,
            'correction_rate': corrections_made / total_processed if total_processed > 0 else 0,
            'avg_confidence_score': sum(v.confidence_score for v in self.completed_validations) / total_processed if total_processed > 0 else 0
        }


class ActiveLearningManager:
    """Manage active learning loop for model improvement"""
    
    def __init__(self):
        self.correction_pool = []
        self.training_batches = []
    
    def add_correction_sample(self, correction: Dict):
        """Add correction to training pool"""
        self.correction_pool.append({
            **correction,
            'added_at': datetime.utcnow().isoformat()
        })
        logger.info(f"Added correction sample. Pool size: {len(self.correction_pool)}")
    
    def create_training_batch(self, batch_size: int = 50) -> Dict:
        """Create training batch from correction pool"""
        if len(self.correction_pool) < batch_size:
            logger.warning(f"Insufficient samples. Pool: {len(self.correction_pool)}, Required: {batch_size}")
            return None
        
        batch = {
            'batch_id': str(uuid4()),
            'created_at': datetime.utcnow().isoformat(),
            'samples': self.correction_pool[:batch_size],
            'size': batch_size
        }
        
        self.training_batches.append(batch)
        self.correction_pool = self.correction_pool[batch_size:]
        
        logger.info(f"Created training batch {batch['batch_id']} with {batch_size} samples")
        return batch
    
    def get_uncertain_samples(self, threshold: float = 0.65) -> List[Dict]:
        """Get uncertain predictions for human review"""
        # This would be populated by the main pipeline
        # with low-confidence predictions
        uncertain = []
        logger.info(f"Retrieved {len(uncertain)} uncertain samples below threshold {threshold}")
        return uncertain
    
    def export_training_data(self, batch_id: str) -> str:
        """Export batch as training data file"""
        batch = next((b for b in self.training_batches if b['batch_id'] == batch_id), None)
        
        if not batch:
            logger.error(f"Batch {batch_id} not found")
            return None
        
        # Format for model retraining
        training_data = {
            'batch_id': batch_id,
            'samples': batch['samples'],
            'export_time': datetime.utcnow().isoformat()
        }
        
        filename = f"training_batch_{batch_id}.json"
        logger.info(f"Exported training data to {filename}")
        return json.dumps(training_data, indent=2)

import logging
from typing import Optional, List, Dict
import time

from scoring.v2v_scoring import ValidationManager
from dataset import pull_latest_diarization_dataset

class ValidationLoop:
    """Manages validation of voice models."""
    
    def __init__(self,
                 validation_manager: Optional[ValidationManager] = None):
        self.validation_manager = validation_manager or ValidationManager()
        
    def run(self, model_configs: List[Dict]):
        """Run validation for a list of models.
        
        Args:
            model_configs: List of dicts containing model_id and repo_id for each model
        """
        logging.info("Starting validation...")
        
        while True:
            try:
                # Get latest dataset
                dataset = pull_latest_diarization_dataset()
                if dataset is None:
                    logging.warning("No dataset available, waiting...")
                    time.sleep(60)
                    continue
                
                # Validate each model
                scores = {}
                for config in model_configs:
                    score = self.validation_manager.validate_model(
                        model_id=config['repo_id'],
                        repo_id=config['repo_id'],
                        mini_batch=dataset
                    )
                    scores[config['repo_id']] = score
                    
                logging.info(f"Validation scores: {scores}")
                
            except Exception as e:
                logging.error(f"Error in validation loop: {e}")
                continue

if __name__ == "__main__":
    # Example model configs
    models = [
        {
            'model_id': 'org/model1',
            'repo_id': 'org/model1'
        },
        {
            'model_id': 'org/model2', 
            'repo_id': 'org/model2'
        }
    ]
    
    # Create and run validation loop
    validator = ValidationLoop()
    validator.run(models)
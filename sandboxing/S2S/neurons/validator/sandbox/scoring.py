# neurons/validator/sandbox/scoring.py

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)

@dataclass
class ScoringResult:
    """Container for scoring results"""
    metrics: Dict[str, float]
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "metrics": self.metrics,
            "details": self.details,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ScoringResult':
        return cls(
            metrics=data.get("metrics", {}),
            details=data.get("details"),
            error=data.get("error")
        )

class ScoringProtocol:
    """Handles score transmission between sandbox and host"""
    
    @staticmethod
    def serialize_scores(scores: Dict[str, float]) -> bytes:
        """Serialize scores to bytes"""
        result = ScoringResult(metrics=scores)
        return json.dumps(result.to_dict()).encode()

    @staticmethod
    def deserialize_scores(data: bytes) -> ScoringResult:
        """Deserialize scores from bytes"""
        try:
            result_dict = json.loads(data.decode())
            return ScoringResult.from_dict(result_dict)
        except Exception as e:
            logger.error(f"Score deserialization failed: {e}")
            return ScoringResult(metrics={}, error=str(e))

    @staticmethod
    def validate_scores(scores: Dict[str, float]) -> bool:
        """Validate score format and values"""
        required_metrics = {
            "mimi_score",
            "wer_score", 
            "pesq_score",
            "length_penalty",
            "anti_spoofing_score",
            "combined_score"
        }
        
        # Check required metrics
        if not all(metric in scores for metric in required_metrics):
            return False
            
        # Validate ranges
        return all(isinstance(v, (int, float)) and 0 <= v <= 1 
                  for v in scores.values())
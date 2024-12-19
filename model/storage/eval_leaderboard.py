from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker, relationship
from sqlalchemy.exc import OperationalError
from contextlib import contextmanager
from datetime import datetime
import bittensor as bt
from typing import Optional, Dict, List
import time
from vali_api.config import DBHOST, DBNAME, DBUSER, DBPASS

Base = declarative_base()

# Global variables for engine and Session
_engine: Optional[object] = None
Session: Optional[sessionmaker] = None

def init_database():
    """Initialize the database connection and create tables."""
    global _engine, Session
    
    if _engine is not None:
        bt.logging.warning("Database already initialized")
        return

    try:
        connection_string = f'mysql://{DBUSER}:{DBPASS}@{DBHOST}/{DBNAME}'
        _engine = create_engine(connection_string)
        Session = sessionmaker(bind=_engine)
        Base.metadata.create_all(_engine)
        bt.logging.info("Database initialized successfully")
    except Exception as e:
        bt.logging.error(f"Failed to initialize database: {str(e)}")
        raise

def get_session() -> Session:
    """Get a database session."""
    if Session is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return Session()

class EvaluationModel(Base):
    __tablename__ = 'sn21_evals_test'

    eval_id = Column(Integer, primary_key=True)
    miner_hotkey = Column(String(255))
    miner_uid = Column(Integer)
    model_name = Column(String(255))
    model_type = Column(String(255))
    eval_date = Column(DateTime)
    competition_id = Column(String(10))
    
    results = relationship("EvaluationResult", back_populates="evaluation")

class EvaluationResult(Base):
    __tablename__ = 'sn21_eval_results_test'

    eval_result_id = Column(Integer, primary_key=True)
    eval_id = Column(Integer, ForeignKey('sn21_evals_test.eval_id'))
    task = Column(String(255))
    result_name = Column(String(255))
    result = Column(Float)
    competition_id = Column(String(10))

    evaluation = relationship("EvaluationModel", back_populates="results")

class EvalLeaderboardManager:
    def __init__(self, max_retries=3, retry_delay=1):
        if Session is None:
            raise RuntimeError("Database not initialized. Call init_database() first.")
            
        self.session = get_session()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def execute_with_retry(self, operation, *args, **kwargs):
        """Execute database operation with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except OperationalError as e:
                if attempt < self.max_retries - 1:
                    bt.logging.warning(f"Database error. Attempt {attempt + 1}/{self.max_retries}. Retrying...")
                    time.sleep(self.retry_delay)
                else:
                    raise
            except Exception as e:
                bt.logging.error(f"Error executing operation: {str(e)}")
                raise

    def get_v1_leaderboard(self) -> List[Dict]:
        """
        Get turn metrics leaderboard data for competition_id 'v1'.
        Returns IPU, Pause, and Gap Duration metrics.
        """
        def _get_metrics():
            with self.session_scope() as session:
                results = []
                
                # Query evaluations for v1 competition
                evals = session.query(EvaluationModel).filter(
                    EvaluationModel.competition_id == 'v1'
                ).all()
                
                for eval in evals:
                    metrics = {
                        'Miner Model': eval.model_name,
                        'Model': eval.model_type,
                        'IPU': None,
                        'Pause': None,
                        'Gap Duration': None
                    }
                    
                    # Get turn metrics from results
                    turn_results = [r for r in eval.results if r.task == 'turn_metrics']
                    for result in turn_results:
                        if result.result_name == 'mean_ipu_duration':
                            metrics['IPU'] = round(result.result, 2)
                        elif result.result_name == 'mean_pause_duration':
                            metrics['Pause'] = round(result.result, 2)
                        elif result.result_name == 'mean_gap_duration':
                            metrics['Gap Duration'] = round(result.result, 2)
                    
                    # Only add if we have any turn metrics
                    if any(v is not None for v in [metrics['IPU'], metrics['Pause'], metrics['Gap Duration']]):
                        results.append(metrics)
                
                return results

        try:
            return self.execute_with_retry(_get_metrics)
        except Exception as e:
            bt.logging.error(f"Failed to get v1 leaderboard: {str(e)}")
            return []

    def close(self):
        """Safely close the session."""
        try:
            self.session.close()
        except:
            pass
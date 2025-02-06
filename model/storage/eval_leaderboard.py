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
    deleted_at = Column(DateTime)

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

    
    def get_metrics_timeseries(self) -> Dict[str, List[Dict]]:
        """
        Get time series data for all turn metrics.
        Returns data in format {metric_name: [{date: "", models: [{modelType: "", modelName: "", score: 123}, ...]}, ...]}
        Groups models by model_type and includes model names in the data.
        """
        def _get_timeseries():
            with self.session_scope() as session:
                evals = session.query(EvaluationModel).order_by(EvaluationModel.eval_date).all()
            
                metrics = set()
                # Get all non-deleted results that aren't the excluded metric
                sample_results = session.query(EvaluationResult).filter(
                    EvaluationResult.result_name != "exact_match_stderr,flexible-extract",
                    EvaluationResult.deleted_at.is_(None)  # SQLAlchemy's proper way to check for NULL
                ).all()
                for result in sample_results:
                    metric_name = result.result_name
                    metrics.add(metric_name)
                timeseries_data = {metric: {} for metric in metrics}
            
                for eval_ in evals:
                    date_str = eval_.eval_date.strftime('%Y-%m-%d')
                    model_type = eval_.model_type
                    model_name = eval_.model_name
                    competition_id = eval_.competition_id
                
                    # Only process non-deleted results
                    for result in [r for r in eval_.results if r.deleted_at is None]:
                        metric_name = result.result_name
                        if metric_name in metrics:
                            if date_str not in timeseries_data[metric_name]:
                                timeseries_data[metric_name][date_str] = {
                                    'date': date_str,
                                    'models': []
                                }
                            
                            # Check if we already have an entry for this model_type
                            existing_model = next(
                                (m for m in timeseries_data[metric_name][date_str]['models'] 
                                 if m['modelType'] == model_type),
                                None
                            )
                            
                            if existing_model:
                                # Update existing entry
                                existing_model['score'] = result.result
                                if model_name not in existing_model['modelName']:
                                    existing_model['modelName'] = model_name
                            else:
                                # Add new entry
                                timeseries_data[metric_name][date_str]['models'].append({
                                    'modelType': model_type,
                                    'modelName': model_name,
                                    'score': result.result,
                                    'competition_id': competition_id,
                                    'task': result.task
                                })
            
                return {
                    metric: [
                        data_point for data_point in sorted(data.values(), key=lambda x: x['date'])
                        if data_point['models']  # Filter out entries with empty models list
                    ]
                    for metric, data in timeseries_data.items()
                }
        
        try:
            return self.execute_with_retry(_get_timeseries)
        except Exception as e:
            bt.logging.error(f"Failed to get metrics timeseries: {str(e)}")
            return {}


    def close(self):
        """Safely close the session."""
        try:
            self.session.close()
        except:
            pass
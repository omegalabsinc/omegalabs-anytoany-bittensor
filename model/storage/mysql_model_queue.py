from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON, func, desc, exists, ForeignKey, or_, and_
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker, aliased, relationship
from contextlib import contextmanager
from collections import defaultdict

import time
import json

from datetime import datetime, timedelta
import bittensor as bt
from typing import Optional

from model.data import ModelId
from vali_api.config import DBHOST, DBNAME, DBUSER, DBPASS

Base = declarative_base()

# Global variables for engine and Session
_engine: Optional[object] = None
Session: Optional[sessionmaker] = None

def init_database():
    """
    Initialize the database connection and create tables.
    Must be called before using any database operations.
    """
    global _engine, Session
    
    if _engine is not None:
        bt.logging.warning("Database already initialized")
        return

    try:
        connection_string = f'mysql://{DBUSER}:{DBPASS}@{DBHOST}/{DBNAME}'
        _engine = create_engine(connection_string)
        Session = sessionmaker(bind=_engine)
        
        # Create all tables
        Base.metadata.create_all(_engine)
        bt.logging.info("Database initialized successfully")
        
    except Exception as e:
        bt.logging.error(f"Failed to initialize database: {str(e)}")
        raise

def get_session() -> Session:
    """
    Get a database session. Raises exception if database not initialized.
    """
    if Session is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return Session()

class ModelQueue(Base):
    __tablename__ = 'sn21_model_queue'

    hotkey = Column(String(255), primary_key=True)
    uid = Column(String(255), primary_key=True, index=True)
    block = Column(Integer, index=True)
    competition_id = Column(String(255), index=True)
    model_metadata = Column(JSON)
    is_new = Column(Boolean, default=True)
    is_being_scored = Column(Boolean, default=False)
    is_being_scored_by = Column(String(255), default=None)
    scoring_updated_at = Column(DateTime, default=None)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to ScoreHistory
    scores = relationship("ScoreHistory", 
                          back_populates="model", 
                          foreign_keys="[ScoreHistory.hotkey, ScoreHistory.uid]",
                          primaryjoin="and_(ModelQueue.hotkey==ScoreHistory.hotkey, ModelQueue.uid==ScoreHistory.uid)")

    def __repr__(self):
        return f"<ModelQueue(hotkey='{self.hotkey}', uid='{self.uid}', competition_id='{self.competition_id}', is_new={self.is_new})>"

class ScoreHistory(Base):
    __tablename__ = 'sn21_score_history'

    id = Column(Integer, primary_key=True)
    hotkey = Column(String(255), ForeignKey('sn21_model_queue.hotkey', ondelete='SET NULL'), index=True, nullable=True)
    uid = Column(String(255), ForeignKey('sn21_model_queue.uid', ondelete='SET NULL'), index=True, nullable=True)
    competition_id = Column(String(255), index=True)
    model_metadata = Column(JSON)
    score = Column(Float)
    scored_at = Column(DateTime, default=datetime.utcnow)
    block = Column(Integer)
    model_hash = Column(String(255))
    scorer_hotkey = Column(String(255), index=True)
    is_archived = Column(Boolean, default=False)

    # Relationship to ModelQueue
    model = relationship("ModelQueue", 
                        back_populates="scores", 
                        foreign_keys=[hotkey, uid],
                        primaryjoin="and_(ModelQueue.hotkey==ScoreHistory.hotkey, ModelQueue.uid==ScoreHistory.uid)")

    def __repr__(self):
        return f"<ScoreHistory(hotkey='{self.hotkey}', uid='{self.uid}', score={self.score}, scored_at={self.scored_at}, model_metadata={self.model_metadata} is_archived={self.is_archived})>"

class ModelIdEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ModelId):
            return {
                'namespace': obj.namespace,
                'name': obj.name,
                'epoch': obj.epoch,
                'commit': obj.commit,
                'hash': obj.hash,
                'competition_id': obj.competition_id
            }
        return super().default(obj)

class ModelQueueManager:
    def __init__(self, max_scores_per_model=5, rescore_interval_hours=24, max_retries=3, retry_delay=1):
        if Session is None:
            raise RuntimeError("Database not initialized. Call init_database() first.")

        self.session = get_session()
        self.max_scores_per_model = max_scores_per_model
        self.rescore_interval = timedelta(hours=rescore_interval_hours)
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

    def reset_session(self):
        """Reset the session in case of connection issues."""
        try:
            self.session.close()
        except:
            pass
        try:
            self.session = get_session()
        except RuntimeError as e:
            bt.logging.error(f"Failed to reset session: {str(e)}")
            raise

    def execute_with_retry(self, operation, *args, **kwargs):
        """Execute an operation with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except OperationalError as e:
                if "Lost connection" in str(e) and attempt < self.max_retries - 1:
                    bt.logging.warning(f"Lost connection to MySQL. Attempt {attempt + 1}/{self.max_retries}. Retrying...")
                    self.reset_session()
                    time.sleep(self.retry_delay)
                else:
                    raise
            except SQLAlchemyError as e:
                if attempt < self.max_retries - 1:
                    bt.logging.warning(f"Database error. Attempt {attempt + 1}/{self.max_retries}. Retrying...")
                    self.reset_session()
                    time.sleep(self.retry_delay)
                else:
                    raise

    def store_updated_model(self, uid, hotkey, model_metadata, updated):
        """
        Store or update model metadata with retry logic.
        
        Args:
            uid (str): Model UID
            hotkey (str): Model hotkey
            model_metadata: Model metadata object
            updated (bool): Whether this is an update
            
        Returns:
            bool: Success status
        """
        def _store_model():
            with self.session_scope() as session:
                try:
                    # Query existing model with lock
                    existing_model = session.query(ModelQueue).filter_by(
                        hotkey=hotkey, 
                        uid=uid
                    ).with_for_update().first()

                    # Serialize metadata
                    serialized_metadata = json.dumps(model_metadata.__dict__, cls=ModelIdEncoder)

                    if existing_model:
                        if updated:
                            bt.logging.debug(f"Updating model metadata for UID={uid}, Hotkey={hotkey}")
                            existing_model.model_metadata = serialized_metadata
                            existing_model.is_new = updated
                            existing_model.block = model_metadata.block
                            existing_model.updated_at = datetime.utcnow()
                    else:
                        # Create new model entry
                        new_model = ModelQueue(
                            hotkey=hotkey,
                            uid=uid,
                            competition_id=model_metadata.id.competition_id,
                            model_metadata=serialized_metadata,
                            is_new=updated,
                            block=model_metadata.block
                        )
                        session.add(new_model)

                    bt.logging.debug(f"Stored model for UID={uid}, Hotkey={hotkey} in database. Is new = {updated}")
                    return True

                except Exception as e:
                    bt.logging.error(f"Error in _store_model: {str(e)}")
                    bt.logging.error(f"Model metadata: {model_metadata}")
                    raise

        try:
            return self.execute_with_retry(_store_model)
        except Exception as e:
            bt.logging.error(f"Failed to store model after {self.max_retries} attempts: {str(e)}")
            return False

    def get_next_model_to_score(self):
        """Get next model to score with retry logic."""
        def _get_next_model():
            with self.session_scope() as session:
                try:
                    now = datetime.utcnow()
                    subquery = session.query(
                        ScoreHistory.hotkey,
                        ScoreHistory.uid,
                        func.count(ScoreHistory.id).label('score_count'),
                        func.max(ScoreHistory.scored_at).label('last_scored_at')
                    ).filter(
                        ScoreHistory.is_archived == False
                    ).group_by(
                        ScoreHistory.hotkey, 
                        ScoreHistory.uid
                    ).subquery()

                    next_model = session.query(ModelQueue).outerjoin(
                        subquery, 
                        and_(
                            ModelQueue.hotkey == subquery.c.hotkey, 
                            ModelQueue.uid == subquery.c.uid
                        )
                    ).filter(
                        ModelQueue.is_being_scored == False,
                        or_(
                            subquery.c.score_count == None,
                            and_(
                                subquery.c.score_count < self.max_scores_per_model,
                                or_(
                                    subquery.c.last_scored_at == None,
                                    subquery.c.last_scored_at < now - self.rescore_interval
                                )
                            )
                        )
                    ).order_by(
                        desc(ModelQueue.is_new),
                        (subquery.c.score_count == None).desc(),
                        subquery.c.score_count.asc(),
                        func.rand()
                    ).with_for_update().first()

                    if next_model:
                        # Create a dictionary with the model's attributes
                        model_data = {
                            'hotkey': next_model.hotkey,
                            'uid': next_model.uid,
                            'block': next_model.block,
                            'competition_id': next_model.competition_id,
                            'model_metadata': next_model.model_metadata,
                            'is_new': next_model.is_new,
                            'is_being_scored': next_model.is_being_scored,
                            'is_being_scored_by': next_model.is_being_scored_by,
                            'scoring_updated_at': next_model.scoring_updated_at,
                            'updated_at': next_model.updated_at
                        }
                        bt.logging.debug(f"Found next model to score: hotkey={model_data['hotkey']}, uid={model_data['uid']}")
                        return model_data
                    else:
                        bt.logging.debug("No models available for scoring")
                        return None

                except Exception as e:
                    bt.logging.error(f"Error in _get_next_model: {str(e)}")
                    raise

        try:
            return self.execute_with_retry(_get_next_model)
        except Exception as e:
            bt.logging.error(f"Failed to get next model after {self.max_retries} attempts: {str(e)}")
            return None

    def mark_model_as_being_scored(self, model_hotkey, model_uid, scorer_hotkey):
        """Mark model as being scored with retry logic."""
        def _mark_model():
            with self.session_scope() as session:
                model = session.query(ModelQueue).filter_by(
                    hotkey=model_hotkey, 
                    uid=model_uid
                ).with_for_update().first()
                
                if model and not model.is_being_scored:
                    model.is_being_scored = True
                    model.is_being_scored_by = scorer_hotkey
                    model.scoring_updated_at = datetime.utcnow()
                    return True
                return False

        try:
            return self.execute_with_retry(_mark_model)
        except Exception as e:
            bt.logging.error(f"Failed to mark model as being scored after {self.max_retries} attempts: {str(e)}")
            return False

    def submit_score(self, model_hotkey, model_uid, scorer_hotkey, model_hash, score):
        """Submit score with retry logic."""
        def _submit_score():
            with self.session_scope() as session:
                try:
                    model = session.query(ModelQueue).filter_by(
                        hotkey=model_hotkey, 
                        uid=model_uid
                    ).with_for_update().first()
                    
                    if not model:
                        bt.logging.error(f"No model found for hotkey {model_hotkey} and uid {model_uid}")
                        return False

                    existing_scores = session.query(ScoreHistory).filter_by(
                        hotkey=model_hotkey, 
                        uid=model_uid
                    ).count()

                    # temporarily allow scoring from any hotkey
                    new_score = ScoreHistory(
                        hotkey=model_hotkey,
                        uid=model_uid,
                        competition_id=model.competition_id,
                        score=score,
                        block=model.block,
                        model_hash=model_hash,
                        scorer_hotkey=scorer_hotkey,
                        model_metadata=model.model_metadata 
                    )
                    session.add(new_score)
                    model.is_being_scored = False
                    model.is_being_scored_by = None
                    model.scoring_updated_at = None
                    bt.logging.info(f"Successfully submitted score for model {model_hotkey} by {scorer_hotkey}")
                    return True
                    """
                    if existing_scores == 0 or (model.is_being_scored and model.is_being_scored_by == scorer_hotkey):
                        new_score = ScoreHistory(
                            hotkey=model_hotkey,
                            uid=model_uid,
                            competition_id=model.competition_id,
                            score=score,
                            block=model.block,
                            model_hash=model_hash,
                            scorer_hotkey=scorer_hotkey,
                            model_metadata=model.model_metadata 
                        )
                        session.add(new_score)
                        model.is_being_scored = False
                        model.is_being_scored_by = None
                        model.scoring_updated_at = None
                        bt.logging.info(f"Successfully submitted score for model {model_hotkey} by {scorer_hotkey}")
                        return True
                    else:
                        bt.logging.error(f"Failed to submit score for model {model_hotkey} by {scorer_hotkey}. "
                                    f"Model: {model}, is_being_scored: {model.is_being_scored}, "
                                    f"is_being_scored_by: {model.is_being_scored_by}, "
                                    f"existing_scores: {existing_scores}")
                        return False
                    """

                except Exception as e:
                    bt.logging.error(f"Error in _submit_score: {str(e)}")
                    raise

        try:
            return self.execute_with_retry(_submit_score)
        except Exception as e:
            bt.logging.error(f"Failed to submit score after {self.max_retries} attempts: {str(e)}")
            return False

    def reset_stale_scoring_tasks(self, max_scoring_time_minutes=10):
        """Reset stale scoring tasks with retry logic."""
        def _reset_stale_tasks():
            with self.session_scope() as session:
                try:
                    stale_time = datetime.utcnow() - timedelta(minutes=max_scoring_time_minutes)
                    stale_models = session.query(ModelQueue).filter(
                        ModelQueue.is_being_scored == True,
                        ModelQueue.scoring_updated_at < stale_time
                    ).with_for_update().all()

                    reset_count = 0
                    for model in stale_models:
                        model.is_being_scored = False
                        model.is_being_scored_by = None
                        model.scoring_updated_at = None
                        reset_count += 1
                        bt.logging.info(f"Reset scoring task for stale model: hotkey={model.hotkey}, uid={model.uid}")

                    return reset_count

                except Exception as e:
                    bt.logging.error(f"Error in _reset_stale_tasks: {str(e)}")
                    raise

        try:
            return self.execute_with_retry(_reset_stale_tasks)
        except Exception as e:
            bt.logging.error(f"Failed to reset stale tasks after {self.max_retries} attempts: {str(e)}")
            return 0
    
    def get_all_model_scores(self):
        """Get all model scores with retry logic."""
        def _get_all_scores():
            with self.session_scope() as session:
                try:
                    # First, get the latest score timestamps
                    latest_scores = session.query(
                        ScoreHistory.hotkey,
                        ScoreHistory.uid,
                        func.max(ScoreHistory.scored_at).label('latest_score_time')
                    ).filter(
                        ScoreHistory.is_archived == False
                    ).group_by(
                        ScoreHistory.hotkey, 
                        ScoreHistory.uid
                    ).subquery('latest_scores')

                    # Get score details
                    latest_score_details = session.query(
                        ScoreHistory
                    ).join(
                        latest_scores,
                        and_(
                            ScoreHistory.hotkey == latest_scores.c.hotkey,
                            ScoreHistory.uid == latest_scores.c.uid,
                            ScoreHistory.scored_at == latest_scores.c.latest_score_time
                        )
                    ).subquery('latest_score_details')

                    # Get final results
                    results = session.query(
                        ModelQueue.uid,
                        ModelQueue.hotkey,
                        ModelQueue.competition_id,
                        latest_score_details.c.score,
                        latest_score_details.c.scored_at,
                        latest_score_details.c.block,
                        latest_score_details.c.model_hash,
                        latest_score_details.c.scorer_hotkey
                    ).outerjoin(
                        latest_score_details,
                        and_(
                            ModelQueue.hotkey == latest_score_details.c.hotkey,
                            ModelQueue.uid == latest_score_details.c.uid
                        )
                    ).all()

                    scores_by_uid = defaultdict(list)
                    for result in results:
                        if result.score is not None:
                            scores_by_uid[result.uid].append({
                                'hotkey': result.hotkey,
                                'competition_id': result.competition_id,
                                'score': result.score,
                                'scored_at': result.scored_at.isoformat() if result.scored_at else None,
                                'block': result.block,
                                'model_hash': result.model_hash,
                            })
                        else:
                            scores_by_uid[result.uid].append({
                                'hotkey': result.hotkey,
                                'competition_id': result.competition_id,
                                'score': None,
                                'scored_at': None,
                                'block': None,
                                'model_hash': None,
                            })

                    return dict(scores_by_uid)

                except Exception as e:
                    bt.logging.error(f"Error in _get_all_scores: {str(e)}")
                    raise

        try:
            return self.execute_with_retry(_get_all_scores)
        except Exception as e:
            bt.logging.error(f"Failed to get all scores after {self.max_retries} attempts: {str(e)}")
            return {}

    def archive_scores_for_deregistered_models(self, registered_hotkey_uid_pairs):
        """Archive deregistered models with retry logic."""
        def _archive_scores():
            with self.session_scope() as session:
                try:
                    all_models = session.query(
                        ModelQueue.hotkey, 
                        ModelQueue.uid
                    ).with_for_update().all()

                    deregistered_models = set(
                        (model.hotkey, model.uid) for model in all_models
                    ) - set(registered_hotkey_uid_pairs)

                    for hotkey, uid in deregistered_models:
                        # Mark scores as archived
                        archive_result = session.query(ScoreHistory).filter_by(
                            hotkey=hotkey,
                            uid=uid,
                            is_archived=False
                        ).update(
                            {"is_archived": True},
                            synchronize_session=False
                        )

                        # Remove from ModelQueue
                        delete_result = session.query(ModelQueue).filter_by(
                            hotkey=hotkey,
                            uid=uid
                        ).delete(synchronize_session=False)

                        bt.logging.debug(
                            f"Processed deregistered model - Hotkey: {hotkey}, "
                            f"UID: {uid}, Archived scores: {archive_result}, "
                            f"Removed from queue: {delete_result}"
                        )

                    return len(deregistered_models)

                except Exception as e:
                    bt.logging.error(f"Error in _archive_scores: {str(e)}")
                    raise

        try:
            result = self.execute_with_retry(_archive_scores)
            print(f"Archived scores and removed {result} deregistered models from the queue.")
            return result
        except Exception as e:
            bt.logging.error(f"Failed to archive scores after {self.max_retries} attempts: {str(e)}")
            return 0

    def close(self):
        """Safely close the session."""
        try:
            self.session.close()
        except:
            pass

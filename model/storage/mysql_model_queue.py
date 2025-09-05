from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON, func, desc, exists, ForeignKey, or_, and_, case, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
from sqlalchemy.orm import Session, sessionmaker, aliased, relationship
from contextlib import contextmanager
from collections import defaultdict

import time
import json

from datetime import datetime, timedelta, timezone
import bittensor as bt
from typing import Optional

from model.data import ModelId
from vali_api.config import DBHOST, DBNAME, DBUSER, DBPASS, IS_PROD

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

    # Try different MySQL drivers in order of preference
    drivers_to_try = [
        ('mysql', 'mysqlclient (MySQLdb)'),
        ('mysql+pymysql', 'PyMySQL')
    ]
    
    for driver, driver_name in drivers_to_try:
        try:
            connection_string = f'{driver}://{DBUSER}:{DBPASS}@{DBHOST}/{DBNAME}'
            bt.logging.info(f"Attempting database connection with {driver_name}")
            
            _engine = create_engine(connection_string)
            Session = sessionmaker(bind=_engine)
            
            # Test the connection
            with Session() as session:
                session.execute(text('SELECT 1'))
            
            # Create all tables
            Base.metadata.create_all(_engine)
            bt.logging.info(f"Database initialized successfully with {driver_name}")
            return
            
        except ImportError as e:
            bt.logging.warning(f"Driver {driver_name} not available: {e}")
            continue
        except Exception as e:
            bt.logging.error(f"Failed to connect with {driver_name}: {e}")
            if driver == drivers_to_try[-1][0]:  # Last driver in list
                raise
            continue
    
    raise RuntimeError("Failed to initialize database with any available MySQL driver")

def get_session() -> Session:
    """
    Get a database session. Raises exception if database not initialized.
    """
    if Session is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return Session()

def get_table_name(base_name: str) -> str:
    """Helper function to get the correct table name with suffix if not in production."""
    return f"{base_name}{'_test' if not IS_PROD else ''}"

class ModelQueue(Base):
    __tablename__ = get_table_name('sn21_model_queue')

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

    # Relationship to use dynamic table name (lambda function)
    scores = relationship(
        "ScoreHistory",
        back_populates="model",
        foreign_keys="[ScoreHistory.hotkey, ScoreHistory.uid]",
        primaryjoin=lambda: and_(
            ModelQueue.hotkey == ScoreHistory.hotkey,
            ModelQueue.uid == ScoreHistory.uid
        )
    )

    def __repr__(self):
        return f"<ModelQueue(hotkey='{self.hotkey}', uid='{self.uid}', competition_id='{self.competition_id}', is_new={self.is_new})>"

class ScoreHistory(Base):
    __tablename__ = get_table_name('sn21_score_history')

    id = Column(Integer, primary_key=True)
    hotkey = Column(String(255), ForeignKey(f"{get_table_name('sn21_model_queue')}.hotkey", ondelete='SET NULL'), index=True, nullable=True)
    uid = Column(String(255), ForeignKey(f"{get_table_name('sn21_model_queue')}.uid", ondelete='SET NULL'), index=True, nullable=True)
    competition_id = Column(String(255), index=True)
    model_metadata = Column(JSON)
    score = Column(Float)
    scored_at = Column(DateTime, default=datetime.utcnow)
    block = Column(Integer)
    model_hash = Column(String(255))
    scorer_hotkey = Column(String(255), index=True)
    is_archived = Column(Boolean, default=False)
    metric_scores = Column(MySQLJSON, nullable=True)
    wandb_run_id = Column(String(255), nullable=True)
    wandb_run_url = Column(String(512), nullable=True)
    # Relationship to ModelQueue using dynamic table name (lambda function)
    model = relationship(
        "ModelQueue",
        back_populates="scores",
        foreign_keys=[hotkey, uid],
        primaryjoin=lambda: and_(
            ModelQueue.hotkey == ScoreHistory.hotkey,
            ModelQueue.uid == ScoreHistory.uid
        )
    )

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
                        if existing_model.model_metadata != serialized_metadata or existing_model.block != model_metadata.block:
                            bt.logging.debug(f"Updating existing model metadata for UID={uid}, Hotkey={hotkey}. Old metadata: {existing_model.model_metadata}, New metadata: {serialized_metadata}")
                            existing_model.model_metadata = serialized_metadata
                            existing_model.is_new = True
                            existing_model.block = model_metadata.block
                            existing_model.updated_at = datetime.utcnow()
                    else:
                        # Create new model entry
                        new_model = ModelQueue(
                            hotkey=hotkey,
                            uid=uid,
                            competition_id=model_metadata.id.competition_id,
                            model_metadata=serialized_metadata,
                            is_new=True,
                            block=model_metadata.block
                        )
                        session.add(new_model)
                        bt.logging.debug(f"Stored new model for UID={uid}, Hotkey={hotkey} in database. Is new = {updated}")

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

    def get_next_model_to_score(self, competition_id: str):
        """
        Get next model to score with retry logic.
        
        The updated prioritization logic ensures:
        1. New models (highest priority)
        2. Models never scored with non-zero scores
        3a. High-scoring models not scored for over a week
        3b. Models not scored for more than 7 days (safety net for winning models)
        4. Models eligible by standard criteria (not scored in 5 days or < 5 scores)
        5. Everything else (lowest priority)
        
        Zero-scored models that are frequently scored are downgraded in priority
        to prevent them from consuming too many resources.
        """
        def _get_next_model():
            with self.session_scope() as session:
                try:
                    now = datetime.utcnow()

                    # ---- START: Query to find overall highest score ----
                    overall_max_score_value = session.query(func.max(ScoreHistory.score)).filter(
                        ScoreHistory.competition_id == competition_id,
                        ScoreHistory.is_archived == False,
                        ScoreHistory.score > 0  # Consider only positive scores as relevant for "highest"
                    ).scalar()

                    if overall_max_score_value is not None:
                        bt.logging.info(f"Overall highest positive score in competition '{competition_id}' is: {overall_max_score_value:.4f}")
                    else:
                        bt.logging.info(f"No positive scores found for competition '{competition_id}' to determine an overall highest score.")
                    # ---- END: Query ----
                    
                    # Get latest score timestamp and count for each model
                    score_subquery = session.query(
                        ScoreHistory.hotkey,
                        ScoreHistory.uid,
                        func.count(ScoreHistory.id).label('score_count'),
                        func.max(ScoreHistory.scored_at).label('latest_scored_at'),  # Get the latest score timestamp
                        func.max(ScoreHistory.score).label('max_score')  # Get the maximum score
                    ).filter(
                        ScoreHistory.is_archived == False,
                        ScoreHistory.competition_id == competition_id,
                        ScoreHistory.score > 0  # Only consider non-zero scores
                    ).group_by(
                        ScoreHistory.hotkey, 
                        ScoreHistory.uid
                    ).subquery()

                    # Also track all scores (including zeros) for high-frequency zero score detection
                    all_scores_subquery = session.query(
                        ScoreHistory.hotkey,
                        ScoreHistory.uid,
                        func.count(ScoreHistory.id).label('all_score_count'),
                        func.max(ScoreHistory.scored_at).label('latest_all_scored_at'),
                        func.sum(case((ScoreHistory.score > 0, 1), else_=0)).label('non_zero_count')
                    ).filter(
                        ScoreHistory.is_archived == False,
                        ScoreHistory.competition_id == competition_id
                    ).group_by(
                        ScoreHistory.hotkey, 
                        ScoreHistory.uid
                    ).subquery()

                    five_days_ago = now - timedelta(days=5)
                    weekly_rescore_threshold_time = now - timedelta(days=7)  # Define a 7-day threshold
                    
                    # Check if we have new models before proceeding
                    have_new_models = session.query(ModelQueue).filter(
                        ModelQueue.is_being_scored == False,
                        ModelQueue.competition_id == competition_id,
                        ModelQueue.is_new == True
                    ).first() is not None
                    
                    # Check if we have never-scored models
                    never_scored_count = session.query(func.count(ModelQueue.uid)).filter(
                        ModelQueue.is_being_scored == False,
                        ModelQueue.competition_id == competition_id,
                        ~exists().where(
                            and_(
                                ScoreHistory.hotkey == ModelQueue.hotkey,
                                ScoreHistory.uid == ModelQueue.uid,
                                ScoreHistory.score > 0
                            )
                        )
                    ).scalar()
                    
                    # If no new models and no never-scored models, prioritize high scoring models not scored recently
                    if not have_new_models:
                        # ---- START: Modified logic for dynamic high-score threshold ----
                        if overall_max_score_value is not None and overall_max_score_value > 0: # Ensure we have a valid max score
                            dynamic_high_score_threshold = overall_max_score_value * 0.97
                            bt.logging.info(f"Using dynamic high-score threshold for competition '{competition_id}': >= {dynamic_high_score_threshold:.4f} (based on overall max of {overall_max_score_value:.4f})")

                            # First try to get a high-scoring model not scored in over a week
                            top_model = session.query(ModelQueue).join(
                                score_subquery,
                                and_(
                                    ModelQueue.hotkey == score_subquery.c.hotkey,
                                    ModelQueue.uid == score_subquery.c.uid
                                )
                            ).filter(
                                ModelQueue.is_being_scored == False,
                                ModelQueue.competition_id == competition_id,
                                score_subquery.c.latest_scored_at < weekly_rescore_threshold_time,
                                score_subquery.c.max_score >= dynamic_high_score_threshold  # Use dynamic threshold
                            ).order_by(
                                score_subquery.c.max_score.desc()  # Highest score first
                            ).with_for_update().first()
                            
                            if top_model:
                                # Create a dictionary with the model's attributes
                                model_data = {
                                    'hotkey': top_model.hotkey,
                                    'uid': top_model.uid,
                                    'block': top_model.block,
                                    'competition_id': top_model.competition_id,
                                    'model_metadata': top_model.model_metadata,
                                    'is_new': top_model.is_new,
                                    'is_being_scored': top_model.is_being_scored,
                                    'is_being_scored_by': top_model.is_being_scored_by,
                                    'scoring_updated_at': top_model.scoring_updated_at,
                                    'updated_at': top_model.updated_at
                                }
                                bt.logging.debug(f"Found high-scoring model (dynamic threshold) to score: hotkey={model_data['hotkey']}, uid={model_data['uid']}")
                                return model_data
                        else:
                            bt.logging.info(f"Skipping dynamic high-score prioritization for competition '{competition_id}' as no overall positive max score is available or it's zero.")
                        # ---- END: Modified logic ----
                    
                    # Otherwise, use the standard prioritization logic with the zero-score detection
                    next_model = session.query(ModelQueue).outerjoin(
                        score_subquery, 
                        and_(
                            ModelQueue.hotkey == score_subquery.c.hotkey, 
                            ModelQueue.uid == score_subquery.c.uid
                        )
                    ).outerjoin(
                        all_scores_subquery,
                        and_(
                            ModelQueue.hotkey == all_scores_subquery.c.hotkey,
                            ModelQueue.uid == all_scores_subquery.c.uid
                        )
                    ).filter(
                        ModelQueue.is_being_scored == False,
                        ModelQueue.competition_id == competition_id
                    ).order_by(
                        desc(ModelQueue.is_new),  # 1. Prioritize new models
                        (score_subquery.c.score_count == None).desc(),  # 2. Prioritize models never scored (non-zero)
                        case( # 3. Prioritize models not scored for more than 7 days (safety net)
                            (and_(score_subquery.c.latest_scored_at != None, score_subquery.c.latest_scored_at < weekly_rescore_threshold_time), 0),
                            else_=1
                        ),
                        # 4. Decrease priority for models with all zero scores and frequent scoring
                        case(
                            (and_(
                                all_scores_subquery.c.all_score_count > 10,  # Has many scores
                                all_scores_subquery.c.non_zero_count == 0,   # All scores are zero
                                all_scores_subquery.c.latest_all_scored_at > five_days_ago  # Scored recently
                            ), 1),
                            else_=0
                        ),
                        case( # 5. Prioritize models eligible by standard criteria
                            (or_(
                                score_subquery.c.latest_scored_at == None,
                                score_subquery.c.latest_scored_at <= five_days_ago,
                                score_subquery.c.score_count < 5
                            ), 0),
                            else_=1
                        ),
                        func.rand()  # 6. Random tie-breaker
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

    def submit_score(self, model_hotkey, model_uid, scorer_hotkey, model_hash, score, metric_scores):
        """Submit score with retry logic. Mark the model in queue as scored. Remove from queue."""
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

                    """
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
                    model.is_new = False
                    model.is_being_scored = False
                    model.is_being_scored_by = None
                    model.scoring_updated_at = None
                    model.updated_at = datetime.now(timezone.utc)
                    bt.logging.info(f"Successfully submitted score for model {model_hotkey} by {scorer_hotkey}")
                    return True
                    """
                    
                    if model.is_being_scored and model.is_being_scored_by == scorer_hotkey:
                        # Extract wandb fields from metric_scores if present
                        wandb_run_id = None
                        wandb_run_url = None
                        if metric_scores and isinstance(metric_scores, dict):
                            wandb_run_id = metric_scores.get('wandb_run_id')
                            wandb_run_url = metric_scores.get('wandb_run_url')
                        
                        new_score = ScoreHistory(
                            hotkey=model_hotkey,
                            uid=model_uid,
                            competition_id=model.competition_id,
                            score=score,
                            block=model.block,
                            model_hash=model_hash,
                            scorer_hotkey=scorer_hotkey,
                            model_metadata=model.model_metadata,
                            metric_scores=metric_scores,
                            wandb_run_id=wandb_run_id,
                            wandb_run_url=wandb_run_url
                        )
                        session.add(new_score)
                        model.is_new = False
                        model.is_being_scored = False
                        model.is_being_scored_by = None
                        model.scoring_updated_at = None
                        model.updated_at = datetime.now(timezone.utc)
                        bt.logging.info(f"Successfully submitted score for model {model_hotkey} by {scorer_hotkey}")
                        return True
                    else:
                        bt.logging.error(f"Failed to submit score for model {model_hotkey} by {scorer_hotkey}. "
                                    f"Model: {model}, is_being_scored: {model.is_being_scored}, "
                                    f"is_being_scored_by: {model.is_being_scored_by}")
                        return False

                except Exception as e:
                    bt.logging.error(f"Error in _submit_score: {str(e)}")
                    raise

        try:
            return self.execute_with_retry(_submit_score)
        except Exception as e:
            bt.logging.error(f"Failed to submit score after {self.max_retries} attempts: {str(e)}")
            return False

    def reset_stale_scoring_tasks(self, max_scoring_time_minutes=15):
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
    
    def get_recent_model_scores(self, scores_per_model):
        """
        Get recent scores for all models.
        
        Args:
            scores_per_model (int): Number of recent scores to fetch per model
            
        Returns:
            dict: Dictionary of model scores grouped by UID
        """
        def _get_recent_scores():
            with self.session_scope() as session:
                try:
                    # First, create a subquery that ranks scores by timestamp for each model
                    ranked_scores = (
                        session.query(
                            ScoreHistory,
                            func.row_number().over(
                                partition_by=(ScoreHistory.hotkey, ScoreHistory.uid),
                                order_by=desc(ScoreHistory.scored_at)
                            ).label('score_rank')
                        )
                        .filter(ScoreHistory.is_archived == False)
                        .filter(ScoreHistory.score != 0)
                        .subquery()
                    )

                    # Get the most recent scores for each model
                    recent_scores = session.query(ranked_scores).filter(
                        ranked_scores.c.score_rank <= scores_per_model
                    ).subquery('recent_scores')

                    # Join with ModelQueue to get additional model information
                    results = session.query(
                        ModelQueue.uid,
                        ModelQueue.hotkey,
                        ModelQueue.competition_id,
                        ModelQueue.model_metadata,
                        recent_scores.c.score,
                        recent_scores.c.scored_at,
                        recent_scores.c.block,
                        recent_scores.c.model_hash,
                        recent_scores.c.scorer_hotkey,
                        recent_scores.c.score_rank
                    ).outerjoin(
                        recent_scores,
                        and_(
                            ModelQueue.hotkey == recent_scores.c.hotkey,
                            ModelQueue.uid == recent_scores.c.uid,
                        )
                    ).order_by(
                        ModelQueue.uid,
                        ModelQueue.hotkey,
                        recent_scores.c.scored_at.desc()
                    ).all()

                    scores_by_uid = defaultdict(lambda: defaultdict(list))
                    
                    for result in results:
                        if result.score is not None:
                            # Create a unique key for each hotkey+uid combination
                            model_key = f"{result.hotkey}_{result.uid}"
                            
                            scores_by_uid[result.uid][model_key].append({
                                'hotkey': result.hotkey,
                                'competition_id': result.competition_id,
                                'model_metadata': result.model_metadata,
                                'score': result.score,
                                'scored_at': result.scored_at.isoformat() if result.scored_at else None,
                                'block': result.block,
                                'model_hash': result.model_hash,
                                'scorer_hotkey': result.scorer_hotkey,
                                'rank': result.score_rank
                            })
                        else:
                            # Handle models with no scores
                            model_key = f"{result.hotkey}_{result.uid}"
                            if not scores_by_uid[result.uid][model_key]:  # Only add if no scores exist
                                scores_by_uid[result.uid][model_key].append({
                                    'hotkey': result.hotkey,
                                    'competition_id': None,
                                    'model_metadata': result.model_metadata,
                                    'score': None,
                                    'scored_at': None,
                                    'block': None,
                                    'model_hash': None,
                                    'scorer_hotkey': None,
                                    'rank': None
                                })

                    # Convert defaultdict to regular dict for return
                    return {
                        uid: dict(models) 
                        for uid, models in scores_by_uid.items()
                    }

                except Exception as e:
                    bt.logging.error(f"Error in _get_recent_scores: {str(e)}")
                    raise

        try:
            return self.execute_with_retry(_get_recent_scores)
        except Exception as e:
            bt.logging.error(f"Failed to get recent scores after {self.max_retries} attempts: {str(e)}")
            return {}
    
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

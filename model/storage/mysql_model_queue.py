from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON, func, desc, exists, ForeignKey, or_, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker, aliased, relationship
from collections import defaultdict

import json

from datetime import datetime, timedelta
import bittensor as bt

from model.data import ModelId
from api.config import DBHOST, DBNAME, DBUSER, DBPASS

Base = declarative_base()

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

# Create the engine and session
engine = create_engine(f'mysql://{DBUSER}:{DBPASS}@{DBHOST}/{DBNAME}')
Session = sessionmaker(bind=engine)

# Create the table
Base.metadata.create_all(engine)

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
    def __init__(self, max_scores_per_model=5, rescore_interval_hours=24):
        self.session = Session()
        self.max_scores_per_model = max_scores_per_model
        self.rescore_interval = timedelta(hours=rescore_interval_hours)

    def store_updated_model(self, uid, hotkey, model_metadata, updated):
        try:
            existing_model = self.session.query(ModelQueue).filter_by(hotkey=hotkey, uid=uid).first()

            serialized_metadata = json.dumps(model_metadata.__dict__, cls=ModelIdEncoder)

            if existing_model:
                if updated:
                    bt.logging.debug(f"Updating model metadata for UID={uid}, Hotkey={hotkey}")
                    existing_model.model_metadata = serialized_metadata
                    existing_model.is_new = updated
                    existing_model.block = model_metadata.block
                    existing_model.updated_at = datetime.utcnow()
            else:
                new_model = ModelQueue(
                    hotkey=hotkey,
                    uid=uid,
                    competition_id=model_metadata.id.competition_id,
                    model_metadata=serialized_metadata,
                    is_new=updated,
                    block=model_metadata.block
                )
                self.session.add(new_model)

            self.session.commit()
            bt.logging.debug(f"Stored model for UID={uid}, Hotkey={hotkey} in database. Is new = {updated}")

        except Exception as e:
            self.session.rollback()
            bt.logging.error(f"Error storing model in database: {str(e)}")
            bt.logging.error(f"Model metadata: {model_metadata}")

    def get_next_model_to_score(self):
        try:
            now = datetime.utcnow()
            subquery = self.session.query(
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

            next_model = self.session.query(ModelQueue).outerjoin(
                subquery, 
                and_(
                    ModelQueue.hotkey == subquery.c.hotkey, 
                    ModelQueue.uid == subquery.c.uid
                )
            ).filter(
                ModelQueue.is_being_scored == False,
                or_(
                    subquery.c.score_count == None,  # New models
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
                func.rand()  # Add randomization
            ).first()

            if next_model:
                bt.logging.debug(f"Found next model to score: hotkey={next_model.hotkey}, uid={next_model.uid}")
            else:
                bt.logging.debug("No models available for scoring")

            return next_model

        except Exception as e:
            bt.logging.error(f"Error getting model to score: {e}")
            return None

    def mark_model_as_being_scored(self, model_hotkey, model_uid, scorer_hotkey):
        model = self.session.query(ModelQueue).filter_by(hotkey=model_hotkey, uid=model_uid).first()
        if model and not model.is_being_scored:
            model.is_being_scored = True
            model.is_being_scored_by = scorer_hotkey
            model.scoring_updated_at = datetime.utcnow()
            self.session.commit()
            return True
        return False

    def submit_score(self, model_hotkey, model_uid, scorer_hotkey, model_hash, score):
        model = self.session.query(ModelQueue).filter_by(hotkey=model_hotkey, uid=model_uid).first()
        
        if not model:
            bt.logging.error(f"No model found for hotkey {model_hotkey} and uid {model_uid}")
            return False

        existing_scores = self.session.query(ScoreHistory).filter_by(hotkey=model_hotkey, uid=model_uid).count()

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
            self.session.add(new_score)
            model.is_being_scored = False
            model.is_being_scored_by = None
            model.scoring_updated_at = None
            self.session.commit()
            bt.logging.info(f"Successfully submitted score for model {model_hotkey} by {scorer_hotkey}")
            return True
        else:
            bt.logging.error(f"Failed to submit score for model {model_hotkey} by {scorer_hotkey}. "
                             f"Model: {model}, is_being_scored: {model.is_being_scored}, "
                             f"is_being_scored_by: {model.is_being_scored_by}, "
                             f"existing_scores: {existing_scores}")
            return False

    def reset_stale_scoring_tasks(self, max_scoring_time_minutes=10):
        stale_time = datetime.utcnow() - timedelta(minutes=max_scoring_time_minutes)
        stale_models = self.session.query(ModelQueue).filter(
            ModelQueue.is_being_scored == True,
            ModelQueue.scoring_updated_at < stale_time
        ).all()

        for model in stale_models:
            model.is_being_scored = False
            model.is_being_scored_by = None
            model.scoring_updated_at = None
            bt.logging.info(f"Reset scoring task for stale model: hotkey={model.hotkey}, uid={model.uid}")

        self.session.commit()
        return len(stale_models)
    
    def get_all_model_scores(self):
        try:
            # First, get the latest score timestamps for each hotkey+uid combination
            latest_scores = self.session.query(
                ScoreHistory.hotkey,
                ScoreHistory.uid,
                func.max(ScoreHistory.scored_at).label('latest_score_time')
            ).filter(
                ScoreHistory.is_archived == False
            ).group_by(
                ScoreHistory.hotkey, 
                ScoreHistory.uid
            ).subquery('latest_scores')

            # Then, get the actual score details by joining with ScoreHistory
            latest_score_details = self.session.query(
                ScoreHistory
            ).join(
                latest_scores,
                and_(
                    ScoreHistory.hotkey == latest_scores.c.hotkey,
                    ScoreHistory.uid == latest_scores.c.uid,
                    ScoreHistory.scored_at == latest_scores.c.latest_score_time
                )
            ).subquery('latest_score_details')

            # Finally, join with ModelQueue to get all the information
            query = self.session.query(
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
            )

            results = query.all()

            # Process scored models
            scores_by_uid = defaultdict(list)
            for result in results:
                if result.score is not None:  # This model has been scored
                    scores_by_uid[result.uid].append({
                        'hotkey': result.hotkey,
                        'competition_id': result.competition_id,
                        'score': result.score,
                        'scored_at': result.scored_at.isoformat() if result.scored_at else None,
                        'block': result.block,
                        'model_hash': result.model_hash,
                        #'scorer_hotkey': result.scorer_hotkey
                    })
                else:  # This model hasn't been scored yet
                    scores_by_uid[result.uid].append({
                        'hotkey': result.hotkey,
                        'competition_id': result.competition_id,
                        'score': None,
                        'scored_at': None,
                        'block': None,
                        'model_hash': None,
                        #'scorer_hotkey': None
                    })

            return dict(scores_by_uid)

        except Exception as e:
            bt.logging.error(f"Error retrieving all model scores: {str(e)}")
            return {}

    def archive_scores_for_deregistered_models(self, registered_hotkey_uid_pairs):
        """
        Archive deregistered models while preserving score history.
        
        Args:
            registered_hotkey_uid_pairs (list[tuple[str, str]]): List of (hotkey, uid) pairs for registered miners
        """
        try:
            all_models = self.session.query(ModelQueue.hotkey, ModelQueue.uid).all()
            deregistered_models = set((model.hotkey, model.uid) for model in all_models) - set(registered_hotkey_uid_pairs)

            for hotkey, uid in deregistered_models:
                try:
                    # Mark all related scores as archived
                    archive_result = self.session.query(ScoreHistory).filter_by(
                        hotkey=hotkey,
                        uid=uid,
                        is_archived=False
                    ).update(
                        {"is_archived": True},
                        synchronize_session=False
                    )

                    # Remove from ModelQueue
                    delete_result = self.session.query(ModelQueue).filter_by(
                        hotkey=hotkey,
                        uid=uid
                    ).delete(synchronize_session=False)

                    bt.logging.debug(
                        f"Processed deregistered model - Hotkey: {hotkey}, "
                        f"UID: {uid}, Archived scores: {archive_result}, "
                        f"Removed from queue: {delete_result}"
                    )

                except Exception as e:
                    bt.logging.error(f"Error processing model {hotkey}, {uid}: {str(e)}")
                    continue

            # Commit all changes
            self.session.commit()
            print(f"Archived scores and removed {len(deregistered_models)} deregistered models from the queue.")

        except Exception as e:
            self.session.rollback()
            bt.logging.error(f"Error archiving scores for deregistered models: {str(e)}")

    def close(self):
        self.session.close()

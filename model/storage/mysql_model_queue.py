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
    hotkey = Column(String(255), ForeignKey('sn21_model_queue.hotkey'), index=True)
    uid = Column(String(255), ForeignKey('sn21_model_queue.uid'), index=True)
    competition_id = Column(String(255), index=True)
    model_metadata = Column(JSON)
    score = Column(Float)
    scored_at = Column(DateTime, default=datetime.utcnow)
    block = Column(Integer)
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
        now = datetime.utcnow()
        subquery = self.session.query(
            ScoreHistory.hotkey,
            ScoreHistory.uid,
            func.count(ScoreHistory.id).label('score_count'),
            func.max(ScoreHistory.scored_at).label('last_scored_at')
        ).filter(ScoreHistory.is_archived == False).group_by(ScoreHistory.hotkey, ScoreHistory.uid).subquery()

        next_model = self.session.query(ModelQueue).outerjoin(
            subquery, and_(ModelQueue.hotkey == subquery.c.hotkey, ModelQueue.uid == subquery.c.uid)
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
            subquery.c.score_count.asc().nullsfirst(),
            subquery.c.last_scored_at.asc().nullsfirst(),
            desc(ModelQueue.updated_at)
        ).first()

        return next_model

    def mark_model_as_being_scored(self, model_hotkey, model_uid, scorer_hotkey):
        model = self.session.query(ModelQueue).filter_by(hotkey=model_hotkey, uid=model_uid).first()
        if model and not model.is_being_scored:
            model.is_being_scored = True
            model.is_being_scored_by = scorer_hotkey
            self.session.commit()
            return True
        return False

    def submit_score(self, model_hotkey, model_uid, scorer_hotkey, score):
        model = self.session.query(ModelQueue).filter_by(hotkey=model_hotkey, uid=model_uid).first()
        if model and model.is_being_scored and model.is_being_scored_by == scorer_hotkey:
            new_score = ScoreHistory(
                hotkey=model_hotkey,
                uid=model_uid,
                competition_id=model.competition_id,
                score=score,
                block=model.block,
                scorer_hotkey=scorer_hotkey,
                model_metadata=model.model_metadata 
            )
            self.session.add(new_score)
            model.is_being_scored = False
            model.is_being_scored_by = None
            self.session.commit()
            return True
        return False

    def reset_stale_scoring_tasks(self, max_scoring_time_minutes=10):
        stale_time = datetime.utcnow() - timedelta(minutes=max_scoring_time_minutes)
        stale_models = self.session.query(ModelQueue).filter(
            ModelQueue.is_being_scored == True,
            ModelQueue.updated_at < stale_time
        ).all()

        for model in stale_models:
            model.is_being_scored = False
            model.is_being_scored_by = None

        self.session.commit()
        return len(stale_models)
    
    def get_all_model_scores(self):
        try:
            latest_scores = self.session.query(
                ScoreHistory.hotkey,
                ScoreHistory.uid,
                func.max(ScoreHistory.scored_at).label('latest_score_time')
            ).filter(ScoreHistory.is_archived == False).group_by(ScoreHistory.hotkey, ScoreHistory.uid).subquery()

            query = self.session.query(
                ModelQueue.uid,
                ModelQueue.hotkey,
                ModelQueue.competition_id,
                ScoreHistory.score,
                ScoreHistory.scored_at,
                ScoreHistory.block,
                ScoreHistory.scorer_hotkey
            ).join(
                latest_scores,
                and_(
                    ModelQueue.hotkey == latest_scores.c.hotkey,
                    ModelQueue.uid == latest_scores.c.uid,
                    ModelQueue.hotkey == ScoreHistory.hotkey,
                    ModelQueue.uid == ScoreHistory.uid,
                    ScoreHistory.scored_at == latest_scores.c.latest_score_time
                )
            )

            results = query.all()

            scores_by_uid = defaultdict(list)
            for result in results:
                scores_by_uid[result.uid].append({
                    'hotkey': result.hotkey,
                    'competition_id': result.competition_id,
                    'score': result.score,
                    'scored_at': result.scored_at.isoformat(),
                    'block': result.block,
                    'scorer_hotkey': result.scorer_hotkey
                })

            unscored_models = self.session.query(
                ModelQueue.uid,
                ModelQueue.hotkey,
                ModelQueue.competition_id
            ).outerjoin(
                ScoreHistory,
                and_(ModelQueue.hotkey == ScoreHistory.hotkey, ModelQueue.uid == ScoreHistory.uid)
            ).filter(
                ScoreHistory.id == None
            ).all()

            for model in unscored_models:
                scores_by_uid[model.uid].append({
                    'hotkey': model.hotkey,
                    'competition_id': model.competition_id,
                    'score': None,
                    'scored_at': None,
                    'block': None,
                    'scorer_hotkey': None
                })

            return dict(scores_by_uid)

        except Exception as e:
            bt.logging.error(f"Error retrieving all model scores: {str(e)}")
            return {}

    def archive_scores_for_deregistered_models(self, registered_hotkey_uid_pairs):
        try:
            all_models = self.session.query(ModelQueue.hotkey, ModelQueue.uid).all()
            deregistered_models = set((model.hotkey, model.uid) for model in all_models) - set(registered_hotkey_uid_pairs)

            for hotkey, uid in deregistered_models:
                # Archive scores
                self.session.query(ScoreHistory).filter_by(hotkey=hotkey, uid=uid).update({"is_archived": True})
                
                # Remove from ModelQueue
                self.session.query(ModelQueue).filter_by(hotkey=hotkey, uid=uid).delete()

            self.session.commit()
            bt.logging.debug(f"Archived scores and removed {len(deregistered_models)} deregistered models from the queue.")

        except Exception as e:
            self.session.rollback()
            bt.logging.error(f"Error archiving scores for deregistered models: {str(e)}")

    def close(self):
        self.session.close()

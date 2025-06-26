from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON, func, desc, exists, ForeignKey, or_, and_, case
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker, aliased, relationship
from contextlib import contextmanager
from collections import defaultdict

import time
from datetime import datetime, timedelta, timezone
import bittensor as bt
from typing import Optional

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

def get_table_name(base_name: str) -> str:
    """Helper function to get the correct table name with suffix if not in production."""
    return f"{base_name}{'_test' if not IS_PROD else ''}"

class BaselineScore(Base):
    __tablename__ = get_table_name('sn21_baseline_scores')
    id = Column(Integer, primary_key=True)
    competition_id = Column(String(255), nullable=False, index=True)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<BaselineScore(competition_id='{self.competition_id}', score={self.score}, created_at='{self.created_at}')>"

class MinerReputation(Base):
    __tablename__ = get_table_name('sn21_miner_reputations')
    hotkey = Column(String(255), primary_key=True)
    reputation = Column(Float, default=0.5, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<MinerReputation(hotkey='{self.hotkey}', reputation={self.reputation})>"

class ReputationHistory(Base):
    __tablename__ = get_table_name('sn21_reputation_history')
    id = Column(Integer, primary_key=True)
    hotkey = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    reputation = Column(Float, nullable=False)

    def __repr__(self):
        return f"<ReputationHistory(hotkey='{self.hotkey}', timestamp='{self.timestamp}', reputation={self.reputation})>"

class ReputationStore:
    def __init__(self, max_retries=3, retry_delay=1):
        # Ensure DB is initialized and get the sessionmaker
        if Session is None:
            raise RuntimeError("Database not initialized. Call init_database() first.")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = get_session()

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
        except Exception as e:
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

    def get_latest_baseline_score(self, competition_id):
        def _get():
            with self.session_scope() as session:
                latest_baseline = (
                    session.query(BaselineScore)
                    .filter(BaselineScore.competition_id == competition_id)
                    .order_by(BaselineScore.created_at.desc())
                    .first()
                )
                if latest_baseline is None:
                    return None
                else:
                    return {"competition_id": latest_baseline.competition_id, "score": latest_baseline.score, "created_at": latest_baseline.created_at}

        return self.execute_with_retry(_get)

    def get_all_reputations(self):
        def _get():
            with self.session_scope() as session:
                records = session.query(MinerReputation).all()
                return {
                    record.hotkey: {
                        "reputation": record.reputation,
                        "last_updated": record.last_updated.isoformat() if record.last_updated else None
                    }
                    for record in records
                }
        return self.execute_with_retry(_get)

    def get_reputation(self, hotkey):
        def _get():
            with self.session_scope() as session:
                record = session.query(MinerReputation).filter(MinerReputation.hotkey == hotkey).first()
                if not record:
                    return None
                return {
                    "hotkey": record.hotkey,
                    "reputation": record.reputation,
                    "last_updated": record.last_updated.isoformat() if record.last_updated else None
                }
        return self.execute_with_retry(_get) 

def main():
    """
    Main function to demonstrate the ReputationStore's three get methods.
    """
    try:
        # Initialize the database
        print("Initializing database...")
        init_database()
        print("Database initialized successfully!")
        
        # Create ReputationStore instance
        print("\nCreating ReputationStore instance...")
        reputation_store = ReputationStore()
        print("ReputationStore created successfully!")
        
        # Test 1: Get latest baseline score
        print("\n=== Testing get_latest_baseline_score ===")
        competition_id = "v1"
        baseline_score = reputation_store.get_latest_baseline_score(competition_id)
        if baseline_score:
            print(f"Latest baseline score for competition '{competition_id}':")
            print(f"  Score: {baseline_score['score']}")
            print(f"  Created at: {baseline_score['created_at']}")
        else:
            print(f"No baseline score found for competition '{competition_id}'")
        
        # Test 2: Get all reputations
        print("\n=== Testing get_all_reputations ===")
        all_reputations = reputation_store.get_all_reputations()
        if all_reputations:
            print(f"Found {len(all_reputations)} miner reputations:")
            for hotkey, data in all_reputations.items():
                print(f"  Hotkey: {hotkey}")
                print(f"    Reputation: {data['reputation']}")
                print(f"    Last Updated: {data['last_updated']}")
                break
        else:
            print("No miner reputations found in database")
        
        # Test 3: Get specific reputation
        print("\n=== Testing get_reputation ===")
        test_hotkey = "test_hotkey_123"
        reputation = reputation_store.get_reputation(test_hotkey)
        if reputation:
            print(f"Reputation for hotkey '{test_hotkey}':")
            print(f"  Hotkey: {reputation['hotkey']}")
            print(f"  Reputation: {reputation['reputation']}")
            print(f"  Last Updated: {reputation['last_updated']}")
        else:
            print(f"No reputation found for hotkey '{test_hotkey}'")
        
        print("\n=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



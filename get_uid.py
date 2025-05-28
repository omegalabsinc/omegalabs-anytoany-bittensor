import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, desc, Float, JSON, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime as dt
import os
import json # Used for potentially pretty-printing the output

# Attempt to import DB config like in mysql_model_queue.py
from vali_api.config import DBHOST, DBNAME, DBUSER, DBPASS, IS_PROD

print(f"DBHOST: {DBHOST}")
print(f"DBNAME: {DBNAME}")
print(f"DBUSER: {DBUSER}")
print(f"DBPASS: {DBPASS}")
print(f"IS_PROD: {IS_PROD}")

# --- Data Class Definitions (Inspired by model.data) ---
class ModelId:
    def __init__(self, namespace: str, name: str, hash: str, commit: str, competition_id: str):
        self.namespace = namespace
        self.name = name
        self.hash = hash
        self.commit = commit
        self.competition_id = competition_id

    @classmethod
    def from_dict(cls, data: dict) -> "ModelId":
        required_keys = ['namespace', 'name', 'hash', 'commit', 'competition_id']
        if not all(key in data for key in required_keys):
            raise ValueError(f"Missing one or more required keys in ModelId data: {data}")
        return cls(
            namespace=data['namespace'],
            name=data['name'],
            hash=data['hash'],
            commit=data['commit'],
            competition_id=data['competition_id']
        )
    
    @classmethod
    def from_compressed_str(cls, compressed_str: str) -> "ModelId":
        parts = compressed_str.split(":")
        if len(parts) != 5:
            raise ValueError(f"Invalid compressed ModelId string: {compressed_str}")
        return cls(namespace=parts[0], name=parts[1], hash=parts[2], commit=parts[3], competition_id=parts[4])

    def to_dict(self):
        return {
            "namespace": self.namespace,
            "name": self.name,
            "hash": self.hash,
            "commit": self.commit,
            "competition_id": self.competition_id,
        }

    def __repr__(self):
        return f"ModelId(namespace='{self.namespace}', name='{self.name}', hash='{self.hash}', commit='{self.commit}', competition_id='{self.competition_id}')"

class ModelMetadata:
    def __init__(self, id: ModelId, block: int):
        self.id = id
        self.block = block

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        if not data or 'id' not in data or 'block' not in data:
            raise ValueError(f"Invalid or missing 'id' or 'block' in ModelMetadata data: {data}")
        
        model_id_data = data['id']
        if not isinstance(model_id_data, dict):
            raise ValueError(f"ModelMetadata 'id' field must be a dictionary, got: {type(model_id_data)}")
            
        model_id_obj = ModelId.from_dict(model_id_data)
        return cls(id=model_id_obj, block=data['block'])

    def to_dict(self):
        return {
            "id": self.id.to_dict(),
            "block": self.block,
        }

    def __repr__(self):
        return f"ModelMetadata(id={self.id!r}, block={self.block})"

# --- SQLAlchemy Model Definition ---
Base = declarative_base()

def get_table_name(base_name: str) -> str:
    """Helper function to get the correct table name with suffix if not in production."""
    # Ensure IS_PROD has a value, default to True if import failed and not set
    is_prod_env = IS_PROD if IS_PROD is not None else True 
    return f"{base_name}{'_test' if not is_prod_env else ''}"

class SN21ScoreHistoryDBEntry(Base):
    __tablename__ = get_table_name('sn21_score_history')

    id = Column(Integer, primary_key=True, autoincrement=True)
    hotkey = Column(String(255), index=True, nullable=True)
    uid = Column(String(255), index=True, nullable=True) 
    competition_id = Column(String(255), nullable=True)
    model_metadata = Column(JSON, nullable=True) 
    score = Column(Float, nullable=True)
    scored_at = Column(DateTime, nullable=True, index=True)
    block = Column(Integer, nullable=True)
    model_hash = Column(String(255), nullable=True)
    scorer_hotkey = Column(String(255), nullable=True)
    is_archived = Column(Boolean, default=False, nullable=True)

    def __repr__(self):
        return f"<SN21ScoreHistoryDBEntry(id={self.id}, uid='{self.uid}', hotkey='{self.hotkey}', competition_id='{self.competition_id}', scored_at='{self.scored_at}', is_archived={self.is_archived})>"


# --- Database Connection and Query Logic ---
# Ensure DBPASS is at least an empty string if it's meant to be optional but present in the string
# The main check is for the essential components of the connection.
if DBHOST and DBNAME and DBUSER: # Check if crucial DB params are loaded
    # DBPASS will be used as is from the import. If it's an empty string for no password, that's fine.
    DATABASE_URL = f'mysql://{DBUSER}:{DBPASS}@{DBHOST}/{DBNAME}'
elif DBHOST and DBNAME and DBUSER and DBPASS is None: # Explicitly handle if DBPASS could be None and means no password
    DATABASE_URL = f'mysql://{DBUSER}@{DBHOST}/{DBNAME}' # Format for no password if DBPASS is None
else:
    DATABASE_URL = None # Will cause failure if get_db_session is called
    print("Critical database configuration (DBHOST, DBNAME, or DBUSER) missing. DATABASE_URL not set.")

def get_db_session(db_url: str):
    if not db_url:
        raise ConnectionError("Database URL is not configured. Cannot create session.")
    engine = create_engine(db_url)
    Base.metadata.create_all(engine) 
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def get_uid_and_metadata_from_db(session):
    """
    Fetches a UID and its ModelMetadata from the sn21_score_history table.
    Prioritizes the most recently scored, non-archived entry.
    """
    try:
        latest_score_entry = (
            session.query(SN21ScoreHistoryDBEntry)
            .filter(SN21ScoreHistoryDBEntry.is_archived == False)
            .order_by(desc(SN21ScoreHistoryDBEntry.scored_at))
            .first()
        )
    except sqlalchemy.exc.OperationalError as e:
        print(f"MySQL Connection Error: {e}")
        print("Please ensure MySQL server is running and credentials in vali_api.config are correct.")
        return None, None
    except sqlalchemy.exc.NoSuchTableError as e:
        print(f"Error: Table {SN21ScoreHistoryDBEntry.__tablename__} does not exist. {e}")
        print("Ensure migrations have run or the table name is correct (check IS_PROD in vali_api.config).")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during database query: {e}")
        return None, None

    if latest_score_entry:
        try:
            if not latest_score_entry.model_metadata:
                print(f"Error: model_metadata is missing or null for entry id {latest_score_entry.id}")
                return None, None
            
            model_metadata_dict = latest_score_entry.model_metadata
            if not isinstance(model_metadata_dict, dict):
                if isinstance(model_metadata_dict, str):
                    try:
                        model_metadata_dict = json.loads(model_metadata_dict)
                    except json.JSONDecodeError as json_e:
                        print(f"Error decoding model_metadata JSON string for entry id {latest_score_entry.id}: {json_e}")
                        return None, None
                else:
                    print(f"Error: model_metadata is not a dictionary for entry id {latest_score_entry.id}, type: {type(model_metadata_dict)}")
                    return None, None

            model_metadata_obj = ModelMetadata.from_dict(model_metadata_dict)
            
            uid_to_return = latest_score_entry.uid
            try:
                if uid_to_return is not None:
                    uid_to_return = int(uid_to_return)
            except ValueError:
                print(f"Warning: UID '{uid_to_return}' from DB could not be converted to int for entry id {latest_score_entry.id}. Returning as string.")

            return uid_to_return, model_metadata_obj
        except ValueError as e_parse:
            print(f"Error parsing ModelMetadata from DB entry id {latest_score_entry.id}: {e_parse}")
            return None, None
        except Exception as e_proc:
            print(f"An unexpected error occurred while processing entry id {latest_score_entry.id}: {e_proc}")
            return None, None
    else:
        print(f"No non-archived entries found in the {SN21ScoreHistoryDBEntry.__tablename__} table.")
        return None, None

# --- Main Execution ---
if __name__ == "__main__":
    if not DATABASE_URL:
        print("Exiting: Database connection not configured due to missing credentials from vali_api.config.")
    else:
        print(f"Attempting to connect to database: {DBNAME} on {DBHOST}")
        db_session = None # Initialize to ensure it's defined for finally block
        try:
            db_session = get_db_session(DATABASE_URL)
            uid, metadata = get_uid_and_metadata_from_db(db_session)
            if uid is not None and metadata is not None:
                print(f"Retrieved data from {SN21ScoreHistoryDBEntry.__tablename__} table:")
                print(f"  UID: {uid} (type: {type(uid)})")
                print(f"  ModelMetadata: {json.dumps(metadata.to_dict(), indent=4)}")
            else:
                print(f"Could not retrieve UID and ModelMetadata from the {SN21ScoreHistoryDBEntry.__tablename__} table.")
        except ConnectionError as e_conn:
            print(f"Failed to establish database session: {e_conn}")
        except Exception as e_main:
            print(f"An error occurred in main execution: {e_main}")
        finally:
            if db_session:
                db_session.close()
                print("Database session closed.") 
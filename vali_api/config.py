import os
from dotenv import load_dotenv
import json

# Define the path to the api.env file
env_path = os.path.join(os.path.dirname(__file__), 'api.env')

# Load the environment variables from the api.env file
load_dotenv(dotenv_path=env_path)

IS_PROD = False if os.getenv("IS_PROD") == "False" else True

NETWORK = None
NETUID = 21

if not IS_PROD:
    NETWORK = "test"
    NETUID = 96

DBHOST = os.getenv("DBHOST")
DBNAME = os.getenv("DBNAME")
DBUSER = os.getenv("DBUSER")
DBPASS = os.getenv("DBPASS")

SENTRY_DSN = os.getenv("SENTRY_DSN")

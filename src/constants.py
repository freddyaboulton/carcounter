import os

_db_user = os.environ['DB_USERNAME']
_db_password = os.environ['DB_PASSWORD']
_db_uri = os.environ['DB_URI']

DB_CONNECTION = f"postgresql+psycopg2://{_db_user}:{_db_password}@{_db_uri}"
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
CAR_INDEX = 3
MODEL_ENDPOINT = os.environ['MODEL_ENDPOINT']

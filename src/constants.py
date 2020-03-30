import os

DB_USER = os.environ['DB_USERNAME']
DB_PASSWORD = os.environ['DB_PASSWORD']
DB_URI = os.environ['DB_URI']
DB_NAME = os.environ['DB_NAME']
#DB_CONNECTION = f"postgresql+psycopg2://{_db_user}:{_db_password}@{_db_uri}"
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
CAR_INDEX = 3
MODEL_ENDPOINT = os.environ['MODEL_ENDPOINT']

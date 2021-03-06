import json
import os
from typing import Any, Dict, List

import numpy as np
import psycopg2
import redis
import requests
from PIL import Image
import cv2

from src.constants import (CAR_INDEX, DB_NAME, DB_PASSWORD, DB_URI, DB_USER,
                       IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_ENDPOINT)
from src.data_types import Box, TrackedObject
from src.tracker import CentroidTracker

headers= {'content-type': 'application/json'}

cache = redis.Redis(host='localhost', port=6379, db=0)

state = cache.get('tracker-state')
if state:
    tracker = CentroidTracker.deserialize(json.loads(state))
else: 
    tracker = CentroidTracker()

# datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

def get_boxes_from_pred(response: Dict[str, Any], min_score: float = 0.15) -> List[Box]:
    predictions = response['predictions'][0]
    classes = predictions['detection_classes']
    pred_boxes = predictions['detection_boxes']
    scores = predictions['detection_scores'] 
    boxes = []
    for label, box, score in zip(classes, pred_boxes, scores): 
        if label == CAR_INDEX and score > min_score:
            boxes.append((int(box[1] * IMAGE_HEIGHT),
                          int(box[0] * IMAGE_WIDTH),
                          int(box[3] * IMAGE_HEIGHT),
                          int(box[2] * IMAGE_WIDTH)))
    
    return boxes


def draw_objects_on_image(tracked_objects: List[TrackedObject], image: np.ndarray) -> np.ndarray:

    image_with_objects = image.copy()
    for tracked_object in tracked_objects:
       tracked_object.draw_bounding_box(image_with_objects)
    
    return image_with_objects


def insert_into_table(date_time: str, ids: List[int]) -> None:
    sql = "INSERT INTO carcount(time, track_id) VALUES "
    
    conn = psycopg2.connect(host=DB_URI, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    cur = conn.cursor()
    values = ", ".join([f"('{date_time}', {str(ID)})" for ID in ids])
    
    cur.execute(sql + values)
    conn.commit()
    cur.close()


def cache_tracker_state() -> None:
    tracker_state = tracker.serialize()
    cache.set('tracker-state', json.dumps(tracker_state))


def to_redis(cache, image: np.ndarray, key: str) -> None:
   """Store given Numpy array 'a' in Redis under key 'n'"""
   _, buffer = cv2.imencode('.png', image)

   image_bytes = np.array(buffer).tostring()

   # Write into redis server
   cache.set(key, image_bytes)


def from_redis(cache, key: str) -> np.ndarray:
   """Retrieve Numpy array from Redis key 'n'"""
   image_bytes = cache.get(key)
   
   # Decoding CV2+Redis
   decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
   return decoded


def process_image(date_time: str, image: np.ndarray) -> None:

    assert image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    image_inf = np.expand_dims(image, 0)

    assert image_inf.shape == (1, 300, 300, 3)

    data = json.dumps({'signature_name': 'serving_default',
                       'instances': image_inf.tolist()})
    response = requests.post('http://localhost:8501/v1/models/ssd:predict', data=data, headers=headers)
    response = json.loads(response.text)
    predicted_boxes = get_boxes_from_pred(response)
    print('boxes\n'); print(predicted_boxes)
    print(tracker.objects)
    current_objects = tracker.update(predicted_boxes)
    print(current_objects)
    img_with_objects = draw_objects_on_image(current_objects, image)
    
    ids = [obj.id for obj in current_objects]
    if ids:
        insert_into_table(date_time, ids)
    
    cache_tracker_state()

    to_redis(cache, img_with_objects, 'latest-image')



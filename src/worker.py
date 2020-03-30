import os
from typing import List, Dict, Any
from data_types import Box, TrackedObject
from constants import CAR_INDEX, IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_ENDPOINT, DB_CONNECTION
from tracker import CentroidTracker
import requests
import json
import numpy as np
import psycopg2
import redis
from PIL import Image

headers= {'content-type': 'application/json'}

cache = redis.Redis(host='localhost', port=6379, db=0)

state = cache.get('tracker-state')
if state:
    tracker = CentroidTracker.deserialize(json.loads(state))
else:
    tracker = CentroidTracker()

# datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

def get_boxes_from_pred(response: Dict[str, Any], min_score: float = 0.15) -> List[Box]:
    predictions = response['predictions']
    classes = predictions['classes']
    boxes = predictions['detection_boxes']
    scores = predictions['detection_scores']
    boxes = []
    for label, box, score in zip(classes, boxes, scores):
        if label == CAR_INDEX and score > min_score:
            boxes.append((int(box[0] * IMAGE_HEIGHT),
                          int(box[1] * IMAGE_WIDTH),
                          int(box[2] * IMAGE_HEIGHT),
                          int(box[3] * IMAGE_WIDTH)))
    
    return boxes


def draw_objects_on_image(tracked_objects: List[TrackedObject], image: np.ndarray) -> np.ndarray:

    image_with_objects = image.copy()
    for tracked_object in tracked_objects:
       tracked_object.draw_bounding_box(image_with_objects)
    
    return image_with_objects


def insert_into_table(date_time: str, ids: List[int]) -> None:
    sql = """ INSERT INTO carcount(time, track_id) VALUES(%s) """
    
    conn = psycopg2.connect(DB_CONNECTION)
    cur = conn.cursor()
    values = [(date_time, str(ID)) for ID in ids]
    cur.execute_many(sql, values)
    conn.commit()
    cur.close()


def cache_tracker_state() -> None:
    tracker_state = tracker.serialize()
    cache.set('tracker-state', json.dumps(tracker_state))


def process_image(date_time: str, image: np.ndarray) -> None:

    assert image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    image = np.expand_dims(image, 0)
    data = json.dumps({'signature_name': 'serving_default',
                       'instances': image.astype('uint8').tolist()})
    response = requests.post(MODEL_ENDPOINT, data)
    response = json.loads(response.text)
    predicted_boxes = get_boxes_from_pred(response)
    current_objects = tracker.update(predicted_boxes)

    img_with_objects = draw_objects_on_image(current_objects, image)
    
    ids = [obj.track_id for obj in current_objects]
    insert_into_table(date_time, ids)
    
    cache_tracker_state()

    Image.fromarray(img_with_objects.astype("uint8")).save(f"server/img_{date_time}.png")



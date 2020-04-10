from typing import Tuple, List, Dict, Any
from random import randint
import numpy as np
import cv2

Box = Tuple[int, int, int, int]

class TrackedObject:
    
    def __init__(self, track_id: int, bounding_box: Box):
        
        min_x, min_y, max_x, max_y = bounding_box

        self.corners = (min_x, min_y), (max_x, max_y)
        self.color = self._get_color()
        self.id = track_id
    
    def _get_color(self):
        return (randint(0, 255), randint(0, 255), randint(0, 255))
    
    @classmethod
    def from_pytorch(cls, box_corners: List[int]):
        return cls(*[int(coordinate) for coordinate in box_corners])
    
    @property
    def centroid(self):
        return (self.corners[1][0] + self.corners[0][0]) / 2,\
               (self.corners[1][1] + self.corners[0][1]) / 2
    
    @property
    def cv2_rectangle(self):
        return (self.corners[0][0], self.corners[0][1],
                self.corners[1][0] - self.corners[0][0],
                self.corners[1][1] - self.corners[0][1])
    
    def draw_bounding_box(self, image: np.ndarray):
        cv2.rectangle(image, self.corners[0], self.corners[1],
                      self.color, 1)
    
    def update_box_from_cv2_tracker(self, new_box):
        self.corners = (int(new_box[0]), int(new_box[1])), (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))

    def update_from_box(self, box: Box) -> None:
        self.corners = (box[0], box[1]), (box[2], box[3])
    
    def serialize(self) -> Dict[str, Any]:
        return {'box': [*self.corners[0], *self.corners[1]],
                'color': self.color,
                'id': self.id}
    
    @classmethod
    def deserialize(cls, state: Dict[str, Any]) -> Any:
        track = TrackedObject(state['id'], state['box'])
        track.color = state['color']
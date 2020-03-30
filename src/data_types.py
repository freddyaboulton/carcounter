from typing import Tuple, List, Dict, Any
from random import randint
import numpy as np
import cv2

Box = Tuple[int, int, int, int]

class TrackedObject:

    _ID = 0
    
    def __init__(self, min_x: int, min_y: int, max_x: int, max_y: int):

        self.corners = (min_x, min_y), (max_x, max_y)
        self.color = self._get_color()
        self.id = self._ID
        self.update_counter()

    @classmethod
    def update_counter(cls,):
        cls._ID += 1
    
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
        return {'corners': [self.corners[0], self.corners[1]],
                'color': self.color,
                'id': self.id}
    
    @classmethod
    def deserialize(cls, state: Dict[str, Any]) -> Any:
        track = TrackedObject(state['corners'][0][0],
                              state['corners'][0][1],
                              state['corners'][1][0],
                              state['corners'][1][1])
        track.color = state['color']
        track.id = state['id']
        # ensures the tracker uses the latest id
        cls._ID = state['id']

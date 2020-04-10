from typing import Dict, List, Any
import numpy as np
from src.data_types import TrackedObject, Box
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker:

    def __init__(self, starting_id: int = 0, max_missing_frames: int = 5):

        self.objects: Dict[int, TrackedObject] = OrderedDict()
        self.disappeared: Dict[int, int] = OrderedDict()
        self.max_missing_frames = max_missing_frames
        self.current_id = starting_id
    
    def register(self, box: Box) -> None:
        tracked_object = TrackedObject(self.current_id, box)
        self._update_id()
        self.objects[tracked_object.id] = tracked_object
        self.disappeared[tracked_object.id] = 0
    
    def _update_id(self) -> None:
        self.current_id += 1
    
    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def serialize(self,) -> Dict[str, Any]:
        objects = {ID: track.serialize() for ID, track in self.objects.items()}
        return {'objects': objects,
                'disappeared': self.disappeared,
                'max_missing_frames': self.max_missing_frames,
                'current_id': self.current_id}
    
    @classmethod
    def deserialize(cls, state: Dict[str, Any]) -> Any:
        objects = OrderedDict({ID: TrackedObject.deserialize(track)\
                               for ID, track in state['objects'].items()})
        disappeared = OrderedDict(state['disappeared'])
        tracker = CentroidTracker(state['current_id'], state['max_missing_frames'])
        tracker.objects = objects
        tracker.disappeared = disappeared
        return tracker
    
    def update(self, boxes: List[Box]) -> List[TrackedObject]:
        
        if len(boxes) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[object_id] > self.max_missing_frames:
                    self.deregister(object_id)
            # return early as there are no centroids or tracking info
            # to update
            return list(self.objects.values())
        
        if len(self.objects) == 0:
            for box in boxes:
                self.register(box)

        else:
        
            # initialize an array of input centroids for the current frame
            input_centroids = np.zeros((len(boxes), 2), dtype="int")
            # loop over the bounding box rectangles
            for (i, (start_x, start_y, end_x, end_y)) in enumerate(boxes):
                # use the bounding box coordinates to derive the centroid
                center_x = int((start_x + end_x) / 2.0)
                center_y = int((start_y + end_y) / 2.0)
                input_centroids[i] = (center_x, center_y)
            
            object_ids = list(self.objects.keys())
            centroids = [track.centroid for track in self.objects.values()]

            D = dist.cdist(np.array(centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id].update_from_box(boxes[row])
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unused_rows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[object_id] > self.max_missing_frames:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(boxes[col])
        
        return list(self.objects.values())
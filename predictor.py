from typing import List, Dict
from sieve.types import FrameSingleObject, SingleObject, BoundingBox, TemporalObject
from sieve.predictors import TemporalProcessor

from retinaface import RetinaFace
import cv2

class FaceDetector(TemporalProcessor):
    def setup(self):
        self.model = RetinaFace
        a = cv2.imread('dummy.png')
        self.model.detect_faces(a)
    
    def predict(self, frame: FrameSingleObject) -> List[SingleObject]:
        frame_number = frame.temporal_object.frame_number
        frame_data = frame.temporal_object.get_array()
        faces = self.model.detect_faces(frame_data)

        output_objects = []

        for k, face in faces.items():
            out_cls = "face"
            out_bbox = BoundingBox.from_array(face['facial_area'])
            out_score = face['score']

            output_objects.append(
                SingleObject(
                    cls=out_cls,
                    temporal_object=TemporalObject(
                        frame_number=frame_number,
                        bounding_box=out_bbox,
                        score=out_score,
                    )
                )
            )
        
        return output_objects

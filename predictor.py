from typing import List
from sieve.types import FrameSingleObject, BoundingBox
from sieve.predictors import TemporalPredictor
from sieve.types.outputs import Detection
from retinaface import RetinaFace
from sieve.types.constants import FRAME_NUMBER, BOUNDING_BOX, SCORE, CLASS
import cv2
import requests

class FaceDetector(TemporalPredictor):

    def setup(self):
        self.model = RetinaFace
        #Dummy image to warm up model
        url = 'https://github.com/sieve-community/face-detection/blob/master/dummy.png?raw=true' 
        res = requests.get(url, stream = True)
        with open('dummy.png', 'wb') as f:
            for chunk in res.iter_content(chunk_size = 1024*1024):
                if chunk:
                    f.write(chunk)
        a = cv2.imread('dummy.png')
        #Warm up model
        self.model.detect_faces(a) 
    
    def predict(self, frame: FrameSingleObject) -> List[Detection]:
        # Get the frame array
        frame_number = frame.get_temporal().frame_number
        frame_data = frame.get_temporal().get_array()

        # Run the model
        faces = self.model.detect_faces(frame_data)

        output_objects = []

        if type(faces) == tuple: #Edge case where no faces are detected
            return output_objects

        #Iterate through each face and create a new detection object
        for k, face in faces.items():
            out_cls = "face"
            out_bbox = BoundingBox.from_array(face['facial_area'])
            out_score = face['score']
            out_dict = {
                FRAME_NUMBER: frame_number,
                BOUNDING_BOX: out_bbox,
                SCORE: out_score,
                CLASS: out_cls
            }

            output_objects.append(
                Detection(
                    **out_dict
                )
            )
        
        return output_objects

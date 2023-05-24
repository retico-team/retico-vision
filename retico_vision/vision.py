
import cv2
import numpy as np
from PIL import Image

import retico_core



class ImageIU(retico_core.IncrementalUnit):
    """An image incremental unit that receives raw image data from a source.

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
        image (bytes[]): The image of this IU
        rate (int): The frame rate of this IU
        nframes (int): The number of frames of this IU
    """

    @staticmethod
    def type():
        return "Image IU"

    def __init__(
        self, 
        creator=None, 
        iuid=0, 
        previous_iu=None, 
        grounded_in=None,
        rate=None,
        nframes=None, 
        image=None,
        **kwargs
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=image
        )
        self.image = image
        self.rate = rate
        self.nframes = nframes

    def set_image(self, image, nframes, rate):
        """Sets the audio content of the IU."""
        self.image = image
        self.payload = image
        self.nframes = int(nframes)
        self.rate = int(rate)

    def get_json(self):
        payload = {}
        payload['image'] = np.array(self.payload).tolist()
        payload['nframes'] = self.nframes
        payload['rate'] = self.rate
        return payload

    def create_from_json(self, json_dict):
        self.image =  Image.fromarray(np.array(json_dict['image'], dtype='uint8'))
        self.payload = self.image
        self.nframes = json_dict['nframes']
        self.rate = json_dict['rate']

class DetectedObjectsIU(retico_core.IncrementalUnit):
    """An image incremental unit that maintains a list of detected objects and their bounding boxes.

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
    """

    @staticmethod
    def type():
        return "Detected Objects IU"

    def __init__(
        self, 
        creator=None, 
        iuid=0, 
        previous_iu=None,
        grounded_in=None,
        **kwargs
    ):
        super().__init__(
            creator=creator,
            iuid=iuid, 
            previous_iu=previous_iu,
            grounded_in=grounded_in, 
            payload=None
        )
        self.image = None
        self.detected_objects = None
        self.num_objects = 0

    def set_detected_objects(self, image, detected_objects):
        """Sets the content for the IU"""
        self.image = image
        self.payload = detected_objects
        self.detected_objects = detected_objects
        self.num_objects = len(detected_objects)

    def get_json(self):
        payload = {}
        payload['image'] = np.array(self.payload).tolist()
        payload['detected_objects'] = self.detected_objects
        payload['num_objects'] = self.num_objects
        return payload

    def create_from_json(self, json_dict):
        self.image =  Image.fromarray(np.array(json_dict['image'], dtype='uint8'))
        self.detected_objects = json_dict['detected_objects']
        self.payload = self.detected_objects
        self.num_objects = json_dict['num_objects']

class WebcamModule(retico_core.AbstractProducingModule):
    """A module that produces IUs containing images that are captures by
    a web camera."""

    @staticmethod
    def name():
        return "Webcam Module"

    @staticmethod
    def description():
        return "A prodicing module that records images from a web camera."

    @staticmethod
    def output_iu():
        return ImageIU

    def __init__(self, width=None, height=None, rate=None, pil=True, **kwargs):
        """
        Initialize the Webcam Module.
        Args:
            width (int): Width of the image captured by the webcam; will use camera default if unset
            height (int): Height of the image captured by the webcam; will use camera default if unset
            rate (int): The frame rate of the recording; will use camera default if unset
        """
        super().__init__(**kwargs)
        self.pil = pil
        self.width = width
        self.height = height
        self.rate = rate
        self.cap = cv2.VideoCapture(0)

        self.setup()

    def process_update(self, _):
        ret, frame = self.cap.read() # ret should be false if camera is off
        if ret:
            output_iu = self.create_iu()
            # output_iu.set_image(frame, self.width, self.height, self.rate)
            if self.pil:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            output_iu.set_image(frame, 1, self.rate)
            return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        else:
            print('camera may not be on')

    def setup(self):
        """Set up the webcam for recording."""
        cap = self.cap
        if self.width != None:
            cap.set(3, self.width)
        else:
            self.width = int(cap.get(3))
        if self.height != None:
            cap.set(4, self.height)
        else:
            self.height = int(cap.get(4))
        if self.rate != None:
            cap.set(5, self.rate)
        else:
            self.rate = int(cap.get(5))

    def shutdown(self):
        """Close the video stream."""
        self.cap.release()        




import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Label
import threading
from collections import deque
import json
import math
import os
import datetime
import time
import retico_core
from urllib.parse import urlparse, urlunparse, quote



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

    def to_zmq(self, update_type):
        """
        returns a formatted string that can be sent across zeromq
        """
        payload = {}
        payload["originatingTime"] = datetime.datetime.now().isoformat() #zmq expected format
        payload["update_type"] = str(update_type)
        message = {}
        message['image'] = np.array(self.payload).tolist()
        message['nframes'] = self.nframes
        message['rate'] = self.rate
        payload["message"] = json.dumps(message)
        return payload

    def from_zmq(self, zmq_data):
        zmq_data = json.loads(zmq_data['message'])
        self.image =  Image.fromarray(np.array(zmq_data['image'], dtype='uint8'))
        self.payload = self.image
        self.nframes = zmq_data['nframes']
        self.rate = zmq_data['rate']

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
        self.object_type = None

    def set_detected_objects(self, image, detected_objects, object_type):
        """Sets the content for the IU"""
        self.image = image
        self.payload = detected_objects
        self.detected_objects = detected_objects
        self.num_objects = len(detected_objects)
        self.object_type = object_type

    def to_zmq(self, update_type):
        """
        returns a formatted string that can be sent across zeromq
        """
        payload = {}
        payload["originatingTime"] = datetime.datetime.now().isoformat() #zmq expected format
        payload["update_type"] = str(update_type)
        message = {}
        message['image'] = np.array(self.image).tolist()
        message['detected_objects'] = self.detected_objects.tolist()
        message['num_objects'] = self.num_objects
        message['object_type'] = self.object_type
        payload["message"] = json.dumps(message)
        return payload

    def from_zmq(self, zmq_data):
        zmq_data = json.loads(zmq_data['message'])
        self.image =  Image.fromarray(np.array(zmq_data['image'], dtype='uint8'))
        self.detected_objects = zmq_data['detected_objects']
        self.payload = self.detected_objects
        self.num_objects = zmq_data['num_objects']

class ObjectFeaturesIU(retico_core.IncrementalUnit):
    """An image incremental unit that maintains a list of feature vectors for detected objects in a scene.

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
    """

    @staticmethod
    def type():
        return "Object Features IU"

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
        self.object_features = None
        self.num_objects = 0
        self.image = None

    def set_object_features(self, image, object_features):
        """Sets the content of the IU."""
        self.image = image
        self.payload = object_features
        self.object_features = object_features
        self.num_objects = len(object_features)

    def to_zmq(self, update_type):
        """
        returns a formatted string that can be sent across zeromq
        """
        payload = {}
        payload["originatingTime"] = datetime.datetime.now().isoformat() #zmq expected format
        payload["update_type"] = str(update_type)
        message = {}
        message['image'] = np.array(self.image).tolist()
        message['object_features'] = self.object_features
        message['num_objects'] = self.num_objects
        payload["message"] = json.dumps(message)
        return payload

    def from_zmq(self, zmq_data):
        zmq_data = json.loads(zmq_data['message'])
        self.image =  Image.fromarray(np.array(zmq_data['image'], dtype='uint8'))
        self.object_features = zmq_data['object_features']
        self.payload = zmq_data['object_features']
        self.num_objects = zmq_data['num_objects']

class WebcamModule(retico_core.AbstractProducingModule):
    """A module that produces IUs containing images that are captured by
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

class VideoPlaybackModule(retico_core.AbstractProducingModule):
    """A module that produces IUs containing images from a video file,
    simulating a webcam stream."""

    @staticmethod
    def name():
        return "Video Playback Module"

    @staticmethod
    def description():
        return "A producing module that plays back video files frame by frame."

    @staticmethod
    def output_iu():
        return ImageIU

    def __init__(self, video_path, width=None, height=None, fps=None, loop=True, **kwargs):
        """
        Initialize the Video Playback Module.
        
        Args:
            video_path (str): Path to the video file to play
            width (int): Width of the video frames; uses video's original width if None
            height (int): Height of the video frames; uses video's original height if None
            fps (float): Target frames per second for playback; uses video's original fps if None
            loop (bool): Whether to loop the video when it reaches the end
        """
        super().__init__(**kwargs)
        self.video_path = video_path
        self.width = width
        self.height = height
        self.fps = fps
        self.loop = loop
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.video_fps = 0
        self.frame_delay = 0
        self.last_frame_time = 0
        self._setup_video()

    def _setup_video(self):
        """Set up the video capture and get video properties."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Set target fps
        if self.fps is None:
            self.fps = self.video_fps

        # Calculate frame delay for timing
        self.frame_delay = 1.0 / self.fps if self.fps > 0 else 0
        
        print(f"Video loaded: {self.video_path}")
        print(f"Total frames: {self.total_frames}, Original FPS: {self.video_fps}, Target FPS: {self.fps}")

    def process_update(self, _):
        """Process update to read and send the next video frame."""
        current_time = time.time()
        
        # Control frame rate timing
        if self.last_frame_time > 0:
            elapsed = current_time - self.last_frame_time
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)
        
        self.last_frame_time = time.time()
        
        ret, frame = self.cap.read()
        
        if not ret:
            if self.loop and self.total_frames > 0:
                # Reset to beginning of video
                self.restart_video()
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame after reset")
                    return None
            else:
                print("End of video reached")
                return None
        
        self.current_frame += 1
        
        if ret:
            output_iu = self.create_iu()
            
            # Convert frame format if needed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            # Resize frame if width and height are specified
            if self.width is not None and self.height is not None:
                frame = frame.resize((self.width, self.height))

            output_iu.set_image(frame, 1, self.fps)
            return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        
        return None

    def setup(self):
        """Set up the video playback module."""
        if self.cap is None or not self.cap.isOpened():
            self._setup_video()

    def shutdown(self):
        """Release the video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def restart_video(self):
        """Restart the video from the beginning."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0

    def get_video_info(self):
        """Get information about the current video."""
        if self.cap is not None:
            return {
                'path': self.video_path,
                'total_frames': self.total_frames,
                'current_frame': self.current_frame,
                'fps': self.video_fps,
                'target_fps': self.fps,
                'duration': self.total_frames / self.video_fps if self.video_fps > 0 else 0,
                'current_time': self.current_frame / self.video_fps if self.video_fps > 0 else 0
            }
        return None        

    
class IPCameraModule(retico_core.AbstractProducingModule):
    """
    A module that produces IUs containing images from an IP camera feed.
    """

    @staticmethod
    def name():
        return "IP Camera Module"

    @staticmethod
    def description():
        return "A producing module that captures images from IP camera feeds."

    @staticmethod
    def output_iu():
        return ImageIU

    def __init__(self, 
                 camera_url, 
                 width=None, 
                 height=None, 
                 username=None, 
                 password=None,
                 # NOTE: Kept rate to be consistent with the WebcamModule but it seems does not actually set the FPS for either the webcam or IP camera.
                 rate=30, 
                 pil=True,
                 timeout=10,
                 retry_attempts=3,
                 retry_delay=5,
                 **kwargs):
        """
        Initialize the IP Camera Module.
        
        Args:
            camera_url (str): URL of the IP camera feed (e.g., "http://192.168.1.100:8080/video")
            width (int): Desired width of captured frames (optional)
            height (int): Desired height of captured frames (optional)
            username (str): Username for camera authentication (optional), used to create a URL with credentials 
            password (str): Password for camera authentication (optional), used to create a URL with credentials
            rate (int): Frame rate for capture (default: 30)
            pil (bool): Whether to convert frames to PIL Image format (default: True)
            timeout (int): Connection timeout in seconds (default: 10)
            retry_attempts (int): Number of retry attempts on connection failure (default: 3)
            retry_delay (int): Delay between retry attempts in seconds (default: 5)
        """
        super().__init__(**kwargs)
        
        self.camera_url = camera_url
        self.username = username
        self.password = password
        self.width = width
        self.height = height
        self.rate = rate
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.pil = pil
        
        self.cap = None
        self.is_connected = False
        self.connection_thread = None
        self.should_stop = False

    def setup(self):
        """Set up the IP camera connection."""
        self._connect_to_camera()
    
    def make_auth_url(self, raw_url, user, pwd):
        pu = urlparse(raw_url)
        auth_netloc = f"{quote(user)}:{quote(pwd)}@{pu.netloc}"
        pu = pu._replace(netloc=auth_netloc)
        return urlunparse(pu)
    
    def _connect_to_camera(self):
        """Establish connection to the IP camera with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                print(f"Attempting to connect to IP camera: {self.camera_url} (attempt {attempt + 1}/{self.retry_attempts})")
                
                # Create capture object with appropriate settings
                if self.username and self.password:
                    # For authenticated streams, modify URL to include credentials
                    auth_url = self.make_auth_url(self.camera_url, self.username, self.password)
                    self.cap = cv2.VideoCapture(auth_url)
                else:
                    self.cap = cv2.VideoCapture(self.camera_url)
                
                # Set capture properties
                if self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
                    
                    if self.rate:
                        self.cap.set(cv2.CAP_PROP_FPS, self.rate)
                
                # Test connection by reading a frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.is_connected = True
                    print("Successfully connected to IP camera")
                    
                    # Get actual frame properties
                    self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    
                    print(f"Camera properties - Width: {self.actual_width}, Height: {self.actual_height}, FPS: {self.actual_fps}")
                    return True
                else:
                    print("Failed to read frame from camera")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                        
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
            
            if attempt < self.retry_attempts - 1:
                print(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        print("Failed to connect to IP camera after all retry attempts")
        self.is_connected = False
        return False

    def process_update(self, _):
        """Process update to capture and return a frame from the IP camera."""
        if not self.should_stop and (not self.is_connected or not self.cap):
            self._connect_to_camera()
            return None
            
        try:
            ret, frame = self.cap.read()
            
            if ret:
                output_iu = self.create_iu()
                
                if self.pil:
                    # Convert BGR to RGB for PIL
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    
                    if self.width and self.height:
                        frame = frame.resize((self.width, self.height))
                
                output_iu.set_image(frame, 1, self.rate)
                return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            else:
                print("Failed to read frame from IP camera")
                self.is_connected = False
                
        except Exception as e:
            print(f"Error reading from IP camera: {str(e)}")
            self.is_connected = False

    def shutdown(self):
        """Close the IP camera connection."""
        self.should_stop = True
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        print("IP camera connection closed")
        
    
class ImageCropperModule(retico_core.AbstractModule):
    """A module that crops images"""

    @staticmethod
    def name():
        return "Image Cropper Module"

    @staticmethod
    def description():
        return "A module that crops images"


    @staticmethod
    def input_ius():
        return [ImageIU]

    @staticmethod
    def output_iu():
        return ImageIU

    def __init__(self, top=-1, bottom=-1, left=-1, right=-1, **kwargs):
        """
        Initialize the Webcam Module.
        Args:
            width (int): Width of the image captured by the webcam; will use camera default if unset
            height (int): Height of the image captured by the webcam; will use camera default if unset
            rate (int): The frame rate of the recording; will use camera default if unset
        """
        super().__init__(**kwargs)
        self.top =  top
        self.bottom = bottom
        self.left = left
        self.right = right

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            image = iu.image
            width, height = image.size
            left = self.left if self.left != -1 else 0
            top = self.top if self.top != -1 else 0
            right = self.right if self.right != -1 else width
            bottom = self.bottom if self.bottom != -1 else height
            image = image.crop((left, top, right, bottom)) 
            output_iu = self.create_iu(iu)
            output_iu.set_image(image, iu.nframes, iu.rate)
            return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        
        return None
        
        
class ExtractObjectsModule(retico_core.AbstractModule):
    """A module that produces image IUs containing detected objects segmented 
    by SAM or Yolo."""

    @staticmethod
    def name():
        return "Extract Object Module"

    @staticmethod
    def description():
        return "A module that produces iamges of individual objects from segmentations produced by SAM or Yolo."

    @staticmethod
    def input_ius():
        return [DetectedObjectsIU]
    
    @staticmethod
    def output_iu():
        return ExtractedObjectsIU

    def __init__(self, num_obj_to_display=1, show=False, save=False, keepmask=False, **kwargs):
        """
        Initialize the Display Objects Module
        Args:
            object_type (str): whether object is defined 
                in bounding box or segmentation
            num_obj_to_display (int): amount of objects from
                detected objects to display 
        """
        super().__init__(**kwargs)
        self.num_obj_to_display = num_obj_to_display
        self.show = show
        self.save = save
        self.keepmask = keepmask

    def process_update(self, update_message):
        # print("Extract Objects process update")
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                image_objects = {}
                output_iu = self.create_iu(iu)

                # img_dict = iu.payload
                image = iu.image

                obj_type = iu.object_type
                num_objs = iu.num_objects
                # print(f"Num Objects in Vsison: {num_objs}")

                if (self.num_obj_to_display > num_objs):
                    print("Number of objects detected less than requested. Showing all objects.")
                    self.num_obj_to_display = num_objs
                sam_image = np.array(image) #need image to be in numpy.ndarray format for methods
                if obj_type == 'bb':
                    valid_boxes = iu.payload
                    for i in range(num_objs):
                        res_image = self.extract_bb_object(sam_image, valid_boxes[i])
                        res_image = Image.fromarray(res_image)
                        if self.show:
                            res_image.show()
                        if self.save:
                            res_image.save('extr.jpg')
                        image_objects[f'object_{i+1}'] = res_image
                    output_iu.set_extracted_objects(image, image_objects, num_objs, obj_type)
                elif obj_type == 'seg':
                    valid_segs = iu.payload
                    for i in range(num_objs):
                        res_image = Image.fromarray(self.extract_seg_object(sam_image, valid_segs[i]))
                        if self.show:
                            res_image.show()
                        if self.save:
                            res_image.save('extr.jpg')
                        image_objects[f'object_{i+1}'] = res_image
                    output_iu.set_extracted_objects(image, image_objects, num_objs, obj_type)
                else: 
                    print('Object type is invalid. Can\'t retrieve segmented object.')
                    exit()

                # print(image_objects)

                # num_rows = math.ceil(self.num_obj_to_display / 3)
                # if self.num_obj_to_display < 3:
                #     num_cols = self.num_obj_to_display
                # else:
                #     num_cols = 3
                # fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows)) #need to adjust to have matching columsn and rows to fit num_obj_to_display
                # axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]

                # for i in range(self.num_obj_to_display):
                #     res_image = image_objects[f'object_{i+1}']
                #     axs[i].imshow(res_image)
                #     axs[i].set_title(f'Object {i+1}')
                
                # for j in range(self.num_obj_to_display, num_rows * num_cols):
                #     axs[j].axis('off')

                # folder_name = "extracted_objects"
                # if not os.path.exists(folder_name):
                #     os.makedirs(folder_name)
                
                # plt.tight_layout()
                # save_path = os.path.join(folder_name, f'top_{self.num_obj_to_display}_extracted_objs.png')
                # plt.savefig(save_path)
            output_iu.payload['num_objects'] = iu.num_objects
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD) 
            self.append(um)

    def extract_seg_object(self, image, seg):
        ret_image = image.copy()
        ret_image[seg==False] = [255, 255, 255]
        return ret_image
    
    def extract_bb_object(self, image, bbox):
        # Note: Masked and cropped both return np.ndarray of image

        #return a cut out of the bounding boxed object from the image
        if not self.keepmask:
            # Yolov8 returns bbox as left, top, right, bottom
            # casting to int does result in some data loss (potentially smaller or bigger bounding box)
            x1, y1, x2, y2 = [int(val) for val in bbox]
            ret_image = image[y1:y2, x1:x2]
        else: # Does not crop the image, rather keeps original image and whites out the area of the mask
            # keep position of object in image
            mask = np.zeros_like(image)
            x1, y1, x2, y2 = [int(val) for val in bbox] # cast to ints to circumvent issue with cv2 rect

            cv2.rectangle(mask, (0, 0), (image.shape[1], image.shape[0]), (255, 255, 255), -1)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            mask = cv2.bitwise_not(mask)
            
            ret_image = cv2.bitwise_and(image, image, mask=mask)

            ret_image[mask == 0] = [255, 255, 255]

        # ret_image = cv2.cvtColor(ret_image, cv2.COLOR_RGB2BGR)
        return ret_image 

        

class ExtractedObjectsIU(retico_core.IncrementalUnit):
    """A dictionary incremental unit that maintains a dictionary of objects segmented from an Image
    
    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the c
            current one
        grounded_in (IncrementalUnit): A link to the IU this IU is based on
        created_at (float): The UNIX timestamp of the moment the IU is created
    """

    @staticmethod
    def type():
        return "Extracted Objects IU"
    
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
        self.num_objects = 0
        self.object_type = None
        self.extracted_objects = {}

    def set_extracted_objects(self, image, objects_dictionary, num_objects, object_type,):
        """Sets the content for the IU"""
        self.image = image
        self.payload = objects_dictionary
        self.num_objects = num_objects
        self.object_type = object_type
        self.extracted_objects = objects_dictionary

    def to_zmq(self, update_type):
        """
        returns a formatted string that can be sent across zeromq
        """
        payload = {}
        payload["originatingTime"] = datetime.datetime.now().isoformat() #zmq expected format
        payload["update_type"] = str(update_type)
        message = {}
        message['image'] = np.array(self.image).tolist()
        message['segmented_objects_dictionary'] = self.extracted_objects
        message['num_objects'] = self.num_objects
        payload["message"] = json.dumps(message)
        return payload

    def from_zmq(self, zmq_data):
        zmq_data = json.loads(zmq_data['message'])
        self.image =  Image.fromarray(np.array(zmq_data['image'], dtype='uint8'))
        self.num_objects = zmq_data['num_objects']
        self.extracted_objects = zmq_data['segmented_objects_dictionary']
        self.payload = self.extracted_objects
                
class HandPositionsIU(ObjectFeaturesIU):
    """An image incremental unit that maintains a list of multi_hand_landmarks (points on hand) and multi_handedness (whether each hand is left or gith).

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
    """
    @staticmethod
    def type():
        return "Hand Positions IU"
    
    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                 rate=None, nframes=None, sample_width=None, raw_audio=None,
                 **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=None)
        self.multi_hand_landmarks = None
        self.multi_handedness = None
        self.image = None

    def set_landmarks(self, image, multi_hand_landmarks, multi_handedness):
        "Sets landmark content of the IU"
        self.image = image
        self.payload = multi_hand_landmarks
        self.multi_handedness = multi_handedness

    def payload_to_vector(self, count):
        if self.payload is None: return
        tempPayload = self.payload
        self.payload = {}
        # print(self.payload[0])
        for hand_index, hand_landmarks in enumerate(tempPayload):
            # Iterate over the detected landmarks of the hand.
            vector = []
            for landmark in hand_landmarks.landmark:
                vector.append(landmark.x)
                vector.append(landmark.y)
                vector.append(landmark.z)
            if hand_index == 0:
                self.payload['hand' + str(hand_index)] = np.array(vector)
            # print(len(vector))

            # print("vector: ", self.payload)
        # vector = []
        # if self.payload is None: return
        # print("size ", len(self.payload))
        # for landmark in self.payload[0]:
        #     print(type(landmark))
        #     print(landmark)
        #     vector.append(landmark.x)
        #     vector.append(landmark.y)
        #     vector.append(landmark.z)
        #     print(self.payload)
        # self.payload = np.array(self.payload)


    def get_json(self):
        payload = {}
        payload['image'] = np.array(self.payload).tolist()
        payload['multi_hand_landmarks'] = self.multi_hand_landmarks
        payload['multi_handedness'] = self.multi_handedness
        return payload

    def create_from_json(self, json_dict):
        self.image =  Image.fromarray(np.array(json_dict['image'], dtype='uint8'))
        self.multi_hand_landmarks = json_dict['multi_hand_landmarks']
        self.payload = self.multi_hand_landmarks
        self.multi_handedness = json_dict['multi_handedness']

class PosePositionsIU(retico_core.IncrementalUnit):
    """An image incremental unit that maintains a list of pose_landmarks (points on body) and segmentation_mask (pixel values where 1 represents human pixel and 0 represents background pixel)).

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
    """
    @staticmethod
    def type():
        return "Pose Positions IU"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                    rate=None, nframes=None, sample_width=None, raw_audio=None,
                    **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                            grounded_in=grounded_in, payload=None)
        self.pose_landmarks = None
        self.segmentation_mask = None
        self.image = None

    def set_landmarks(self, image, pose_landmarks, segmentation_mask):
        "Sets landmark content of the IU"
        self.image = image
        self.payload = pose_landmarks
        self.segmentation_mask = segmentation_mask

    def get_json(self):
        payload = {}
        payload['image'] = np.array(self.payload).tolist()
        payload['pose_landmarks'] = self.pose_landmarks
        payload['segmentation_mask'] = self.segmentation_mask
        return payload

    def create_from_json(self, json_dict):
        self.image =  Image.fromarray(np.array(json_dict['image'], dtype='uint8'))
        self.pose_landmarks = json_dict['pose_landmarks']
        self.payload = self.pose_landmarks
        self.segmentation_mask = json_dict['segmentation_mask']

class ScreenModule(retico_core.AbstractConsumingModule):
    # Class variable to track the root window
    _tk_root = None

    """
    A module that displays images on the screen.
    """

    @staticmethod
    def name():
        return "ScreenModule"

    @staticmethod
    def description():
        return "A module that displays images on the screen."

    @staticmethod
    def input_ius():
        return [ImageIU]

    def __init__(self, fps=30):
        """
        Initialize the ScreenModule with a specified frames per second (fps).
        Note that the fps parameter is an upper limit for the display refresh rate.
        
        Args:
            fps (int): The desired frames per second for the display. Default is 30.
        """
        super().__init__()
        self._running = False
        self.fps = fps
        self.queue = deque(maxlen=1)
        self.display_thread = None
        self.root = None
        self.label = None
        self.is_main_window = False

    def setup(self):
        """
        Set up the module.
        """
        self._running = True
        
        self._init_gui()
        self._schedule_update()
        
    def _init_gui(self):
        """
        Initialize the GUI components. Called from main thread on macOS.
        """
        try:
            # Create tkinter window: only one Tk, others are Toplevel
            if ScreenModule._tk_root is None:
                self.root = tk.Tk()
                ScreenModule._tk_root = self.root
                self.is_main_window = True
            else:
                self.root = tk.Toplevel(ScreenModule._tk_root)
                self.is_main_window = False
            
            self.root.title("Screen Module")
            
            # Create label to hold the image
            self.label = Label(self.root)
            self.label.pack(expand=True, fill='both')
            
        except Exception as e:
            print(f"Error initializing GUI: {e}")

    def _schedule_update(self):
        """
        Schedule the next update. Used on macOS to avoid threading issues.
        """
        if self._running and self.root and self.root.winfo_exists():
            self._update_display()
            # Schedule next update
            self.root.after(int(1000 / self.fps), self._schedule_update)

    def shutdown(self):
        """
        Clean up the module, e.g., close display windows.
        """
        self._running = False
        
        # Close tkinter window if it exists
        if self.root:
            self.root.quit()
            

    def _update_display(self):
        """
        Update the display with new images from the queue.
        """
        try:
            # Check if root and label exist and window is not destroyed
            if not (self._running and self.root and self.label and self.root.winfo_exists()):
                return

            if len(self.queue) > 0:
                image = self.queue.popleft()
                if image is not None:
                    # Resize image if necessary
                    if hasattr(image, 'size'):
                        pil_image = image
                    else:
                        pil_image = Image.fromarray(image)

                    # Convert to tkinter PhotoImage
                    photo = ImageTk.PhotoImage(pil_image)

                    # Update the label with new image
                    self.label.configure(image=photo)
                    self.label.image = photo  # Keep a reference

        except Exception as e:
            print(f"Error updating display: {e}")
            # If the error is due to a missing image or destroyed widget, stop updating
            if not (self.root and self.root.winfo_exists()):
                return

    def process_update(self, update_message):
        for iu, um in update_message:
            if isinstance(iu, ImageIU):
                image = iu.image
                if self._running and image is not None:
                    # Queue the image for display
                    self.queue.append(image)

class Convert_DetectedObjectsIU_ImageIU(retico_core.AbstractModule):
    """
    A module that converts DetectedObjectsIU to ImageIU for display.
    """
    @staticmethod
    def name():
        return "Convert_DetectedObjectsIU_ImageIU"

    @staticmethod
    def description():
        return "Converts DetectedObjectsIU to ImageIU for display."

    @staticmethod
    def input_ius():
        return [DetectedObjectsIU]

    @staticmethod
    def output_iu():
        return ImageIU

    def __init__(self, num_obj_to_display=1):
        super().__init__()
        self.num_obj_to_display = num_obj_to_display

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            elif isinstance(iu, DetectedObjectsIU):
                # Convert DetectedObjectsIU to ImageIU and draw bounding boxes
                image : Image = iu.image

                output_iu = self.create_iu(iu)

                obj_type = iu.object_type
                num_objs = min(self.num_obj_to_display, iu.num_objects)

                if obj_type == 'bb':
                    valid_boxes = iu.payload
                    for i in range(num_objs):
                        box = valid_boxes[i]
                        if box is not None:
                            x1, y1, x2, y2 = box
                            # Draw bounding box on the image
                            img_draw = ImageDraw.Draw(image)
                            img_draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                            # put a label on the box
                            img_draw.text((x1, y1), f'Object {i+1}', fill='red')
                elif obj_type == 'seg':
                    valid_segs = iu.payload
                    for i in range(num_objs):
                        seg_mask = valid_segs[i]
                        # seg_mask is expected to be a binary mask
                        if seg_mask is not None:
                            # Convert the mask to a PIL Image and apply it to the original with transparency
                            seg_mask_image = Image.fromarray(seg_mask.astype('uint8') * 255)
                            # blend the mask with the original image
                            image = Image.composite(image, Image.new('RGB', image.size, (255, 0, 0)), seg_mask_image)
                            # put a label on the image
                            img_draw = ImageDraw.Draw(image)
                            img_draw.text((10, 10 + i * 20), f'Segmented Object {i+1}', fill='red')
                else: 
                    print('Object type is invalid. Can\'t retrieve segmented object.')
                    exit()

                output_iu.image = image
                um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD) 
                self.append(um)

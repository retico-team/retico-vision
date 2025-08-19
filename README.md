# retico-vision
A ReTiCo module with base modules and incremental units for computer vision.

### Installation and requirements

Requires [retico-core](https://github.com/retico-team/retico-core).

Install the retico-vision package:  
```pip install git+https://github.com/retico-team/retico-vision.git```

### Example
```python
import sys, os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

prefix = '/path/to/prefix'
sys.path.append(prefix+'retico-core')
sys.path.append(prefix+'retico-vision')
sys.path.append(prefix+'retico-sam')
sys.path.append(prefix+'retico-dino')

from retico_core import *
from retico_core.debug import DebugModule
from retico_vision.vision import WebcamModule 
from retico_dino.dino import Dinov2ObjectFeatures
from retico_vision.vision import ExtractObjectsModule
from retico_sam.sam import SAMModule

path_var = 'sam_vit_h_4b8939.pth'

webcam = WebcamModule()
sam = SAMModule(model='h', path_to_chkpnt=path_var, use_bbox=True)  
extractor = ExtractObjectsModule(num_obj_to_display=1)  
feats = Dinov2ObjectFeatures(show=False, top_objects=1)
debug = DebugModule()  

webcam.subscribe(sam)  
sam.subscribe(extractor)  
extractor.subscribe(feats)    
feats.subscribe(debug)

webcam.run()  
sam.run()  
extractor.run()  
feats.run()
debug.run()  

print("Network is running")
input()

webcam.stop()  
sam.stop()  
extractor.stop()   
debug.stop()  
```


Citation
```
```
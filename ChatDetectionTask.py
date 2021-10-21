import glob
import numpy as np
from java.lang import System
import re, os, itertools, time
from keras.models import load_model
from keras.preprocessing.image import image

class ChatDetectionTask:
    
    enabled = False
    configDir = None
    
    def isEnabled(self):
        return True
        
    def init(self, confProps, configFolder):  
        classifierModel = System.getProperty('iped.root') + '/models/model.h5'
        global classifier
        classifier = load_model(classifierModel)
        return
    
    def finish(self):      
        return 
        
    def process(self, item):
       
       categories = item.getCategorySet().toString()
       if not ("Images" in categories):
          return

       def load_image(img_path, show=True):
            img_original = image.load_img(img_path)
            img = image.load_img(img_path, target_size=(64, 64))
            img_tensor = image.img_to_array(img)                    # (height, width, channels)
            img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
            img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
            return img_tensor
      
       path = item.getTempFile().getAbsolutePath()
       new_image = load_image(path)
       pred = classifier.predict(new_image)        # predict() function may be used when flask is not used 
       if pred<.5 : 
           item.setExtraAttribute('possiblyChat', "Possível presença de chat")
       else: 
           return
            

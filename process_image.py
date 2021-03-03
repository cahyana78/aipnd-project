from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    size = 256, 256
    image = Image.open(image)
    image.thumbnail(size, Image.ANTIALIAS)
    
    new_width = 224
    new_weight = 224
    
    # get the dimension
    width, height = image.size
    
    left = (width - new_width)/2
    top = (height - new_weight)/2
    right = (width + new_width)/2
    bottom = (height + new_weight)/2
    
    # crop the center of image
    image = image.crop((left, top, right, bottom))
    
    # get the color
    np_image = np.array(image)/255.0 #convert to 0-1
    image_mean = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - image_mean)/image_std
    np_image = np.transpose(np_image, (2, 0, 1))
    
    image = torch.from_numpy(np_image).type(torch.FloatTensor) #https://knowledge.udacity.com/questions/381618
    
    return image
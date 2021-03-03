from torchvision import models

def pretrained_model(arch):
    if arch == None or arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'vgg':
        model = models.vgg16(pretrained=True)
    else:
        print('Please use densenet, alexnet, or vgg only')
        
        
    return model
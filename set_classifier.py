# https://knowledge.udacity.com/questions/23257
# https://knowledge.udacity.com/questions/269954
from torch import nn, optim
from collections import OrderedDict
from torchvision import models

def set_classifier(model, hidden_units):
    #model =  getattr(models, arch)(pretrained=True) # https://knowledge.udacity.com/questions/262667
    
    if hidden_units == None:
        hidden_units = 512
        
        
    # https://knowledge.udacity.com/questions/384946
    if model == 'densenet121':
        input = model.classifier.in_features  
        
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1',nn.Linear(input, hidden_units)),
                                  ('relu',nn.ReLU()),
                                  ('dropout',nn.Dropout(p=0.2)),
                                  ('fc2',nn.Linear(hidden_units, 102)),
                                  ('output',nn.LogSoftmax(dim=1))])
                          )
    
        model.classifier = classifier
      
        
    elif model == 'alexnet':
        input = model.model.classifier[1].in_features
        
        classifier = nn.Sequential(nn.Linear(input, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1)
                          )
        model.classifier = classifier
        
    elif model == 'vgg16':
        input = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input, hidden_units, bias=True)),
                                                ('relu1', nn.ReLU()),
                                                ('dropout', nn.Dropout(p=0.5)),
                                                ('fc2', nn.Linear(hidden_units, 128, bias=True)),
                                                ('relu2', nn.ReLU()),
                                                ('dropout', nn.Dropout(p=0.5)),
                                                ('fc3', nn.Linear(128, 102, bias=True)),
                                                ('output', nn.LogSoftmax(dim=1))
                                               ]))
        model.classifier= classifier
      

        
              
    
    #criterion = nn.NLLLoss()
    
            
    #optimizer = optim.Adam((model.parameters()), lr)
    
    return model
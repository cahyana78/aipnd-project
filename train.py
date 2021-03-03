# https://knowledge.udacity.com/questions/306359
# https://knowledge.udacity.com/questions/91472
# https://knowledge.udacity.com/questions/123966
# https://knowledge.udacity.com/questions/123031 --> baca ini lagi 
# https://knowledge.udacity.com/questions/23257 --> yang ini

from get_input_args import get_input_args
from process_data import process_data
from pretrained_model import pretrained_model
from set_classifier import set_classifier
from train_model import train_model
from save_checkpoint import save_checkpoint
from torch import nn, optim

from torchvision import models

def main():
    # Read arguments from command line
    in_args = get_input_args()
    
    train_dir = in_args.dir + '/train'
    valid_dir = in_args.dir + '/valid'
    test_dir = in_args.dir + '/test'
    
    trainloaders, testloaders, validloaders = process_data(train_dir, test_dir, valid_dir)
    model = pretrained_model(in_args.arch)
    
    for name, child in model.named_children():
        print(name)
        
    #model =  getattr(models, in_args.arch)(pretrained=True) # https://knowledge.udacity.com/questions/262667
    
   
    for param in model.parameters():
        param.requires_grad = False
           
    model = set_classifier(model, in_args.hidden_units)
    
    for name, child in model.named_children():
        print(name)
        
    model.classifier.requires_grad = True
    
    #params_to_update = []
   # for name, param in model.named_parameters():
    #    if param.requires_grad == True:
    #        params_to_update.append(param)
    #print(params_to_update)
    
    #criterion = nn.NLLLoss()
    
   # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=in_args.lr)
    
    #for param in model.parameters():
    #    param.requires_grad = False    
        
    params_to_update = []
   
    

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)          
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params_to_update, lr = in_args.lr)
    
            
    #optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.lr)
    
   # trmodel = train_model(in_args.epochs, trainloaders, validloaders, in_args.gpu, model, optimizer, criterion)
    #valid_model(trmodel, testloaders, in_args.gpu)
    #save_checkpoint(in_args.arch, in_args.epochs, model, optimizer, in_args.save_dir)
    print('Completed!')
    
if __name__ == '__main__': main()
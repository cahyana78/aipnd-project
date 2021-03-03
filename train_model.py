import torch
from torch import nn, optim
import torch.nn.functional as F


# https://knowledge.udacity.com/questions/23257
def train_model(epochs, trainloaders, validloaders, gpu, model, optimizer, criterion):
    if type(epochs) == type(None):
        epochs = 10
        print("Epochs = 10")
    
    model.to('cuda')
    steps = 0
    running_loss = 0
    print_every = 30
    
    for epoch in range(epochs):
        for images, labels in trainloaders:
            steps += 1
            
            if gpu==True:
                images, labels = images.to('cuda'), labels.to('cuda')           
                optimizer.zero_grad()
            
                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()           
                
                if steps % print_every == 0:
                    model.eval()
                    test_loss = 0
                    accuracy = 0        
                    
                    for images, labels in validloaders:
                        images, labels = images.to('cuda'), labels.to('cuda')
                        logps = model(images)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()
                    
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()     
                        
                        print(f"Epoch {epoch+1}/{epochs}.. "
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"Test loss: {test_loss/len(test_dataloaders):.3f}.. "
                              f"Test accuracy: {accuracy/len(test_dataloaders):.3f}")        
                        
                        running_loss = 0
                        
                        model.train()
                        
    return model
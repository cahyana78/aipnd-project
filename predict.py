# https://knowledge.udacity.com/questions/283565
import json
from predict_get_input_args import predict_get_input_args
from load_checkpoint import load_checkpoint
from process_image import process_image


def main():
    in_args = predict_get_input_args() # Read arguments from command line
    
    #  Read the cat_to_name file from command line argument parser and extract file content into dictionary using json.load.
    with open(in_args.cat_name, 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(in_args.check)
    
    im = process_image(in_args.image)

    prob, classes = predict(im, model, in_args.top, in_args.gpu)
    
    probability = prob.tolist() # https://knowledge.udacity.com/questions/284170
    probability.flatten() # https://knowledge.udacity.com/questions/284170
    names = [cat_to_name[i] for i in classes]
    
    # Display top k most probable flower categories
    # courtesy to Jim C, https://knowledge.udacity.com/questions/284170
    
    for i in range (in_args.top):
        print("Number: {}".format(i))
        print("Flower class name: {}".format(names[i]))
        print("Probability: {:.3f}% ".format(probability [i]*100))
                              
  


  # code inside main 
if __name__ == "__main__":
    main()
    




  
def predict(image_path, model, topK, gpu):
   ''' Predict topK classes with highest probability.
   '''
    # https://knowledge.udacity.com/questions/381618
    #model = model.to(device)
    model.eval()
    
    # https://knowledge.udacity.com/questions/381480
    
    image = process_image(image_path)
    image.unsqueeze_(0)
    
    if gpu==True:
        with torch.no_grad():
            model = model.to(device)
            image = image.to(device)
            logps = model.forward(image)
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(topk, dim=1)
            
            class_to_idx = model.class_to_idx # https://knowledge.udacity.com/questions/459921
            idx_to_class = {v: k for k,v in model.class_to_idx.items()}
            
            prob = [p.item() for p in top_ps[0].data]
            classes = [idx_to_class[i.item()] for i in top_class[0].data]
            
            model.train()   
            
    return prob, classes
   


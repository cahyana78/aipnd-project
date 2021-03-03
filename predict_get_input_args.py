# From project 1
# https://knowledge.udacity.com/questions/56461

import argparse

def predict_get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image path to be predicted as --image, required
      2. Checkpoint of previous training model as --check, required
      3. Number of top-k probabilities for class prediction as --top with default value 5
      4. Json file to map category to flower name as --cat_name with default value cat_to_name.json
      5. The using of GPU or CPU as --gpu with default value True
      
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--image', type = str, help = 'path to the flower images', required=True)
    parser.add_argument('--check', type = str, help = 'the checkpoint of training model', required=True)
    parser.add_argument('--top', type=int, default=5, help='the top-k probabilities for the image')
    parser.add_argument('--cat_name', type = str, default = 'cat_to_name.json', help='json filename with categories to map classnames')  
    parser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')    
       
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

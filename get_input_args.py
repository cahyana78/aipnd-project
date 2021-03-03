# From project 1
# https://knowledge.udacity.com/questions/23257
# https://knowledge.udacity.com/questions/123031
import argparse

def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'flowers'
      2. CNN Model Architecture as --arch with default value 'densenet121'
      3. Number of hidden units of the CNN architecture as --hidden_units with default value 512
      4. Number of learning rate for the CNN architecture as --lr with default value 0.002
      5. The using of GPU or CPU as --gpu with default value True
      6. Number of epochs of the CNN architecture as --epochs with default value 10
      7. Path to save the checkpoint as --save_dir with default value checkpoint.pth
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--dir', type = str, default = 'flowers', help = 'path to the folder of directory flower images')
    parser.add_argument('--arch', type = str, default = 'densenet121', help = 'the CNN model architecture to be used, choose densenet121 or alexnet')
    parser.add_argument('--hidden_units', type=int, default=[512], help='hidden units for layer')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')  
    parser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')    
    parser.add_argument('--epochs', type=int, default=10, help='num of epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save train model to a file')

    
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

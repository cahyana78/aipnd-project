def load_checkpoint(filepath):
  '''
     Load a checkpoint from filepath and rebuild model.  
  '''
    
    checkpoint= torch.load(filepath, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
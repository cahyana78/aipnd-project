# https://knowledge.udacity.com/questions/23257

def save_checkpoint(arch, epochs, model, optimizer, save_dir):
    model.class_to_idx = train_datasets.class_to_idx
    
    checkpoint = {'arch': arch,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    
    return torch.save(checkpoint, save_dir)
    
    
    checkpoint = {'structure': Model.name,
                  'classifier': Model.classifier,
                  'state_dic': Model.state_dict(),
                  'class_to_idx': Model.class_to_idx}
  
import os
import torch

def save_model(args, model, current_epoch):
    out = os.path.join(args.model_path,"checkpoint_{}.tar".format(current_epoch))
    state = model.state_dict()
    torch.save(state, out)

import torch
import torch.nn.functional as F
import utils
import data_loader
import models
import os
import train

def test(args, model_=None):
    device=utils.device_select(args.gpu_number)
    node_features, edge_index, node_type=data_loader.load_data(args.data, args.data_path)
    if model_==None:
        model=models.GCN(channel_in=len(node_features[0]), channel_middle=args.hidden, channel_out=len(list(set(node_type))))
        model.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__))+"/check_points/"+"best.pt"))
        #model=torch.load(os.path.dirname(os.path.realpath(__file__))+"/check_points/"+"best.pt")
    else:
        model=model_
    model=model.to(device)
    model.eval()

    node_features=node_features.to(device)
    edge_index=edge_index.to(device)
    node_type=node_type.to(device)
    out = model(node_features, edge_index)
    loss = F.nll_loss(out, node_type)
    print("loss: "+str(loss))
    return 



# If this file is executed directly from terminal
if __name__ == "__main__":
    args=utils.parser()
    modeling=train.train(args)
    #test(args, model_=modeling)
    test(args)
import torch
import torch.nn.functional as F
import utils
import data_loader
import models

def train(args):
    device=utils.device_select(args.gpu_number)
    node_features, edge_index, node_type=data_loader.load_data(args.data, args.data_path)
    model=models.GCN(channel_in=len(node_features[0]), channel_middle=args.hidden, channel_out=len(list(set(node_type)))).to(device)
    node_features=node_features.to(device)
    edge_index=edge_index.to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(node_features, edge_index)
        loss = F.nll_loss(out, node_type)
        print(loss)
        loss.backward()
        optimizer.step()



# If this file is executed directly from terminal
if __name__ == "__main__":
    args=utils.parser()
    train(args)
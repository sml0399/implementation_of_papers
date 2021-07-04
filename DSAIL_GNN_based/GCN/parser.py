import  argparse

def parser():
    args = argparse.ArgumentParser()

    args.add_argument('--data', default='cora', help="set dataset type. Default is cora dataset")
    args.add_argument('--model', default='gcn', help="set GNN model. Default is gcn")
    args.add_argument('--lr', type=float, default=0.01, help="set learning rate. default is 0.01")
    args.add_argument('--epochs', type=int, default=200, help="set epochs. Default is 200")
    args.add_argument('--hidden', type=int, default=16, help="set number of hidden layers. Default is 16")
    args.add_argument('--dropout', type=float, default=0.5, help="set dropout ratio. Default is 0.5")
    args.add_argument('--weight_decay', type=float, default=5e-4, help="set weight decay. Default is 5e-4")
    args.add_argument('--early_stopping_size', type=int, default=10, help="set early stopping window size(successive number of occurence of val_error >= train_error). Default is 10")
    args.add_argument('--max_degree', type=int, default=3, help="set max degree. Default is 3")

    return args.parse_args()

# If this file is executed directly from terminal
if __name__ == "__main__":
	parser()
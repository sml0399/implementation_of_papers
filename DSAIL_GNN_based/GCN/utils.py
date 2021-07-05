import argparse
import torch

def device_select(gpu_number=0):
    '''
    Description: 
        Function for selecting the device to be used for training/testing models. 
        If cuda is available, this function will select gpu with given gpu_number. Default value is 0
        If cuda is not available, cpu will be chosen as device.

    Input: 
        gpu_number(int): number representing the gpu_number.

    Output: 
        device(string): string representing device to be used. One of "cpu", "cuda:<gpu_number>"
    '''


    device =""
    if torch.cuda.is_available():
        device ="cuda:"+str(gpu_number)
    else:
        device="cpu"
    print("using device: "+device)
    return device


def parser():
    '''
    Description: 
        Function for parsing arguments when executing python program file.

    Input: 
        None

    Output: 
        parser: refer to https://docs.python.org/3/howto/argparse.html

    '''


    args = argparse.ArgumentParser()
    args.add_argument('--data', default='cora', help="set dataset type. Default is cora dataset. Available datasets are: cora, citeseer, nell, user_defined. In order to use user_defined, the path of the data must be given by --data_path and data must have node features and edge_index")
    args.add_argument('--data_path', default='', help="ONLY used when you set --data as 'user_defined'. set path of user_defined dataset. Default is ''. You must change load_user_defined function at utils.py")
    args.add_argument('--model', default='gcn', help="set GNN model. Default is gcn")
    args.add_argument('--lr', type=float, default=0.01, help="set learning rate. default is 0.01")
    args.add_argument('--epochs', type=int, default=200, help="set epochs. Default is 200")
    args.add_argument('--hidden', type=int, default=16, help="set number of hidden layers. Default is 16")
    args.add_argument('--dropout', type=float, default=0.5, help="set dropout ratio. Default is 0.5")
    args.add_argument('--weight_decay', type=float, default=5e-4, help="set weight decay. Default is 5e-4")
    args.add_argument('--early_stopping_size', type=int, default=10, help="set early stopping window size(successive number of occurence of val_error >= train_error). Default is 10")
    args.add_argument('--max_degree', type=int, default=3, help="set max degree. Default is 3")
    args.add_argument('--gpu_number', type=int, default=0, help="set the gpu number to be used. Default is 0th GPU")
    return args.parse_args()





# If this file is executed directly from terminal
if __name__ == "__main__":
    args=parser()
    device_select(args.gpu_number)

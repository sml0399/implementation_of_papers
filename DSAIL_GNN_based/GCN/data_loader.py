import utils
import sys
import os
import numpy as np
import torch

def load_user_defined(data_path):
    '''
    TODO:
    In order to load user_defined data, you must fill this function. If you use only dataset in this repository, you don't need to implement this function.

    Description:
        load graph data at 'data_path' and return node_features and edge_index.
        Manually create node_feature vectors and edge_index vectors. 

    Input:  
        data_path(string): path of the dataset location

    Output: 
        node_features(tensor): initial node features. Shape must be [ number_of_nodes, number_of_features_per_node]
        edge_index(tensor): list storing edges. Edges are represented by pair of node index. Shape must be [2, number_of_edges]
        node_type(tensor): list of node types. Shape must be [ number_of_nodes ]
    '''
    return [0],[0]



def load_data(data_name, user_defined_data_path=None):
    '''
    Description:
        Function for loading data
        If user uses only dataset in the repository, data_name will be one of them( cora, citeseer, nell). user_defined_data_path is not used in this case
        If user tries to use user defined dataset, data_name will be 'user_defined'. User needs to implement load_user_defined function manually. 
            user_defined_data_path will be given by user. Just toss this information to load_user_defined function.
        Otherwise, other types of data_name will be treated as error
    
    Input:  
        data_name(string): name of the dataset. One of cora, citeseer, nell, user_defined
        user_defined_data_path(string): path of the dataset location

    Output: 
        node_features(tensor): initial node features. Shape must be [ number_of_nodes, number_of_features_per_node]
        edge_index(tensor): list storing edges. Edges are represented by pair of node index. Shape must be [2, number_of_edges]
        node_type(tensor): list of node types. Shape must be [ number_of_nodes ]
    
    
    '''

    # get the root path of this github repository
    root_path=os.path.abspath(__file__)
    while(1):
        if(root_path[-24:]=="implementation_of_papers"):
            break
        else:
            root_path=os.path.dirname(root_path)
    root_path=root_path+"/"
    

    # Now set data_path (implemenation_of_papers/dataset/<dataset_name>) and load data
    if data_name=="cora":
        data_path=root_path+"dataset/cora/"

        content=open(data_path+"cora.content","r")                                               # node feature file
        content_parsed1=content.read().split('\n')[:-1]                                          # divide by line
        content.close()
        content_parsed2=sorted([a.split('\t') for a in content_parsed1], key=lambda x:int(x[0])) # parse by tab and sort by node number
        node_numbers=[int(a[0]) for a in content_parsed2]                                             # get set of node numbers
        node_type=[a[-1] for a in content_parsed2]                                               # list of type per node
        node_features=[[int(b) for b in a[1:-1]] for a in content_parsed2]                                         # list of features per node
        type_list=list(set(node_type))
        node_type=[type_list.index(a) for a in node_type]

        edge_list=open(data_path+"cora.cites","r")                                               # edge (node pairs) file
        edge_list_parsed1=edge_list.read().split('\n')[:-1]                                      # divide by line
        edge_list.close()
        edge_list_parsed2=[a.split('\t') for a in edge_list_parsed1]                             # parse by tab 
        edge_index=[[int(a[0]), int(a[1])] for a in edge_list_parsed2]                           # edge list holding (node number(int), node number(int)) pairs
        edge_index=np.array(edge_index).T.tolist()                                               # make transpose of list to make shape [2, number_of_edges]
        for i in range(len(edge_index[0])):                                                      # converting node numbers
            edge_index[0][i]=node_numbers.index(edge_index[0][i])
            edge_index[1][i]=node_numbers.index(edge_index[1][i])
            
        return torch.Tensor(node_features), torch.Tensor(edge_index).to(torch.int64), torch.Tensor(node_type).to(torch.int64)



    elif data_name=="citeseer":
        data_path=root_path+"dataset/citeseer/"

        content=open(data_path+"citeseer.content","r")                                               # node feature file
        content_parsed1=content.read().split('\n')[:-1]                                          # divide by line
        content.close()
        content_parsed2=sorted([a.split('\t') for a in content_parsed1], key=lambda x:x[0]) # parse by tab and sort by node number
        node_numbers=[a[0] for a in content_parsed2]                                             # get set of node numbers
        node_type=[a[-1] for a in content_parsed2]                                               # list of type per node
        node_features=[[int(b) for b in a[1:-1]] for a in content_parsed2]                                         # list of features per node
        type_list=list(set(node_type))
        node_type=[type_list.index(a) for a in node_type]

        edge_list=open(data_path+"citeseer.cites","r")                                               # edge (node pairs) file
        edge_list_parsed1=edge_list.read().split('\n')[:-1]                                      # divide by line
        edge_list.close()
        edge_list_parsed2=[a.split('\t') for a in edge_list_parsed1]                             # parse by tab 
        edge_index=[[a[0], a[1]] for a in edge_list_parsed2]                           # edge list holding (node number(int), node number(int)) pairs
        edge_index=np.array(edge_index).T.tolist()                                               # make transpose of list to make shape [2, number_of_edges]
        i=0;j=0
        total_len=len(edge_index[0])
        while((i+j)<total_len):                                                       # converting node numbers
            if (edge_index[0][i] not in node_numbers) or (edge_index[1][i] not in node_numbers):
                j+=1
                del edge_index[0][i]
                del edge_index[1][i]
                continue 
            edge_index[0][i]=node_numbers.index(edge_index[0][i])
            edge_index[1][i]=node_numbers.index(edge_index[1][i])
            i+=1
            
        return torch.Tensor(node_features), torch.Tensor(edge_index).to(torch.int64), torch.Tensor(node_type).to(torch.int64)



############################ TODO: load data and return node_features and edge_index   ##################################
    elif data_name=="nell":
        data_path=root_path+"dataset/nell/"

        return




    elif data_name=="user_defined": # user_defined case. toss to load_user_defined function
        return load_user_defined(user_defined_data_path)

    else: # will be treated as error
        print("error in data name")
        sys.exit()






# If this file is executed directly from terminal
if __name__ == "__main__":
    args=utils.parser()
    load_data(args.data, args.data_path)
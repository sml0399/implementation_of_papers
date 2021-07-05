import utils
import sys
import os

def load_data(data_name, user_defined_data_path=None):
    '''
    Description:
        Function for loading data
        If user uses only dataset in the repository, data_name will be one of them( cora, citeseer, nell). user_defined_data_path is not used in this case
        If user tries to use user defined dataset, data_name will be 'user_defined'. User needs to implement load_user_defined function manually. 
            user_defined_data_path will be given by user. Just toss this information to load_user_defined function.
        Otherwise, other types of data_name will be treated as error
    
    Input:  
        data_name: name of the dataset. One of cora, citeseer, nell, user_defined
        user_defined_data_path: path of the dataset location

    Output: 
        node_features: initial node features. Shape must be [ number_of_nodes, number_of_features_per_node]
        edge_index: list storing edges. Edges are represented by pair of node index. Shape must be [2, number_of_edges]
    
    
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
        return
    elif data_name=="citeseer":
        data_path=root_path+"dataset/citeseer/"
        return
    elif data_name=="nell":
        data_path=root_path+"dataset/nell/"
        return
    elif data_name=="user_defined":
        return load_user_defined(user_defined_data_path)
    else:
        print("error in data name")
        sys.exit()



def load_user_defined(data_path):
    '''
    TODO:
    In order to load user_defined data, you must fill this function. If you use only dataset in this repository, you don't need to implement this function.

    Description:
        load graph data at 'data_path' and return node_features and edge_index.
        Manually create node_feature vectors and edge_index vectors. 

    Input:  
        data_path: path of the dataset location

    Output: 
        node_features: initial node features. Shape must be [ number_of_nodes, number_of_features_per_node]
        edge_index: list storing edges. Edges are represented by pair of node index. Shape must be [2, number_of_edges]
    '''
    return [0],[0]



# If this file is executed directly from terminal
if __name__ == "__main__":
    args=utils.parser()
    load_data(args.data, args.data_path)
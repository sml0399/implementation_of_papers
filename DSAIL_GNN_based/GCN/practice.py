import utils
import sys
import os
import numpy as np


def load_data(data_name="cora"):
    root_path=os.path.abspath(__file__)
    while(1):
        if(root_path[-24:]=="implementation_of_papers"):
            break
        else:
            root_path=os.path.dirname(root_path)
    root_path=root_path+"/"

    if data_name=="cora":
        data_path=root_path+"dataset/citeseer/"

        content=open(data_path+"citeseer.content","r")                                               # node feature file
        content_parsed1=content.read().split('\n')[:-1]                                          # divide by line
        content.close()
        content_parsed2=sorted([a.split('\t') for a in content_parsed1], key=lambda x:x[0]) # parse by tab and sort by node number
        node_numbers=[a[0] for a in content_parsed2]                                             # get set of node numbers
        node_type=[a[-1] for a in content_parsed2]                                               # list of type per node
        node_features=[a[1:-1] for a in content_parsed2]                                         # list of features per node

        edge_list=open(data_path+"citeseer.cites","r")                                               # edge (node pairs) file
        edge_list_parsed1=edge_list.read().split('\n')[:-1]                                      # divide by line
        edge_list.close()
        edge_list_parsed2=[a.split('\t') for a in edge_list_parsed1]                             # parse by tab 
        edge_index=[[a[0], a[1]] for a in edge_list_parsed2]                           # edge list holding (node number(int), node number(int)) pairs
        edge_index=np.array(edge_index).T.tolist()                                               # make transpose of list to make shape [2, number_of_edges]
        i=0;j=0
        while(i+j<len(edge_index[0])):                                                      # converting node numbers
            if (edge_index[0][i] not in node_numbers) or (edge_index[1][i] not in node_numbers):
                j+=1
                del edge_index[0][i]
                del edge_index[1][i]
                continue 
            edge_index[0][i]=node_numbers.index(edge_index[0][i])
            edge_index[1][i]=node_numbers.index(edge_index[1][i])
            i+=1
            
        return node_features, edge_index, node_type

load_data("cora")
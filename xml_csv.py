import xml.etree.ElementTree as ET
import Utils.file_utils as WR




#%%

#%%

#%%
"""
from torch_geometric.data import DataLoader
import Dataset as dataset

batch_size = 128
dataset = DataLoader(data, batch_size=batch_size, shuffle=True)
    
classes = ["piping_line", "signal_line"]
#"equipment_symbol", "instrument_symbol", "pipe_symbol", "text", 
print('dataset : ',len(dataset))

nums_train = int(len(dataset)*0.8)
nums_val = len(dataset) - nums_train
nums_test = int(nums_val*0.2)
nums_val = nums_val - nums_test

# dataset 분할
trainset, validationset, testset = torch.utils.data.dataset.random_split(dataset, [nums_train, nums_val, nums_test])
print('trainset : ',len(trainset))
print('validationset : ',len(validationset))
print('testset : ',len(testset))
"""

#%%
from torch_geometric.utils import to_networkx
G = to_networkx(data, to_undirected=True)

#%%
import networkx as nx
nx.draw(G, pos=nx.spring_layout(G), with_labels=True)


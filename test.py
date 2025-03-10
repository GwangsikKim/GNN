import torch
import xml.etree.ElementTree as ET
import Model.model as model
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
#%%

def list_split(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def convert_type_id(type_id):
    
    if type_id == "equipment_symbol":
        return 0
    elif type_id == "instrument_symbol":
        return 1
    elif type_id == "pipe_symbol":
        return 2
    elif type_id == "text":
        return 3
    elif type_id == "piping_line":
        return 4
    elif type_id == "signal_line":
        return 5
    else:
        return -1
            
def find_node_index(node_list, node_id):
    
    for index, node in enumerate(node_list):
        if node[0] == node_id:
            return index
    
    return -1        
    
def read_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    node_list = []
    
    symbol_objects = root.findall("symbol_object")
    
    for symbol_object in symbol_objects:
        type_id = convert_type_id(symbol_object.findtext("type"))
        assert type_id != -1
            
        symbol_id = symbol_object.findtext("id")
        
        node_list.append((symbol_id, [type_id]))
    
    line_objects = root.findall("line_object")
    
    for line_object in line_objects:
        type_id = convert_type_id(line_object.findtext("type"))
        assert type_id != -1
            
        line_id = line_object.findtext("id")
        
        node_list.append((line_id, [type_id]))

    edge_list = []
    
    connection_objects = root.findall("connection_object")    
    
    for connection_object in connection_objects:
        if connection_object.find("connection") is not None:
            from_id = connection_object.find("connection").attrib["From"]
            to_id = connection_object.find("connection").attrib["To"]
            
            from_index = find_node_index(node_list, from_id)
            assert from_index != -1
            
            to_index = find_node_index(node_list, to_id)
            assert to_index != -1
            
            edge1 = [from_index, to_index]
            edge2 = [to_index, from_index]
            
            edge_list.append(edge1)
            edge_list.append(edge2)
            
    node_feature_list = []
    
    for node in node_list:
        node_feature_list.append(node[1])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    x = torch.tensor(node_feature_list, dtype=torch.float)
    y = torch.tensor(node_feature_list, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index.t().contiguous())

    return data
#%%
file_path = "Dataset/15.xml"
data = read_xml(file_path)

####





#%%
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(42)

# Initialize model
model = model(data, data.edge_index)#, hidden_channels=16, num_classes=6)

# Use GPU
device = torch.device("cpu")#("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# Initialize Optimizer
learning_rate = 0.1
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)
# Define loss function (CrossEntropyLoss for Classification Problems with 
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(data.x, data.edge_index)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc

losses = []
for epoch in range(1000):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
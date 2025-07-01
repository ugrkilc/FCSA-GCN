import torch
import torch.nn as nn
from gfe_module import GFE_one, GFE_two
import torch.nn.functional as F
import graph.ntu_rgb_d as Graph

from temporal import tcn_unit_attention

def import_class(name):    
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)        
    return mod

class Model(nn.Module):
    def __init__(self, num_classes,residual, dropout, num_person, graph, num_nodes,input_channels):
        super().__init__()
        self.num_person = num_person
        self.indices_25_to_11 = [
        [0],          # 1 → 1
        [1, 20],      # 2,21 → 2
        [2, 3],       # 3,4 → 3
        [4, 5, 6],    # 5,6,7 → 4
        [7, 21, 22],  # 8,22,23 → 5
        [8, 9, 10],   # 9,10,11 → 6
        [11, 23, 24], # 12,24,25 → 7
        [12, 13],     # 13,14 → 8
        [14, 15],     # 15,16 → 9
        [16, 17],     # 17,18 → 10
        [18, 19]      # 19,20 → 11
        ]

        self.indices_11_to_6 = [
            [0, 1],  # 1,2 → 1
            [2],     # 3 → 2
            [3, 4],  # 4,5 → 3
            [5, 6],  # 6,7 → 4
            [7, 8],  # 8,9 → 5
            [9, 10]  # 10,11 → 6
        ]
        
        self.fc = nn.Conv2d(384, num_classes, kernel_size=1, padding=0)
        self.fc_other = nn.Conv2d(128, num_classes, kernel_size=1, padding=0)
        # Initialize graph
        self._initialize_graph(graph, num_nodes)      
      
        self.gfe_one = GFE_one(num_person, num_classes, dropout, residual, self.A_25.size(), input_channels)
        self.gfe_two = GFE_two(num_person, num_classes, dropout, residual, self.A_25.size(), self.A_11.size(), self.A_6.size())

        self.tsa_first= tcn_unit_attention(in_channels=64, out_channels=64,frames=300)
        self.tsa_other= tcn_unit_attention()
  

    def _initialize_graph(self, graph, num_nodes):
        if graph is None:
            raise ValueError("Graph cannot be None")
        else:
            Graph = import_class(graph)
            self.graph_instance_25 = Graph(25)
            self.graph_instance_11=Graph(11)
            self.graph_instance_6=Graph(6)
     
        
        A_25 = torch.tensor(self.graph_instance_25.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_25', A_25)

        A_11 = torch.tensor(self.graph_instance_11.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_11', A_11)

        A_6 = torch.tensor(self.graph_instance_6.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_6', A_6)

    
    def predict(self, x):
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0)//self.num_person, self.num_person, -1, 1, 1).mean(dim=1)
        x = self.fc_other(x)
        return x.view(x.size(0), -1)
      
    
    def forward(self, x):
         
        feature_map = self.gfe_one(x, self.A_25)    
        feature_map=self.tsa_first(feature_map)

        feature_map_11 = torch.zeros((feature_map.shape[0], feature_map.shape[1], feature_map.shape[2], len(self.indices_25_to_11)), device=feature_map.device)

        for i, idx_list in enumerate(self.indices_25_to_11):
            feature_map_11[:, :, :, i] = feature_map[:, :, :, idx_list].mean(dim=-1)

        feature_map_6 = torch.zeros((feature_map_11.shape[0], feature_map_11.shape[1], feature_map_11.shape[2], len(self.indices_11_to_6)), device=feature_map_11.device)

        for i, idx_list in enumerate(self.indices_11_to_6):
            feature_map_6[:, :, :, i] = feature_map_11[:, :, :, idx_list].mean(dim=-1)
       
        feature_map_25 = self.gfe_two(feature_map, self.A_25, mode=25)     
        feature_map_25 = self.tsa_other(feature_map_25)   
        output_25 = self.predict(feature_map_25)

        feature_map_11 = self.gfe_two(feature_map_11, self.A_11, mode=11) 
        feature_map_11 = self.tsa_other(feature_map_11)
        output_11 = self.predict(feature_map_11)

        feature_map_6 = self.gfe_two(feature_map_6, self.A_6, mode=6)    
        feature_map_6 = self.tsa_other(feature_map_6)  
        output_6 = self.predict(feature_map_6)

        combined = torch.cat((F.avg_pool2d(feature_map_25, feature_map_25.size()[2:]), 
                              F.avg_pool2d(feature_map_11, feature_map_11.size()[2:]), 
                              F.avg_pool2d(feature_map_6, feature_map_6.size()[2:])), dim=1)   
 
        combined = combined.view(combined.size(0)//self.num_person, self.num_person, -1, 1, 1).mean(dim=1)      
        combined = self.fc(combined)     
        combined= combined.view(combined.size(0), -1)  

        return output_6,output_11,output_25,combined


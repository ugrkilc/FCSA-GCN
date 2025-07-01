import os
import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import random
from fcsa_gcn import Model
from feeders.feeder import Feeder as Feeder


# CONFIG CLASS
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


#IMPORT CLASS
def import_class(name):    
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)      
    return mod

# SEED
def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/ntu60_xview.yaml')
p = parser.parse_args()

# load config
with open(p.config, 'r') as f:
    args = yaml.load(f, Loader=yaml.SafeLoader)

args = Config(args)

work_dir = args.work_dir
device = args.device
output_device = device[0] if type(device) is list else device
torch.backends.cudnn.enabled = False

# MODEL

num_classes = args.model_args.num_classes
model_name = args.model_args.model_name


if args.model_args.model_name is None:
    raise ValueError()
else:
    model_class=import_class(args.model_args.model_name)

model = model_class(args.model_args.num_classes,             
                            args.model_args.residual,
                            args.model_args.dropout,
                            args.model_args.num_person,                     
                            args.model_args.graph,      
                            args.model_args.num_nodes,   
                            args.model_args.input_channels        
                        )

model = model.to(args.device)

# LOAD WEIGHT
weights_path = work_dir + '/best_model.pt'
print('Load weights from: ', weights_path)

weights = torch.load(weights_path)
weights = OrderedDict([[k.split('module.')[-1], v.to(output_device)] for k, v in weights.items()])
try:
    model.load_state_dict(weights)
except:
    state = model.state_dict()
    diff = list(set(state.keys()).difference(set(weights.keys())))
    print('Can not find these weights:')
    for d in diff:
        print('# ' + d)
    state.update(weights)
    model.load_state_dict(state)

# LOAD DATA

data_loader = torch.utils.data.DataLoader(dataset=Feeder(data_path=args.test_feeder_args.data_path,
                                                                        label_path=args.test_feeder_args.label_path,                                                                     
                                                                        normalization=args.test_feeder_args.normalization,
                                                                        random_shift=args.test_feeder_args.random_shift,                                                              
                                                                        random_choose=args.test_feeder_args.random_choose,
                                                                        random_move=args.test_feeder_args.random_move,                                                                   
                                                                       ),
                                                        batch_size=args.test_feeder_args.batch_size,
                                                        shuffle=False,
                                                        num_workers=0)

score_list = []
acc_output_6= 0
acc_output_11 = 0
acc_output_25 = 0
acc_result = 0

confusion_matrix = np.zeros((num_classes, num_classes))
process = tqdm(data_loader, dynamic_ncols=True,leave=False, desc='Test')

with torch.no_grad():
    for _, (data, label, _) in enumerate(process):    
        data = data.to(output_device)
        label = label.to(output_device)
  
        output_6, output_11, output_25, result= model(data)

        _, predict_label = torch.max(result.data, 1)
        acc_result += (predict_label == label).sum().item()

        for l, p in zip(label.view(-1), predict_label.view(-1)):
            confusion_matrix[l.long(), p.long()] += 1

        score_list.append(result.data.cpu().numpy())


# save score
print('Save score: {}/test_score.pkl'.format(work_dir))
score = np.concatenate(score_list)
score_dict = dict(zip(data_loader.dataset.sample_name, score))
with open('{}/test_score.pkl'.format(work_dir), 'wb') as f:
    pickle.dump(score_dict, f)

print('Accuracy : {:.3f}'.format((100. * acc_result /len(data_loader.dataset))))


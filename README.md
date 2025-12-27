# FCSA-GCN

This repo is the official implementation for <mark>FCSA-GCN: Fine-to-Coarse Self-Attention Graph Convolutional Network for Skeleton-Based Action Recognition</mark>. 

# Architecture of FCSA-GCN
![image](https://github.com/ugrkilc/FCSA-GCN/blob/main/figures/agms-gcn.jpg)
# Data Preparation
### There are 3 datasets to download:
- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton


### Data Processing
##### Directory Structure
- Put downloaded data into the following directory structure:
  ```
  - Ntu60_Skeleton/          # from `nturgbd_skeletons_s001_to_s017.zip`
  - Ntu120_Skeleton/         # from `nturgbd_skeletons_s018_to_s032.zip`
  
##### Generating Data
- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:
  ```
  python ntu60_gen_joint_data.py
  python ntu120_gen_joint_data.py
  python gen_bone_data.py
  python ntu_gen_motion_data.py
  python merge_joint_bone_data.py
  python merge_joint_joint_motion.py
  python merge_bone_bone_motion.py
  python merge_joint_motion_bone_motion.py

# Training & Testing
### Training
- You can modify the training and model settings through the configuration files found in the config/ directory. Each dataset has its own dedicated config file, which can be edited to adjust the parameters as needed.
- Load the config file and train the model.
- Cross-view: Train the model with NTU-RGB+D60 coordinate (joint, bone) data. 
  ```
  python main.py --config config/ntu60_xview.yaml
- Cross-subject: Train the model with NTU-RGB+D60 coordinate (joint, bone) data. 
  ```
  python main.py --config config/ntu60_xsub.yaml  

# Citation
Please cite the following paper if you use this repository in your research.

    @article{kilic2025fine,
      title={Fine-to-coarse self-attention graph convolutional network for skeleton-based action recognition},
      author={Kilic, Ugur and Karadag, Ozge Oztimur and Ozyer, Gulsah Tumuklu},
      journal={Applied Soft Computing},
      pages={114268},
      year={2025},
      publisher={Elsevier}
      }
    
# Contact
For any questions, feel free to contact: `ugur.kilic@erzurum.edu.tr`

# Acknowledge
This repo is based on ST-GCN and STA-GCN, thanks to their excellent work.






# Point Transform Registration
6-DoF Pose Estimation of Uncooperative Space Object Using Deep Learning with Point Cloud

## Prerequisites:
PyTorch 1.8.2  
open3d  
h5py  
numpy  
tqdm  
TensorboardX  

## Training
python main.py

## Testing
python main.py --test  
Refined by ICP with batch size = 1：    
python main.py --test --icp  

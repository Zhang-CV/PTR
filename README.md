# Point Transform Registration
Article: 6-DoF Pose Estimation of Uncooperative Space Object Using Deep Learning with Point Cloud

## Prerequisites:
PyTorch 1.8.2  
open3d  
h5py  
numpy  
tqdm  
TensorboardX  

## Training
unrar x ./data/train_data.rar
unrar x ./data/eval_data.rar
unrar x ./data/test_data.rar
python main.py

## Testing
python main.py --test  
Refined by ICP with batch size = 1：    
python main.py --test --icp  

## Citation
Please cite this paper if you want to use it in your work,  
@INPROCEEDINGS{9843444,  author={Zhang, Shaodong and Hu, Weiduo and Guo, Wulong},     
booktitle={2022 IEEE Aerospace Conference (AERO)},   
title={6-DoF Pose Estimation of Uncooperative Space Object Using Deep Learning with Point Cloud},   
year={2022},  
volume={},  
number={},  
pages={1-7},  
doi={10.1109/AERO53065.2022.9843444}}

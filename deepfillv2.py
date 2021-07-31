import argparse
import os



# ----------------------------------------
#        Initialize the parameters
# ----------------------------------------
parser = argparse.ArgumentParser()
# General parameters
parser.add_argument('--results_path', type = str, default = './results', help = 'testing samples path that is a folder')
parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
parser.add_argument('--gpu_ids', type = str, default = "1", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
# Training parameters
parser.add_argument('--epoch', type = int, default = 40, help = 'number of epochs of training')
parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
# Network parameters
parser.add_argument('--in_channels', type = int, default = 4, help = 'input RGB image + 1 channel mask')
parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
parser.add_argument('--latent_channels', type = int, default = 48, help = 'latent channels')
parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
parser.add_argument('--activation', type = str, default = 'elu', help = 'the activation type')
parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
# Dataset parameters
parser.add_argument('--baseroot', type = str, default = '../../inpainting/dataset/Places/img_set')
parser.add_argument('--baseroot_mask', type = str, default = '../../inpainting/dataset/Places/img_set')
opt = parser.parse_args()
    

#     python3 test.py \
# --baseroot './test_data/' \
# --baseroot_mask './test_data_mask/' \
# --results_path './results' \
# --gan_type 'WGAN' \
# --gpu_ids '1' \
# --epoch 40 \
# --batch_size 1 \
# --num_workers 8 \
# --pad_type 'zero' \
# --activation 'elu' \
# --norm 'none' \
    
# ----------------------------------------
#       Choose CUDA visible devices
# ----------------------------------------

# Enter main function
import deepfill_tester

def deepfill(img,mask):
    deepfill_tester.WGAN_tester(opt, img, mask)
        
    
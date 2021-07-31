import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import torchvision.transforms.functional as TF

import network
import test_dataset
import utils
from pprint import pprint

def deepfill(opt,inPilImg,inPilMask):
    
    # Save the model if pre_train == True
    def load_model_generator(net, epoch):
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, 4)
        model_name = os.path.join('pretrained_model', model_name)
        pretrained_dict = torch.load(model_name)
        generator.load_state_dict(pretrained_dict)

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    results_path = "results"
    # configurations
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Build networks
    generator = utils.create_generator(opt).eval()
    print('-------------------------Loading Pretrained Model-------------------------')
    load_model_generator(generator, opt.epoch)
    print('-------------------------Pretrained Model Loaded-------------------------')

    # To device
    generator = generator.cuda()

    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    # Testing loop

    batch_idx = 0
    # img = cv2.imread("/content/DeepFillv2_Pytorch/test_data/1.png")
    img = pil2cv(inPilImg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

    # mask = cv2.imread("/content/DeepFillv2_Pytorch/test_data_mask/1.png")[:, :, 0]
    mask = pil2cv(inPilMask)[:, :, 0]
    mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
    img = torch.unsqueeze(img,0)
    mask = torch.unsqueeze(mask,0)

    img = img.cuda()
    mask = mask.cuda()
    pprint(img)

    pprint(img.size())
    pprint(mask.size())
    # Generator output
    with torch.no_grad():
        first_out, second_out = generator(img, mask)

    # forward propagation
    first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
    second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

    masked_img = img * (1 - mask) + mask
    mask = torch.cat((mask, mask, mask), 1)

    # Recover normalization: * 255 because last layer is sigmoid activated
    img = img * 255
    # Process img_copy and do not destroy the data of img
    img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
    img_copy = np.clip(img_copy, 0, 255)
    img_copy = img_copy.astype(np.uint8)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

    return cv2pil(img_copy)

    # img_list = [second_out_wholeimg]
    # name_list = ['second_out']
    # utils.save_sample_png(sample_folder = results_path, sample_name = '%d' % (batch_idx + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
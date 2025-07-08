import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
#from DC_TEST import backward
import cv2
import torch
import cv2
import os
import numpy as np


def backward(speckles_data, mask_data):
    speckles_data =  speckles_data[0,0,:,:]
    speckles_data =  speckles_data[76:1124,436:1484]
    print("speckles_data",speckles_data.shape)
    speckles_data_fft = np.fft.fft2(speckles_data)
    mask_data_fft = np.fft.fft2(mask_data)

    k = 1000 #1000
    C = (np.mean(mask_data[:]) * k) ** 2
    aa = speckles_data_fft * np.conj(mask_data_fft)
    bb = (abs(mask_data_fft ** 2) + C).astype(dtype=np.float64)
    rec_img_fft = np.divide(aa, bb)

    rec_img = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(rec_img_fft))))


    rec_img = rec_img[380:636, 386:642]
   
    
    print("rec",rec_img.shape)
    rec_img = rec_img / np.max(rec_img)
    return rec_img



#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
sampling_mode = opt["sde"]["sampling_mode"]

lpips_fn = lpips.LPIPS(net='alex').to(device)

scale = opt['degradation']['scale']

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = []
    test_times = []

    mask = cv2.imread('/home/wwb/LJY/LJY/image-restoration-sde-main/25_5/psf.png')
    print("mask",mask.shape)
    mask = mask[:,:,0]
    mask = mask[76:1124,436:1484]  
    #mask = mask[0:1200,360:1560,0]   
   
    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ, GT = test_data["LQ"], test_data["GT"]
        #########################################################

        LQ = backward(LQ, mask)
        print("LQ:.............",LQ.shape)
        #LQ = LQ[128:384, 128:384]#512×512->256×256
        LQ = torch.tensor(LQ, dtype=torch.float32)
        LQ = LQ.cpu().numpy()
        
        LQ = np.expand_dims(LQ, axis=0) 
        LQ = np.expand_dims(LQ, axis=1)
        LQ = np.repeat(LQ, 3, axis=1)
        LQ = torch.tensor(LQ, dtype=torch.float32)
        print("LQ:............",LQ.shape)
        print("GT:............",GT.shape)

        ########################################################
        noisy_state = sde.noise_state(LQ)
        model.feed_data(noisy_state, LQ, GT)

        tic = time.time()
        #model.test(sde, mode=sampling_mode, save_states=False)
        data= model.test(sde, mode='sde', save_states=True)
        #data= model.test(sde, mode='test',save_states=True, ori=ori_data, model=model)
        toc = time.time()
        test_times.append(toc - tic)
        
        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        
        output = util.tensor2img(SR_img.squeeze())  # uint8
        #output=output[128:384,128:384]
        LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8
        
        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
        util.save_img(output, save_img_path)

        # remove it if you only want to save output images
        LQ_img_path = os.path.join(dataset_dir, img_name + "_LQ.png")
        GT_img_path = os.path.join(dataset_dir, img_name + "_HQ.png")
        util.save_img(LQ_, LQ_img_path)
        util.save_img(GT_, GT_img_path)
print(f"average test time: {np.mean(test_times):.4f}")


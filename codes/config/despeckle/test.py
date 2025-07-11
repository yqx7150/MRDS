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
import options as optionpython 
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
#from DC_TEST import backward
import cv2
import torch


def backward(speckles_data, mask_data):


    speckles_data =  speckles_data[0,0,:,:]
    speckles_data =  speckles_data[16:1040, 366:1390]
    print("speckle",speckles_data.shape)
    speckles_data_fft = np.fft.fft2(speckles_data)
    mask_data_fft = np.fft.fft2(mask_data)

    k = 300
    C = (np.mean(mask_data[:]) * k) ** 2
    aa = speckles_data_fft * np.conj(mask_data_fft)
    bb = (abs(mask_data_fft ** 2) + C).astype(dtype=np.float64)
    rec_img_fft = np.divide(aa, bb)

    rec_img = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(rec_img_fft))))
    
    rec_img = rec_img[384:640, 394:650]

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
    test_results["psnr_tc"] = []
    test_results["ssim_tc"] = []
    test_times = []


    mask = cv2.imread('/home/wwb/LJY/LJY/image-restoration-sde-main/shice/shece/psf.png')  # (2560, 2160, 3)
    #mask = mask[824:1336, 1024:1536, 0]  # 2560×2160×3->512×512
    print(mask.shape)
    mask = mask[88:1112,448:1472,0] 
   
    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ, GT = test_data["LQ"], test_data["GT"]#LQ: torch.Size([1, 3, 2815, 2415])
        #########################################################
        print("LQ:............",LQ.shape)

        LQ = backward(LQ, mask)
        print("before LQ:............",LQ.shape)

        LQ = torch.tensor(LQ, dtype=torch.float32)
        LQ = LQ.cpu().numpy()
        LQ = np.expand_dims(LQ, axis=0) 
        LQ = np.expand_dims(LQ, axis=1)
        LQ = np.repeat(LQ, 3, axis=1)
        LQ = torch.tensor(LQ, dtype=torch.float32)
        print("LQ:............",LQ.shape)
        print("GT:............",GT.shape)



        #########################################################
        noisy_state = sde.noise_state(LQ)
        model.feed_data(noisy_state, LQ, GT)
        print("**********************",type(noisy_state))
        output_dir = "./noisy_state/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "noisy_state.png")
        torch.save(noisy_state, output_file)
        
        

        tic = time.time()
        #model.test(sde, mode=sampling_mode, save_states=False)
        data= model.test(sde, mode='sde', save_states=True)
        #data= model.test(sde, mode='test',save_states=True, ori=ori_data, model=model)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        output = util.tensor2img(SR_img.squeeze())  # uint8
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

        if need_GT:
            gt_img = GT_ / 255.0
            sr_img = output / 255.0
            tc_img = LQ_ /255.0
            crop_border = opt["crop_border"] if opt["crop_border"] else scale
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
                cropped_tc_img = tc_img
            else:
                cropped_sr_img = sr_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
                cropped_gt_img = gt_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
                cropped_tc_img = tc_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]



            cropped_sr_img = cropped_sr_img[:, :, 0]
            cropped_gt_img = cropped_gt_img[:, :, 0]
            cropped_tc_img = cropped_tc_img[:, :, 0]
            
            # print("cropped_sr_img:.............",cropped_sr_img.shape)
            # print("cropped_gt_img:.............",cropped_gt_img.shape)
            psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)
            ssim = util.calculate_ssim(sr_img * 255, gt_img * 255)
            psnr_tc = util.calculate_psnr(tc_img * 255, gt_img * 255)
            ssim_tc = util.calculate_ssim(tc_img * 255, gt_img * 255)
            lp_score = lpips_fn(
                GT.to(device) * 2 - 1, SR_img.to(device) * 2 - 1).squeeze().item()

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["lpips"].append(lp_score)
            test_results["psnr_tc"].append(psnr_tc)
            test_results["ssim_tc"].append(ssim_tc)

            if len(gt_img.shape) == 3:
                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    if crop_border == 0:
                        cropped_sr_img_y = sr_img_y
                        cropped_gt_img_y = gt_img_y
                    else:
                        cropped_sr_img_y = sr_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                        cropped_gt_img_y = gt_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                    psnr_y = util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    ssim_y = util.calculate_ssim(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )

                    test_results["psnr_y"].append(psnr_y)
                    test_results["ssim_y"].append(ssim_y)

                    logger.info(
                        "img{:3d}:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                            i, img_name, psnr, ssim, lp_score, psnr_y, ssim_y
                        )
                    )
            else:
                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                        img_name, psnr, ssim
                    )
                )

                test_results["psnr_y"].append(psnr)
                test_results["ssim_y"].append(ssim)
        else:
            logger.info(img_name)


    ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    ave_psnr_tc = sum(test_results["psnr_tc"]) / len(test_results["psnr_tc"])
    ave_ssim_tc = sum(test_results["ssim_tc"]) / len(test_results["ssim_tc"])
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim
        )
    )
    if test_results["psnr_y"] and test_results["ssim_y"]:
        ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
        ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
        logger.info(
            "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                ave_psnr_y, ave_ssim_y
            )
        )

    logger.info(
            "----average LPIPS\t: {:.6f}\n".format(ave_lpips)
        )

    print(f"average test time: {np.mean(test_times):.4f}")
    print("psnr_tc",ave_psnr_tc)
    print("ssim_tc",ave_ssim_tc)
    

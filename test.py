import argparse, time, os, random
import scipy.misc as misc
import numpy as np
from tqdm import tqdm

from collections import OrderedDict

import torch

import options.options as option
from utils import util
from models.SRModel import SRModel
from data import create_dataloader
from data import create_dataset

BENCHMARK = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']

def main():
    # os.environ['CUDA_VISIBLE_DEVICES']='1' # You can specify your GPU device here. I failed to perform it by `torch.cuda.set_device()`.
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    # Initialization
    scale = opt['scale']
    dataroot_HR = opt['datasets']['val']['dataroot_HR']
    network_opt = opt['networks']['G']
    if network_opt['which_model'] == "feedback":
        model_name = "%s_f%dt%du%ds%d"%(network_opt['which_model'], network_opt['num_features'], network_opt['num_steps'], network_opt['num_units'], network_opt['num_stages'])
    else:
        model_name = network_opt['which_model']

    bm_list = [dataroot_HR.find(bm)>=0 for bm in BENCHMARK]
    bm_idx = bm_list.index(True)
    bm_name = BENCHMARK[bm_idx]

    # create test dataloader
    dataset_opt = opt['datasets']['val']
    if dataset_opt is None:
        raise ValueError("test dataset_opt is None!")
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)

    if test_loader is None:
        raise ValueError("The training data does not exist")

    if opt['mode'] == 'sr':
        solver = SRModel(opt)
    else:
        raise NotImplementedError

    # load model
    model_pth = os.path.join(solver.model_pth, 'epoch', 'best_checkpoint.pth')
    # model_pth = solver.model_pth
    if model_pth is None:
        raise ValueError("model_pth' is required.")
    print('[Loading model from %s...]'%model_pth)
    model_dict = torch.load(model_pth)
    if 'state_dict' in model_dict.keys():
        solver.model.load_state_dict(model_dict['state_dict'])
    else:
        new_model_dict = OrderedDict()
        for key, value in model_dict.items():
            new_key = 'module.'+key
            new_model_dict[new_key] = value
        model_dict = new_model_dict

        solver.model.load_state_dict(model_dict)
    print('=> Done.')
    print('[Start Testing]')

    start_time = time.time()

    # we only forward one epoch at test stage, so no need to load epoch, best_prec, results from .pth file
    # we only save images and .pth for evaluation. Calculating statistics are handled by matlab.
    # do crop for efficiency
    test_bar = tqdm(test_loader)
    sr_list = []
    path_list = []
    psnr_list = []

    total_psnr = 0.
    for iter, batch in enumerate(test_bar):
        solver.feed_data(batch)
        solver.test(opt['chop'])
        visuals = solver.get_current_visual()   # fetch current iteration results as cpu tensor

        sr_img = np.transpose(util.quantize(visuals['SR'], opt['rgb_range']).numpy(), (1, 2, 0)).astype(np.uint8)
        gt_img = np.transpose(util.quantize(visuals['HR'], opt['rgb_range']).numpy(), (1, 2, 0)).astype(np.uint8)

        # calculate PSNR
        crop_size = opt['scale']
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]

        psnr = util.calc_psnr(cropped_sr_img, cropped_gt_img)
        psnr_list.append(psnr)
        total_psnr += psnr

        sr_list.append(sr_img)
        path_list.append(os.path.splitext(os.path.basename(batch['HR_path'][0]))[0])

    print("=====================================")
    save_txt_path = os.path.join(solver.model_pth, '%s_x%d.txt'%(bm_name, scale))
    line_list = []
    line = "Method : %s\nTest set : %s\nScale : %d "%(model_name, bm_name, scale)
    line_list.append(line+'\n')
    print(line)
    for value, img_name in zip(psnr_list, path_list):
        line = "Image name : %s PSNR = %.2f "%(img_name, value)
        line_list.append(line + '\n')
        print(line)
    line = "Average PSNR is %.2f"%(total_psnr/len(test_bar))
    line_list.append(line)
    print(line)

    # save results
    f = open(save_txt_path, 'w')
    f.writelines(line_list)
    f.close()

    save_img_path = os.path.join('./eval/SR/BI', model_name, bm_name, "x%d"%scale)
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    for img, img_name in zip(sr_list, path_list):
        misc.imsave(os.path.join(save_img_path, img_name+'_%s_x%d.png'%(model_name, scale)), img)

    test_bar.close()
    time_elapse = start_time - time.time()

if __name__ == '__main__':
    main()
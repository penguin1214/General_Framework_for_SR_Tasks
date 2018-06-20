import argparse, time, os
import random

import torch
import torchvision.utils as thutil
import pandas as pd
from tqdm import tqdm

import options.options as option
from utils import util
from models.SRModel import SRModel
from data import create_dataloader
from data import create_dataset

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=True)

if opt['train']['resume'] is False:
    util.mkdir_and_rename(opt['path']['exp_root'])  # rename old experiments if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root' and \
        not key == 'pretrain_G' and not key == 'pretrain_D'))
    option.save(opt)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
else:
    opt = option.dict_to_nonedict(opt)
    if opt['train']['resume_path'] is None:
        raise ValueError("The 'resume_path' does not declarate")

if opt['exec_debug']:
    NUM_EPOCH = 50
    opt['datasets']['train']['dataroot_HR'] = '/pathto/data/DIV2K/DIV2K_train_HR_debug'
    opt['datasets']['train']['dataroot_LR'] = 'pathto/data/DIV2K/DIV2K_train_LR_debug'
else:
    NUM_EPOCH = int(opt['num_epochs'])

def main():
    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('Number of train images in [%s]: %d' % (dataset_opt['name'], len(train_set)))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [%s]: %d' % (dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    if train_loader is None:
        raise ValueError("The training data does not exist")

    # TODO: design an exp that can obtain the location of the biggest error
    solver = SRModel(opt)
    solver.summary(train_set[0]['LR'].size())

    print('[Start Training]')

    start_time = time.time()

    start_epoch = 1
    if opt['train']['resume']:
        checkpoint_path = os.path.join(solver.checkpoint_dir,'checkpoint.pth')
        print('[Loading checkpoint from %s...]'%checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        solver.model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Because the last state had been saved
        solver.optimizer.load_state_dict(checkpoint['optimizer'])
        solver.best_prec = checkpoint['best_prec']
        solver.results = checkpoint['results']
        print('=> Done.')

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        # Initialization
        solver.training_loss = 0.0
        if opt['mode'] == 'sr':
            training_results = {'batch_size': 0, 'training_loss': 0.0}
        else:
            pass    # TODO
        train_bar = tqdm(train_loader)

        # Train model
        for iter, batch in enumerate(train_bar):
            solver.feed_data(batch)
            iter_loss = solver.train_step()
            batch_size = batch['LR'].size(0)
            training_results['batch_size'] += batch_size

            if opt['mode'] == 'sr':
                training_results['training_loss'] += iter_loss * batch_size
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, NUM_EPOCH, iter_loss))
            else:
                pass    # TODO

        train_bar.close()
        time_elapse = time.time() - start_time
        start_time = time.time()

        # validate
        val_results = {'batch_size': 0, 'val_loss': 0.0, 'psnr': 0.0, 'ssim': 0.0}

        if epoch % solver.val_step == 0 and epoch != 0:

            print('[Validating...]')
            start_time = time.time()
            solver.val_loss = 0.0

            index = 0
            visuals_list = []

            for iter, batch in enumerate(val_loader):
                solver.feed_data(batch)
                iter_loss = solver.test()
                batch_size = batch['LR'].size(0)
                val_results['batch_size'] += batch_size

                visuals = solver.get_current_visual()

                sr_img = util.tensor2img_np(visuals['SR'])  # uint8
                gt_img = util.tensor2img_np(visuals['HR'])  # uint8

                # calculate PSNR
                crop_size = opt['scale'] + 2
                cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]

                val_results['val_loss'] += iter_loss * batch_size
                val_results['psnr'] += util.psnr(cropped_sr_img, cropped_gt_img)
                # val_results['psnr']  += util.psnr(sr_img, gt_img))
                # TODO: ssim multichannel ? We should read some recent papers to confirm it
                val_results['ssim'] += util.ssim(cropped_sr_img, cropped_gt_img, multichannel=True)
                # val_results['ssim'] += util.ssim(tensor2numpy4eval(solver.SR.squeeze(0)), tensor2numpy4eval(solver.var_HR.squeeze(0)), multichannel=True)

                if opt['mode'] == 'srgan':
                    pass    # TODO

                visuals_list.extend([util.display_transform()(visuals['HR'].squeeze(0)),
                                     util.display_transform()(visuals['LR'].squeeze(0)),
                                     util.display_transform()(visuals['SR'].squeeze(0))])

            avg_psnr = val_results['psnr']/val_results['batch_size']
            avg_ssim = val_results['ssim']/val_results['batch_size']
            print('Valid Loss: %.4f | Avg. PSNR: %.4f | Avg. SSIM: %.4f'%(iter_loss, avg_psnr, avg_ssim))
            val_images = torch.stack(visuals_list)
            val_images = torch.chunk(val_images, val_images.size(0)//15)

            vis_index = 1
            for image in val_images:
                image = thutil.make_grid(image, nrow=3, padding=5)
                thutil.save_image(image, os.path.join(solver.vis_dir, 'epoch_%d_index_%d.png'%(epoch, vis_index)), padding=5)
                vis_index += 1

            time_elapse = start_time - time.time()

            #if epoch%solver.log_step == 0 and epoch != 0:
            # tensorboard visualization
            solver.training_loss = training_results['training_loss'] / training_results['batch_size']
            solver.val_loss = val_results['val_loss'] / val_results['batch_size']

            solver.tf_log(epoch)

            # statistics
            if opt['mode'] == 'sr':
                solver.results['training_loss'].append(solver.training_loss.data.cpu().numpy()[0])
                solver.results['val_loss'].append(solver.val_loss.data.cpu().numpy()[0])
                solver.results['psnr'].append(avg_psnr)
                solver.results['ssim'].append(avg_ssim)
            else:
                pass    # TODO

            is_best = False
            if solver.best_prec < solver.results['psnr'][-1]:
                solver.best_prec = solver.results['psnr'][-1]
                is_best = True

            solver.save(epoch, is_best)

        # update lr
        solver.update_learning_rate()

    data_frame = pd.DataFrame(
        data={'training_loss': solver.results['training_loss']
            , 'val_loss': solver.results['val_loss']
            , 'psnr': solver.results['psnr']
            , 'ssim': solver.results['ssim']
              },
        index=range(1, NUM_EPOCH)
    )
    data_frame.to_csv(os.path.join(solver.results_dir, 'train_results.csv'),
                      index_label='Epoch')


if __name__ == '__main__':
    main()
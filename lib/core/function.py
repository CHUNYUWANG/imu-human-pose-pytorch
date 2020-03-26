# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import h5py
import numpy as np
import pandas as pd

import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images, save_batch_fusion_heatmaps, save_debug_heatmaps

logger = logging.getLogger(__name__)


# def routing(raw_features, aggre_features, is_aggre, meta):
#     if not is_aggre:
#         return raw_features
#
#     output = []
#     for r, a, m in zip(raw_features, aggre_features, meta):
#         view = torch.zeros_like(a)
#         batch_size = a.size(0)
#         for i in range(batch_size):
#             s = m['source'][i]
#             view[i] = a[i] if s != 'mpii' else r[i]  # make it compatible with dataset rather than only h36m
#         output.append(view)
#     return output
def merge_first_two_dims(tensor):
    dim0 = tensor.shape[0]
    dim1 = tensor.shape[1]
    left = tensor.shape[2:]
    return tensor.view(dim0*dim1, *left)


def train(config, data, model, criterion, optim, epoch, output_dir,
          writer_dict, **kwargs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    model.train()

    end = time.time()
    for i, (input_, target_, weight_, meta_) in enumerate(data):
        data_time.update(time.time() - end)

        output, extra = model(input_, **meta_)

        input = merge_first_two_dims(input_)
        target = merge_first_two_dims(target_)
        weight = merge_first_two_dims(weight_)
        meta = dict()
        for kk in meta_:
            meta[kk] = merge_first_two_dims(meta_[kk])

        target_cuda = target.cuda()
        weight_cuda = weight.cuda()
        loss = 0
        b_imu_fuse = extra['imu_fuse']
        if b_imu_fuse:
            loss += 0.5 * criterion(extra['origin_hms'], target_cuda, weight_cuda)
            target_mask = torch.as_tensor(target_cuda > 0.001, dtype=torch.float32).cuda()
            imu_masked = heatmaps * target_mask
            target_imu_joint = target_cuda * extra['joint_channel_mask'][0]
            loss += 0.5 * criterion(imu_masked, target_imu_joint, weight_cuda)
        else:
            loss += criterion(extra['origin_hms'], target_cuda, weight_cuda)

        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.update(loss.item(), len(input) * input[0].size(0))

        _, acc, cnt, pre = accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy())
        avg_acc.update(acc, cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                  'Memory {memory:.1f}'.format(
                      epoch, i, len(data), batch_time=batch_time,
                      speed=input.shape[0] / batch_time.val,
                      data_time=data_time, loss=losses, acc=avg_acc, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', avg_acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            # for k in range(len(input)):
            view_name = 'view_{}'.format(0)
            prefix = '{}_{}_{:08}'.format(
                os.path.join(output_dir, 'train'), view_name, i)
            meta_for_debug_imgs = dict()
            meta_for_debug_imgs['joints_vis'] = meta['joints_vis']
            meta_for_debug_imgs['joints_2d_transformed'] = meta['joints_2d_transformed']
            save_debug_images(config, input, meta_for_debug_imgs, target,
                              pre * 4, extra['origin_hms'], prefix)
            if extra is not None and 'fused_hms' in extra:
                fuse_hm = extra['fused_hms']
                prefix = '{}_{}_{:08}'.format(
                    os.path.join(output_dir, 'fused_hms'), view_name, i)
                save_debug_heatmaps(config, input, meta_for_debug_imgs, target,
                                    pre * 4, fuse_hm, prefix)


def validate(config, loader, dataset, model, criterion, output_dir,
             writer_dict=None, **kwargs):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    nview = len(config.SELECTED_VIEWS)
    nsamples = len(dataset) * nview
    njoints = config.NETWORK.NUM_JOINTS
    height = int(config.NETWORK.HEATMAP_SIZE[0])
    width = int(config.NETWORK.HEATMAP_SIZE[1])
    all_preds = np.zeros((nsamples, njoints, 3), dtype=np.float32)
    all_heatmaps = np.zeros(
        (nsamples, njoints, height, width), dtype=np.float32)

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input_, target_, weight_, meta_) in enumerate(loader):
            batch = input_.shape[0]
            output, extra = model(input_, **meta_)

            input = merge_first_two_dims(input_)
            target = merge_first_two_dims(target_)
            weight = merge_first_two_dims(weight_)
            meta = dict()
            for kk in meta_:
                meta[kk] = merge_first_two_dims(meta_[kk])

            target_cuda = target.cuda()
            weight_cuda = weight.cuda()
            loss = criterion(output, target_cuda, weight_cuda)

            nimgs = input.size()[0]
            losses.update(loss.item(), nimgs)

            _, acc, cnt, pre = accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy(), thr=0.083)
            avg_acc.update(acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            pred, maxval = get_final_preds(config,
                                           output.clone().cpu().numpy(),
                                           meta['center'],
                                           meta['scale'])

            pred = pred[:, :, 0:2]
            pred = np.concatenate((pred, maxval), axis=2)

            all_preds[idx:idx + nimgs] = pred
            all_heatmaps[idx:idx + nimgs] = output.cpu().numpy()
            # image_only_heatmaps[idx:idx + nimgs] = img_detected.cpu().numpy()
            idx += nimgs

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time,
                          loss=losses, acc=avg_acc)
                logger.info(msg)

                view_name = 'view_{}'.format(0)
                prefix = '{}_{}_{:08}'.format(
                    os.path.join(output_dir, 'validation'), view_name, i)
                meta_for_debug_imgs = dict()
                meta_for_debug_imgs['joints_vis'] = meta['joints_vis']
                meta_for_debug_imgs['joints_2d_transformed'] = meta['joints_2d_transformed']
                save_debug_images(config, input, meta_for_debug_imgs, target,
                                  pre * 4, extra['origin_hms'], prefix)
                if 'fused_hms' in extra:
                    fused_hms = extra['fused_hms']
                    prefix = '{}_{}_{:08}'.format(
                        os.path.join(output_dir, 'fused_hms'), view_name, i)
                    save_debug_heatmaps(config, input, meta_for_debug_imgs, target,
                                      pre * 4, fused_hms, prefix)

        detection_thresholds = [0.075, 0.05, 0.025, 0.0125]  # 150,100,50,25 mm
        perf_indicators = []
        cur_time = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
        for thresh in detection_thresholds:
            name_value, perf_indicator, per_grouping_detected = dataset.evaluate(all_preds, threshold=thresh)
            perf_indicators.append(perf_indicator)
            names = name_value.keys()
            values = name_value.values()
            num_values = len(name_value)
            _, full_arch_name = get_model_name(config)
            logger.info('Detection Threshold set to {} aka {}mm'.format(thresh, thresh * 2000.0))
            logger.info('| Arch   ' +
                        '  '.join(['| {: <5}'.format(name) for name in names]) + ' |')
            logger.info('|--------' * (num_values + 1) + '|')
            logger.info('| ' + '------ ' +
                        ' '.join(['| {:.4f}'.format(value) for value in values]) +
                        ' |')
            logger.info('| ' + full_arch_name)
            logger.info('Overall Perf on threshold {} is {}\n'.format(thresh, perf_indicator))
            logger.info('\n')
            if per_grouping_detected is not None:
                df = pd.DataFrame(per_grouping_detected)
                save_path = os.path.join(output_dir, 'grouping_detec_rate_{}_{}.csv'.format(thresh, cur_time))
                df.to_csv(save_path)

        # save heatmaps and joint locations
        u2a = dataset.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = np.array(list(a2u.values()))

        save_file = config.TEST.HEATMAP_LOCATION_FILE
        file_name = os.path.join(output_dir, save_file)
        file = h5py.File(file_name, 'w')
        file['heatmaps'] = all_heatmaps[:, u, :, :]
        file['locations'] = all_preds[:, u, :]
        file['joint_names_order'] = a
        file.close()

    return perf_indicators[3]  # 25mm as indicator


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

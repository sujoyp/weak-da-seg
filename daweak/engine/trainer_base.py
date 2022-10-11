###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2021
###########################################################################

import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from PIL import Image

from daweak.dataset import get_dataset
from daweak.model import get_segmentation_model, get_discriminator_model
import daweak.util as util

PER_TH = 0.1


def get_prediction(preds):
    scale = preds.shape[-1]*preds.shape[-2]
    t = preds + torch.log(torch.tensor(1/scale).cuda())
    t = torch.logsumexp(t.view(preds.shape[0], preds.shape[1], -1), -1)
    preds = torch.clamp(torch.sigmoid(t), 1e-7, 1-1e-7)
    return preds


def get_weak_labels(labels, num_class, counts):
    labels = labels.cpu().numpy()
    weak_labels_onehot = [
        [np.sum(lbl == i) > PER_TH * counts[i] for i in range(num_class)] for lbl in labels
    ]
    weak_labels_onehot = torch.tensor(np.array(weak_labels_onehot).astype('int'))
    return weak_labels_onehot.float()


def load_threshold(dataset_name, num_class):
    tmp = np.load('pixel_counts/%s_pixel_counts.npy' % dataset_name)
    m = np.zeros(num_class)
    for i in range(10):
        counts = [tmp[tmp[:, i] > PER_TH*m[i], i] for i in range(num_class)]
        m = [np.mean(c) for c in counts]
    return m


palette = [
    128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170,
    30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142,
    0, 0, 70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32
]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


class Trainer:

    def __init__(self, args):
        self.args = args
        w, h = map(int, args.input_size_source.split(','))
        input_size_source = (w, h)

        w, h = map(int, args.input_size_target.split(','))
        input_size_target = (w, h)

        if not os.path.exists(args.snapshot_dir) and not args.val_only:
            os.makedirs(args.snapshot_dir)

        self.logger_fid = None
        if not args.val_only:
            self.logger_fid = open(args.snapshot_dir + "/log.txt", "w")

        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        self.source_th = load_threshold(args.dataset_source, args.num_classes)
        self.target_th = load_threshold(args.dataset_target, args.num_classes)

        # dataset
        data_kwargs = {'data_root': args.data_root, 'max_iters': args.num_steps * args.batch_size}
        if not args.val_only:
            trainset_source = get_dataset(
                args.dataset_source, path=args.data_path_source, split=args.source_split,
                mode='train', size=input_size_source, use_pixeladapt=args.use_pixeladapt,
                **data_kwargs
            )
            trainset_target = get_dataset(
                args.dataset_target, path=args.data_path_target, split=args.target_split,
                mode='val', size=input_size_target, use_points=args.use_pointloss, **data_kwargs
            )
        testset = get_dataset(
            args.dataset_target, path=args.data_path_target, split=args.test_split,
            mode='val', size=input_size_target, data_root=args.data_root
        )

        # dataloader
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True} \
            if args.cuda else {}
        if not args.val_only:
            self.trainloader_source = data.DataLoader(trainset_source, batch_size=args.batch_size,
                                                      drop_last=True, shuffle=True, **kwargs)
            self.trainloader_target = data.DataLoader(trainset_target, batch_size=args.batch_size,
                                                      drop_last=True, shuffle=True, **kwargs)
        self.testloader = data.DataLoader(testset, batch_size=1, drop_last=False,
                                          shuffle=False, **kwargs)

        if args.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        device_ids = list(range(torch.cuda.device_count()))
        # segmentation model
        model = get_segmentation_model(
            args.model, num_classes=args.num_classes, pre_train=args.restore_from
        )
        model.train()
        # discriminators
        model_D1 = get_discriminator_model('discriminator', num_classes=args.num_classes)
        model_D1 = nn.DataParallel(model_D1, device_ids=device_ids) if args.batch_size > 1 else model_D1  # noqa: E501
        model_D2 = get_discriminator_model('discriminator', num_classes=args.num_classes)
        model_D2 = nn.DataParallel(model_D2, device_ids=device_ids) if args.batch_size > 1 else model_D2  # noqa: E501
        model_wD = get_discriminator_model(
            'cw_discriminator', num_features=2048, num_classes=args.num_classes
        )
        model_wD = nn.DataParallel(model_wD, device_ids=device_ids) if args.batch_size > 1 else model_wD  # noqa: E501
        model_D1.train()
        model_D2.train()
        model_wD.train()
        # if self.args.use_pseudo or self.args.use_weak:
        #    model_D1.load_state_dict(torch.load(self.args.restore_from.replace('G-', 'D1-')))
        #    model_D2.load_state_dict(torch.load(self.args.restore_from.replace('G-', 'D2-')))

        # optimizer
        cudnn.enabled = True
        cudnn.benchmark = True

        opt_param = model.optim_parameters(args)
        optimizer = optim.SGD(
            opt_param, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay
        )
        optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_wD = optim.Adam(model_wD.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

        # criterions
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.interp_source = nn.Upsample(
            size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
        self.interp_target = nn.Upsample(
            size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
        self.interp_target_eval = nn.Upsample(
            size=(input_size_target[1]*2, input_size_target[0]*2),
            mode='bilinear',
            align_corners=True
        )

        self.model, self.model_D1, self.model_D2, self.model_wD = \
            model, model_D1, model_D2, model_wD
        self.optimizer, self.optimizer_D1, self.optimizer_D2, self.optimizer_wD = \
            optimizer, optimizer_D1, optimizer_D2, optimizer_wD

        # using cuda
        self.model = self.model.to(self.device)
        self.model_D1 = self.model_D1.to(self.device)
        self.model_D2 = self.model_D2.to(self.device)
        self.model_wD = self.model_wD.to(self.device)

        self.bce_loss = self.bce_loss.to(self.device)
        self.seg_loss = self.seg_loss.to(self.device)

        self.source_label = 0
        self.target_label = 1
        self.best_pred = 0.0
        self.best_iter = 0
        self.is_best = False

    def __del__(self):
        if self.logger_fid:
            self.logger_fid.close()

    # learning rate scheduler
    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def adjust_learning_rate(self, optimizer, i_iter):
        lr = self.lr_poly(self.args.learning_rate, i_iter, self.args.num_steps, self.args.power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def adjust_learning_rate_D(self, optimizer, i_iter):
        lr = self.lr_poly(self.args.learning_rate_D, i_iter, self.args.num_steps, self.args.power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def get_weak_loss(self, preds, labels, device, weak_labels_onehot=None):
        weak_labels_onehot = weak_labels_onehot.to(device)
        preds_im = get_prediction(preds)
        tmp = weak_labels_onehot * torch.log(preds_im) + \
            (1-weak_labels_onehot) * torch.log(1-preds_im)
        loss = -torch.mean(tmp)
        return loss

    def get_adv_loss(self, preds, labels, bce_label, domain, weak_labels=None):
        threshold = self.source_th if domain == 'source' else self.target_th
        if weak_labels is None:
            weak_labels = get_weak_labels(labels, preds.shape[1], threshold)
        weak_labels = [(lbl > 0).nonzero().reshape(-1) for lbl in weak_labels]

        loss = [torch.tensor(0.0).to(preds.device)]
        for i in range(len(preds)):
            if len(weak_labels[i]) > 0:
                p = preds[i].index_select(0, weak_labels[i].to(self.device))
                if self.args.ls_gan:
                    loss += [
                        0.5 * torch.mean(
                            (
                                torch.clamp(torch.sigmoid(p), 1e-7, 1-1e-7) -
                                torch.tensor(1.-bce_label)
                            )**2
                        )
                    ]
                else:
                    t = torch.FloatTensor(p.data.size()).fill_(bce_label).to(self.device)
                    tmp = torch.stack([self.bce_loss(i, j) for i, j in zip(p, t)])
                    loss += [torch.mean(tmp)]
        loss = torch.mean(torch.stack(loss[1:]))
        return loss

    def training(self):
        raise Exception("Abstract function. Make class derivative and implement this function ...")

    # testing during training
    def validation(self, i_iter):
        print('Validating...')
        if self.logger_fid:
            print("Validating...", file=self.logger_fid)

        def eval(model, image, label, name, num_classes, id=0):

            with torch.no_grad():
                _, output = model(image)
            im_output = get_prediction(output).cpu()
            im_label = get_weak_labels(label, im_output.shape[1], self.target_th).cpu()

            if self.args.dataset_target == 'cityscapes':
                output = self.interp_target_eval(output).cpu().data[0].numpy()
            else:
                output = self.interp_target(output).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            label = label.cpu().numpy()

            # compute pixel accuracy
            correct, labeled = util.batch_pix_accuracy(output.copy(), label)

            # compute IoU
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            hist = util.compute_iou(output.copy(), label, num_classes)

            # save segmentation results
            if self.args.val_only:
                name = names[0].split('/')[-1]
                output_col = colorize_mask(output)
                # output = Image.fromarray(output)
                # output.save('%s/%s' % (self.args.result_dir, name))
                output_col.save('%s/%s_color.png' % (self.args.result_dir, name.split('.')[0]))

            return correct, labeled, hist, im_output, im_label

        self.model.eval()
        total_hists = np.zeros((self.args.num_classes, self.args.num_classes))
        total_correct, total_label = 0, 0
        im_output_stack, im_label_stack = torch.zeros(0), torch.zeros(0)

        for i, (images, labels, _, names, _) in enumerate(self.testloader):
            images = images.to(self.device)
            labels = labels.long().to(self.device)

            with torch.no_grad():
                corrects, labeleds, hists, im_output, im_label = eval(
                    self.model, images, labels, names, self.args.num_classes, i
                )
                im_output_stack = torch.cat([im_output_stack, im_output], dim=0)
                im_label_stack = torch.cat([im_label_stack, im_label], dim=0)

            if i % 25 == 0:
                print(f"[Image {i}/{len(self.testloader)}]")

            total_correct += corrects
            total_label += labeleds
            total_hists += hists

        # pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        mIoUs = np.diag(total_hists) / \
            (total_hists.sum(1) + total_hists.sum(0) - np.diag(total_hists))

        if self.args.dataset_source == 'synthia':
            mIoUs_16 = np.delete(mIoUs, [9, 14, 16])
            mIoUs_13 = np.delete(mIoUs, [9, 14, 16, 3, 4, 5])
            mIoU_16 = round(np.nanmean(mIoUs_16) * 100, 2)
            mIoU_13 = round(np.nanmean(mIoUs_13) * 100, 2)
            print("Source dataset is SYNTHIA -> Evaluating for specific classes:")
            print("mIoU_16 = {:f}".format(mIoU_16))
            print("mIoU_13 = {:f}".format(mIoU_13))
            if self.logger_fid:
                print("Source dataset is SYNTHIA -> Evaluating for specific classes:",
                      file=self.logger_fid)
                print("mIoU_16 = {:f}".format(mIoU_16), file=self.logger_fid)
                print("mIoU_13 = {:f}".format(mIoU_13), file=self.logger_fid)

        print("Per-class IoUs:")
        print(mIoUs)
        mIoU = round(np.nanmean(mIoUs) * 100, 2)
        print("mIoU (all classes) = {:f}".format(mIoU))
        if self.logger_fid:
            print("Per-class IoUs:", file=self.logger_fid)
            print(mIoUs, file=self.logger_fid)
            print("mIoU (all classes) = {:f}".format(mIoU), file=self.logger_fid, flush=True)

        # save the current best model
        # new_pred = (pixAcc + mIoU)/2
        if self.args.dataset_source == 'synthia':
            new_pred = mIoU_13
        else:
            new_pred = mIoU

        if new_pred > self.best_pred and not self.args.val_only:
            self.best_pred = new_pred
            self.best_iter = i_iter
            print(f"Storing new best model at iteration {i_iter}")
            if self.logger_fid:
                print(f"Storing new best model at iteration {i_iter}", file=self.logger_fid)
            torch.save(
                self.model.state_dict(),
                osp.join(
                    self.args.snapshot_dir,
                    'G-%s-%s.pth' % (self.args.dataset_source, self.args.dataset_target)
                )
            )
            torch.save(
                self.model_D1.state_dict(),
                osp.join(
                    self.args.snapshot_dir,
                    'D1-%s-%s.pth' % (self.args.dataset_source, self.args.dataset_target)
                )
            )
            torch.save(
                self.model_D2.state_dict(),
                osp.join(
                    self.args.snapshot_dir,
                    'D2-%s-%s.pth' % (self.args.dataset_source, self.args.dataset_target)
                )
            )
            torch.save(
                self.model_wD.state_dict(),
                osp.join(
                    self.args.snapshot_dir,
                    'wD-%s-%s.pth' % (self.args.dataset_source, self.args.dataset_target)
                )
            )
        return new_pred

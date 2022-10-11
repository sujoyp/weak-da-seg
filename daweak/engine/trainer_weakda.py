###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2021
###########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trainer_base import Trainer, get_prediction, get_weak_labels


def get_psuedo_weak_labels(preds, th):
    return (get_prediction(preds) > th).float()


dropout = nn.Dropout(0.1)


def get_pooled_feat(pred, feat):
    '''pred and feat should be of the same dimension, i.e., (B, x, H, W) where x will be num_class
    and num_features respectively'''

    p_shape = pred.shape
    f_shape = feat.shape

    pred = pred.view(p_shape[0], p_shape[1], -1)
    pred = pred.permute(0, 2, 1)
    feat = feat.view(f_shape[0], f_shape[1], -1)
    feat = feat.permute(0, 2, 1)
    # (B, H*W, x)

    pred_attention = F.softmax(pred, dim=1)
    feat_attention = []
    for i in range(pred.shape[0]):
        feat_attention.append(torch.mm(feat[i].t(), pred_attention[i]))

    feat_attention = torch.stack(feat_attention)
    # (B, F, C)

    return feat_attention


class TrainerWeakda(Trainer):

    def training(self):
        print('Starting Training')
        print('Total Iterations:', self.args.num_steps)
        print('Exp = {}'.format(self.args.snapshot_dir))
        if self.logger_fid:
            print('Starting Training', file=self.logger_fid)
            print('Total Iterations:', self.args.num_steps, file=self.logger_fid)
            print('Exp = {}'.format(self.args.snapshot_dir), file=self.logger_fid)
        eps = 1e-7

        iter_source = enumerate(self.trainloader_source)
        iter_target = enumerate(self.trainloader_target)

        for i_iter in range(self.args.num_steps):

            loss_seg_value1 = 0
            loss_adv_target_value1 = 0
            loss_D_value1 = 0

            loss_seg_value2 = 0
            loss_adv_target_value2 = 0
            loss_D_value2 = 0
            loss_weak_cwadv_value2 = 0.0
            loss_weak_D2 = 0.0
            loss_weak_D_value2 = 0.0
            loss_weak_target2 = 0.0
            loss_point_value = 0.0

            self.optimizer.zero_grad()
            self.optimizer_D1.zero_grad()
            self.optimizer_D2.zero_grad()
            self.optimizer_wD.zero_grad()
            self.adjust_learning_rate_D(self.optimizer_D1, i_iter)
            self.adjust_learning_rate_D(self.optimizer_D2, i_iter)
            self.adjust_learning_rate_D(self.optimizer_wD, i_iter)
            self.adjust_learning_rate(self.optimizer, i_iter)

            # don't accumulate gradients in discriminators
            for param in self.model_D1.parameters():
                param.requires_grad = False

            for param in self.model_D2.parameters():
                param.requires_grad = False

            for param in self.model_wD.parameters():
                param.requires_grad = False

            # train with source data
            _, batch = iter_source.__next__()
            images, labels, _, name = batch
            images = images.to(self.device)
            labels = labels.long().to(self.device)

            _, pred2, _, feat2 = self.model(images, get_features=True)

            if self.args.use_weak_cw:
                p_feat2 = get_pooled_feat(pred2.detach(), feat2.detach())

            # pred1 = self.interp_source(pred1)
            pred2 = self.interp_source(pred2)

            # segmentation loss
            loss_seg1 = 0  # self.seg_loss(pred1, labels)
            loss_seg2 = self.seg_loss(pred2, labels)
            loss = loss_seg2 + self.args.lambda_seg * loss_seg1
            loss.backward()
            loss_seg_value1 += 0  # loss_seg1.item()
            loss_seg_value2 += loss_seg2.item()

            if self.args.source_only:
                self.optimizer.step()
            else:
                # train with target data
                _, batch = iter_target.__next__()
                images, labels_target, _, name, point_label_list = batch
                images = images.to(self.device)

                _, pred_target2, _, feat_target2 = self.model(images, get_features=True)

                weak_labels_onehot = None
                if self.args.use_pseudo:
                    weak_labels_onehot = get_psuedo_weak_labels(
                        pred_target2.detach(), self.args.pweak_th)
                else:
                    weak_labels_onehot = get_weak_labels(
                        labels_target, pred_target2.shape[1], self.target_th)
                weak_labels_onehot = weak_labels_onehot.to(self.device)

                if self.args.use_pointloss:
                    point_label_list = point_label_list.long().to(self.device)
                    loss_point = []
                    tmp_interp = self.interp_target(pred_target2)
                    for p in point_label_list[0]:
                        if not weak_labels_onehot[0][p[2]]:
                            continue
                        tmp = torch.softmax(tmp_interp[0, :, p[0], p[1]], dim=0)
                        loss_point.append(-torch.log(tmp)[p[2]])
                    loss_point = torch.mean(torch.stack(loss_point))
                else:
                    loss_point = torch.tensor(0.)

                d_pred_target2 = dropout(pred_target2)
                if self.args.use_weak_cw:
                    p_feat_target2 = get_pooled_feat(pred_target2.detach(), feat_target2)
                    wD_out2 = self.model_wD(p_feat_target2)

                loss_weak_target2 = 0.0
                if self.args.use_weak:
                    d_labels_target = labels_target.unsqueeze(1)
                    loss_weak_target2 = self.get_weak_loss(
                        d_pred_target2, d_labels_target, self.device, weak_labels_onehot)

                # pred_target1 = self.interp_target(pred_target1)
                pred_target2 = self.interp_target(pred_target2)

                loss_weak_cwadv2 = 0.0
                if self.args.use_weak_cw:
                    loss_weak_cwadv2 = self.get_adv_loss(
                        wD_out2, d_labels_target, self.source_label, 'target', weak_labels_onehot)
                D_out2 = self.model_D2(F.softmax(pred_target2, dim=1))

                # adversarial loss
                if self.args.ls_gan:
                    D_out2 = torch.clamp(torch.sigmoid(D_out2), eps, 1-eps)
                    loss_adv_target1 = 0.  # 0.5*torch.mean((D_out1 - torch.tensor(1.))**2)
                    loss_adv_target2 = 0.5*torch.mean((D_out2 - torch.tensor(1.))**2)
                else:
                    loss_adv_target1 = 0.  # self.bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(self.source_label).to(self.device))  # noqa: E501
                    loss_adv_target2 = self.bce_loss(
                        D_out2,
                        torch.FloatTensor(D_out2.data.size()).fill_(
                            self.source_label).to(self.device)
                    )

                loss = self.args.lambda_adv_target1 * loss_adv_target1 + \
                    self.args.lambda_adv_target2 * loss_adv_target2
                loss += self.args.lambda_weak_target2 * loss_weak_target2 + loss_point.cuda() + \
                    self.args.lambda_weak_cwadv2 * loss_weak_cwadv2

                loss.backward()
                loss_adv_target_value2 += loss_adv_target2.item()
                if self.args.use_weak_cw:
                    loss_weak_cwadv_value2 += loss_weak_cwadv2.item()
                loss_point_value = loss_point.item()

                # train discriminators

                # bring back gradients for discriminators
                for param in self.model_D1.parameters():
                    param.requires_grad = True

                for param in self.model_D2.parameters():
                    param.requires_grad = True

                for param in self.model_wD.parameters():
                    param.requires_grad = True

                # train with source data
                # pred1 = pred1.detach()
                pred2 = pred2.detach()

                # D_out1 = self.model_D1(F.softmax(pred1, dim=1))
                D_out2 = self.model_D2(F.softmax(pred2, dim=1))

                # discriminator loss
                if self.args.ls_gan:
                    # D_out1 = torch.clamp(torch.sigmoid(D_out1), eps, 1-eps)
                    D_out2 = torch.clamp(torch.sigmoid(D_out2), eps, 1-eps)
                    # loss_D1 = 0.  # 0.5*torch.mean((D_out1 - torch.tensor(1.))**2)
                    loss_D2 = 0.5*torch.mean((D_out2 - torch.tensor(1.))**2)
                else:
                    # loss_D1 = 0.  # self.bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(self.source_label).to(self.device))  # noqa: E501
                    loss_D2 = self.bce_loss(
                        D_out2,
                        torch.FloatTensor(D_out2.data.size()).fill_(
                            self.source_label).to(self.device)
                    )

                # loss_D1 = loss_D1 / 2
                loss_D2 = loss_D2 / 2
                # loss_D1.backward()
                loss_D2.backward()
                # loss_D_value1 += loss_D1.item()
                loss_D_value2 += loss_D2.item()

                if self.args.use_weak_cw:
                    D_weak_out2 = self.model_wD(p_feat2.detach())
                    loss_weak_D2 = self.get_adv_loss(
                        D_weak_out2, labels.unsqueeze(1), self.source_label, 'source')
                    loss_weak_D2 = loss_weak_D2 / 2
                    loss_weak_D2.backward()
                    loss_weak_D_value2 += loss_weak_D2.item()

                # train with target data
                # pred_target1 = pred_target1.detach()
                pred_target2 = pred_target2.detach()

                # D_out1 = self.model_D1(F.softmax(pred_target1, dim=1))
                D_out2 = self.model_D2(F.softmax(pred_target2, dim=1))

                # discriminator loss
                if self.args.ls_gan:
                    # D_out1 = torch.clamp(torch.sigmoid(D_out1), eps, 1-eps)
                    D_out2 = torch.clamp(torch.sigmoid(D_out2), eps, 1-eps)
                    # loss_D1 = 0.  # 0.5*torch.mean((D_out1 - torch.tensor(0.))**2)
                    loss_D2 = 0.5*torch.mean((D_out2 - torch.tensor(0.))**2)
                else:
                    # loss_D1 = 0.  # self.bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(self.target_label).to(self.device))  # noqa: E501
                    loss_D2 = self.bce_loss(
                        D_out2,
                        torch.FloatTensor(D_out2.data.size()).fill_(
                            self.target_label).to(self.device)
                    )

                loss_D2 = loss_D2 / 2
                loss_D2.backward()
                loss_D_value2 += loss_D2.item()
                if self.args.use_weak_cw:
                    D_weak_out2 = self.model_wD(p_feat_target2.detach())
                    loss_weak_D2 = self.get_adv_loss(
                        D_weak_out2,
                        d_labels_target,
                        self.target_label,
                        'target',
                        weak_labels_onehot
                    )
                    loss_weak_D2 = loss_weak_D2 / 2
                    loss_weak_D2.backward()
                    loss_weak_D_value2 += loss_weak_D2.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                self.optimizer_D2.step()
                if self.args.use_weak_cw:
                    self.optimizer_wD.step()

            # print loss
            if i_iter % self.args.print_loss_every == 0 or i_iter >= self.args.num_steps_stop - 1:
                loss_txt = (
                    'iter = {0:8d}/{1:8d}, seg1 = {2:.3f} seg2 = {3:.3f} adv1 = {4:.3f}, '
                    'adv2 = {5:.3f}, D1 = {6:.3f} D2 = {7:.3f}, weak = {8:.3f}, wadv2 = {9:.3f}, '
                    'wD2 = {10:.3f}, pl = {11:.3f}'.format(
                        i_iter, self.args.num_steps, loss_seg_value1, loss_seg_value2,
                        loss_adv_target_value1, loss_adv_target_value2, loss_D_value1,
                        loss_D_value2, loss_weak_target2, loss_weak_cwadv_value2,
                        loss_weak_D_value2, loss_point_value)
                )
                print(loss_txt)
                if self.logger_fid:
                    print(loss_txt, file=self.logger_fid, flush=True)

            # test during training
            if self.args.val:
                if (i_iter % self.args.save_pred_every == 0 and i_iter != 0) \
                   or i_iter >= self.args.num_steps_stop - 1:
                    self.validation(i_iter)
                    self.is_best = True

                    if self.is_best:
                        print(
                            'Current best accuracy at iter %d: %f' %
                            (self.best_iter, self.best_pred)
                        )
                        if self.logger_fid:
                            print(
                                'Current best accuracy at iter %d: %f' %
                                (self.best_iter, self.best_pred),
                                file=self.logger_fid,
                            )
                    self.model.train()

            if i_iter >= self.args.num_steps_stop - 1:
                print(f"Stop training at {self.args.num_steps_stop} ('num-steps-stop') iters")
                if self.logger_fid:
                    print(
                        f"Stop training at {self.args.num_steps_stop} ('num-steps-stop') iters",
                        file=self.logger_fid
                    )
                break

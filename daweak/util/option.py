###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2019
###########################################################################

import argparse
import torch


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset
        parser.add_argument("--model", type=str, default='deeplab',
                            help="Model, available options : DeepLab")
        parser.add_argument("--dataset-source", type=str, default='gta5',
                            help="Source dataset, available options : 'gta5'")
        parser.add_argument("--dataset-target", type=str, default='cityscapes',
                            help="Target dataset, available options : cityscapes")
        parser.add_argument("--data-root", type=str, default='./data',
                            help="Path to the dataset directory containing the image list.")
        parser.add_argument("--data-path-source", type=str, default='./data',
                            help="Path to the source directory containing the dataset.")
        parser.add_argument("--data-path-target", type=str, default='./data',
                            help="Path to the target directory containing the dataset.")
        parser.add_argument("--num-workers", type=int, default=1,
                            help="number of workers for multithread dataloading.")
        parser.add_argument("--ignore-label", type=int, default=255,
                            help="The index of the label to ignore during the training.")
        parser.add_argument("--input-size-source", type=str, default='1280,720',
                            help="Comma-separated string with height and width of source images.")
        parser.add_argument("--input-size-target", type=str, default='1024,512',
                            help="Comma-separated string with height and width of target images.")
        parser.add_argument("--num-classes", type=int, default=19,
                            help="Number of classes to predict (including background).")
        parser.add_argument('--source-split', type=str, default='train',
                            help='dataset train split for source (default: train)')
        parser.add_argument('--target-split', type=str, default='val',
                            help='dataset train split for target (default: train)')
        parser.add_argument('--test-split', type=str, default='val',
                            help='dataset validation split for target (default: val)')
        # training hyper params
        parser.add_argument("--batch-size", type=int, default=1,
                            help="Number of images sent to the network in one step.")
        parser.add_argument("--num-steps", type=int, default=250000,
                            help="Number of training steps.")
        parser.add_argument("--num-steps-stop", type=int, default=150000,
                            help="Number of training steps for early stopping.")
        parser.add_argument("--seed", type=int, default=1,
                            help="Random seed to have reproducible results.")
        parser.add_argument("--lambda-seg", type=float, default=0.1,
                            help="lambda_seg.")
        parser.add_argument("--lambda-adv-target1", type=float, default=0.0002,
                            help="lambda_adv for adversarial training.")
        parser.add_argument("--lambda-adv-target2", type=float, default=0.001,
                            help="lambda_adv for adversarial training.")
        parser.add_argument("--cpu", action='store_true', default=False,
                            help="choose to use cpu device.")
        # optimizer params
        parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                            help="Base learning rate for training with polynomial decay.")
        parser.add_argument("--learning-rate-D", type=float, default=1e-4,
                            help="Base learning rate for discriminator.")
        parser.add_argument("--power", type=float, default=0.9,
                            help="Decay parameter to compute the learning rate.")
        parser.add_argument("--momentum", type=float, default=0.9,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--weight-decay", type=float, default=0.0005,
                            help="Regularisation parameter for L2-loss.")
        # checking point
        parser.add_argument("--restore-from", type=str, default=None,
                            help="Where restore model parameters from.")
        parser.add_argument("--save-pred-every", type=int, default=5000,
                            help="Save summaries and checkpoint every often.")
        parser.add_argument("--print-loss-every", type=int, default=100,
                            help="Print loss every often.")
        parser.add_argument("--snapshot-dir", type=str, default='./model',
                            help="Where to save snapshots of the model.")
        parser.add_argument('--ls-gan', action='store_true', default=False,
                            help='use LS-GAN loss')
        parser.add_argument('--use-weak', action='store_true', default=False,
                            help='use weakly supervised loss')
        parser.add_argument('--use-weak-cw', action='store_true', default=False,
                            help='use weakly supervised classwise loss between source and target')
        parser.add_argument('--lambda-weak-target2', type=float, default=0.2,
                            help='lambda_weak for weakly supervision loss')
        parser.add_argument('--lambda-weak-cwadv2', type=float, default=0.001,
                            help='lambda_weak for weak supervision class-wise loss between source and target')
        parser.add_argument('--save-feat', action='store_true', default=False,
                            help='eval mode to save features')
        parser.add_argument('--pweak-th', type=float, default=0.2,
                            help='threshold for pseudo-weak labels')
        parser.add_argument('--use-pseudo', action='store_true', default=False,
                            help='use pseudo labels or not')
        parser.add_argument('--source-only', action='store_true', default=False,
                            help='train only on source')
        parser.add_argument('--use-pointloss', action='store_true', default=False,
                            help='use point loss or not')
        parser.add_argument('--use-pixeladapt', action='store_true', default=False,
                            help='use pixel adaptation or not')

        # evaluation option
        parser.add_argument('--val', action='store_true', default=False,
                            help='validation during training')
        parser.add_argument('--val-only', action='store_true', default=False,
                            help='run testing')
        parser.add_argument("--result-dir", type=str, default='./result',
                            help="Where to save segmentation results.")

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.cpu and torch.cuda.is_available()
        print(args)
        return args

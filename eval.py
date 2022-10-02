###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2021
###########################################################################

import os
import numpy as np
import torch
import socket

from daweak.engine import Trainer
import daweak.util as util

print('Host Name: %s' % socket.gethostname())
print('Process ID: %s' % os.getpid())


def main():
    args = util.Options().parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.val_only = True
    trainer = Trainer(args)
    trainer.validation(i_iter=None)


if __name__ == "__main__":
    main()

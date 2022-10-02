###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2021
###########################################################################

import os
import numpy as np
import torch
import socket

from daweak.engine import TrainerWeakda
import daweak.util as util

print('Host Name: %s' % socket.gethostname())
print('Process ID: %s' % os.getpid())


def main():
    args = util.Options().parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    trainer = TrainerWeakda(args)

    if args.val_only:
        pred = trainer.validation(i_iter=None)
        print(pred)
    else:
        trainer.training()


if __name__ == "__main__":
    main()

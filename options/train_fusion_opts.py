import os
from argparse import ArgumentParser

import options.arg as arg

class TrainFusionOpts:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--device', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--insize', type=int, default=256)
        self.parser.add_argument('--outsize', type=int, default=1024)

        self.parser.add_argument('--eval_interval', type=int, default=5000)
        self.parser.add_argument('--eval_num', type=int, default=50)
        self.parser.add_argument('--save_interval', type=int, default=5000)
        self.parser.add_argument('--board_interval', default=100, type=int, help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--max_steps', type=int, default=200000)
        self.parser.add_argument('--start_from_latent_avg', type=bool, default=True)

        # loss
        self.parser.add_argument('--lpips_type', default='alex', type=str, help='LPIPS backbone')
        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')

    def parse(self):
        opts = self.parser.parse_args()
        opts.trainset_lq_path = arg.dataset_paths['trainset_path_ghosting']
        opts.testset_lq_path = arg.dataset_paths['testset_path_ghosting']
        opts.trainset_tg_path = arg.dataset_paths['trainset_path']
        opts.testset_tg_path = arg.dataset_paths['testset_path']

        folder_name = "trainerEF" + "_bs"+str(opts.batch_size) + "_"+str(opts.max_steps)+"iter"
        opts.exp_dir = os.path.join("experiment", "EF", folder_name)

        return opts

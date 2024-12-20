import os
from argparse import ArgumentParser

import options.arg as arg


class TrainHgnnOpts:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--device', type=int, default=0, help='GPU device id')
        self.parser.add_argument('--batch_size', default=6, type=int, help='Batch size of train loader')
        self.parser.add_argument('--num_workers', default=4, type=int, help='num_workers of train loader')

        self.parser.add_argument('--eval_interval', type=int, default=200, help='every which evaluates')
        self.parser.add_argument('--eval_num', type=int, default=30, help='number of images to evaluate during validation')
        self.parser.add_argument('--save_interval', type=int, default=200, help='number of iterations between saving checkpoints')
        self.parser.add_argument('--print_interval', type=int, default=10, help='number of iterations between print log')

        self.parser.add_argument('--max_steps', type=int, default=1000, help='number of iterations to train')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')

        self.parser.add_argument('--alpha', type=int, default=20, help="coefficient to be multiplied to directions")

        # psp parameters
        self.parser.add_argument('--psp_encoder_type', type=str, default='GradualStyleEncoder')
        self.parser.add_argument('--psp_input_nc', type=int, default=3)
        self.parser.add_argument('--psp_output_size', type=int, default=1024)
        self.parser.add_argument('--psp_start_from_latent_avg', type=bool, default=True)
        self.parser.add_argument('--psp_learn_in_w', type=bool, default=False)

        # Hgnn parameters
        self.parser.add_argument('--num_cycle', type=int, default=3)
        self.parser.add_argument('--K', type=int, default=25)
        self.parser.add_argument('--num_edge', type=int, default=256)

        self.parser.add_argument('--h', type=float, default=0.1)
        self.parser.add_argument('--m', type=float, default=0.01)


    def parse(self):
        opts = self.parser.parse_args()
        opts.trainset_path = arg.dataset_paths['trainset_path']
        opts.testset_path = arg.dataset_paths['testset_path']

        opts.psp_ckptpath = arg.model_paths['pSp']
        opts.direction_path = arg.direction_path

        folder_name = "trainerDH" + "_bs"+str(opts.batch_size) + "_"+str(opts.max_steps)+"iter" \
                    + "_h"+str(opts.h) + "_m"+str(opts.m)
        opts.exp_dir = os.path.join("experiment", "DH", folder_name)
        return opts
    
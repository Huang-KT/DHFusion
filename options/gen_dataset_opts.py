from argparse import ArgumentParser

import options.arg as arg


class GenDatasetOpts:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--device', type=int, default=0)
        self.parser.add_argument('--hgnncam_ckpt_path', type=str, required=True)
        self.parser.add_argument('--hgnncam_img_size', type=int, default=256)
        self.parser.add_argument('--hgnncam_num_class', type=int, default=7)

        self.parser.add_argument('--alpha', type=int, default=20, help="coefficient to be multiplied to directions")
        self.parser.add_argument('--src_image_dir', type=str, required=True)
        self.parser.add_argument('--dst_image_dir', type=str, required=True)

        ## psp parameters
        self.parser.add_argument('--psp_encoder_type', type=str, default='GradualStyleEncoder')
        self.parser.add_argument('--psp_input_nc', type=int, default=3)
        self.parser.add_argument('--psp_output_size', type=int, default=1024)
        self.parser.add_argument('--psp_start_from_latent_avg', type=bool, default=True)
        self.parser.add_argument('--psp_learn_in_w', type=bool, default=False)

        ## hgnn parameters
        self.parser.add_argument('--num_cycle', type=int, default=3)
        self.parser.add_argument('--K', type=int, default=25)
        self.parser.add_argument('--num_edge', type=int, default=256)

    def parse(self):
        opts = self.parser.parse_args()
        opts.psp_ckptpath = arg.model_paths['pSp']
        opts.direction_dir = arg.direction_path

        return opts
        
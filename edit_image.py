import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import random
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image

from options.edit_image_opts import EditImageOpts
from model.pSp.psp import pSp
from model.dhnet.hgnncam import HgnnCAM
from model.efnet.fusion import Fusion
from options.arg import random_seed, direction_list, model_paths

# set seed
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True


class Empty:
    pass

def tensor2np(tensor):
    tensor = tensor.squeeze(0)\
        .float().detach().cpu().clamp_(0, 1)
    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    img_np = (img_np * 255.0).round()
    img_np = img_np.astype(np.uint8)
    return img_np

def main(opts):
    opts.device = "cuda:"+str(opts.device)
    hgnncam = HgnnCAM(num_class=opts.hgnncam_num_class, num_cycle=opts.num_cycle,
                      K_neigs=opts.K, num_edge=opts.num_edge)
    hgnncam_state = torch.load(opts.hgnncam_ckpt_path, map_location=torch.device('cpu'))
    hgnncam.load_state_dict(hgnncam_state)
    print("Loading hgnncam from checkpoint:", opts.hgnncam_ckpt_path)

    fusion = Fusion(opts.fusion_in_size, opts.fusion_out_size,
                    start_from_latent_avg=opts.start_from_latent_avg, pretrain=model_paths)
    fusion_state = torch.load(opts.fusion_ckpt_path)
    fusion.load_state_dict(fusion_state)
    print("Loading fusion from checkpoint:", opts.fusion_ckpt_path)

    
    psp_opts = Empty()
    for attr in dir(opts):
        if 'psp' in attr:
            exec(f"psp_opts.{attr.replace('psp_', '')} = opts.{attr}")
    psp_opts.device = opts.device
    psp = pSp(psp_opts)


    psp = psp.to(opts.device)
    hgnncam = hgnncam.to(opts.device)
    fusion = fusion.to(opts.device)
    psp.eval(); hgnncam.eval(); fusion.eval();
    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    print("Save to", opts.output_dir)

    
    for direction_name in direction_list:
        
        direction_dir = os.path.join(opts.output_dir, direction_name)
        if os.path.exists(direction_dir):
            raise Exception('Oops... {} already exists'.format(direction_dir))
        os.makedirs(direction_dir)

        print("direction:", direction_name)
        direction_path = os.path.join("./directions", direction_name+".npy")
        direction = np.load(direction_path)
        direction = direction / np.sqrt((direction * direction).sum())
        direction = torch.from_numpy(direction).float().to(opts.device).unsqueeze(0)
        

        visual = False

        for path in tqdm(os.listdir(opts.image_dir)):
            image_path = os.path.join(opts.image_dir, path)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = totensor(img)
            with torch.no_grad():
                origin_1024 = img.unsqueeze(0)
                origin = F.interpolate(origin_1024,
                                       (opts.hgnncam_img_size, opts.hgnncam_img_size),
                                       mode='area').to(opts.device)
                inverted, latent = psp(origin, resize=True, return_latents=True)
                inverted_256 = psp.face_pool(inverted)
                latent_pi = latent + opts.alpha * direction
                manipulated, _ = psp.decoder([latent_pi], input_is_latent=True, return_latents=False)
                manipulated_256 = psp.face_pool(manipulated)
                image_forward = torch.cat((inverted_256, manipulated_256), dim=1)
                _, [hm_list, _] = hgnncam(image_forward)

            heat_map = hm_list[-1]
            heat_map = mm_normal(heat_map)

            interpolate_mode = 'bilinear'
            heat_map1024 = F.interpolate(heat_map, (1024, 1024), mode=interpolate_mode)
            
            heat_map1024_np = heat_map1024.squeeze().cpu().detach().numpy()
            heat_visual = show_cam_on_image(tensor2np((manipulated + 1) / 2) / 255.0, heat_map1024_np)

            hm16 = heat_map  # (16, 16)
            hm32 = F.interpolate(hm16, (32, 32), mode=interpolate_mode)
            hm64 = F.interpolate(hm16, (64, 64), mode=interpolate_mode)
            output = fusion(x=origin, x_edit=manipulated_256, hm=[hm16, hm32, hm64])

            if visual:
                output_np = tensor2np((output + 1) / 2)
                origin_1024_np = tensor2np((origin_1024 + 1) / 2)
                manipulated_np = tensor2np((manipulated + 1) / 2)
                inverted_np = tensor2np((inverted + 1) / 2)
                img_np = np.concatenate((origin_1024_np, inverted_np, manipulated_np, heat_visual, output_np), axis=1)
                cv2.imwrite(os.path.join(opts.output_dir, direction_name, path), img_np)

            else:
                output_np = tensor2np((output + 1) / 2)
                cv2.imwrite(os.path.join(opts.output_dir, direction_name, path), output_np)


def mm_normal(hm):
    hm_min, _ = torch.min(hm.view(hm.size(0), -1), dim=1, keepdim=True)
    hm_max, _ = torch.max(hm.view(hm.size(0), -1), dim=1, keepdim=True)
    hm_min = hm_min.view(-1, 1, 1, 1)
    hm_max = hm_max.view(-1, 1, 1, 1)
    hm = (hm - hm_min) / (hm_max - hm_min + 1e-8)
    return hm
    

if __name__ == '__main__':
    opts = EditImageOpts().parse()

    main(opts)

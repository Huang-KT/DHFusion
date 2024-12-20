dataset_paths = {
	'trainset_path': 'xxxx',  # FFHQ
	'testset_path': 'xxxx',  # CelebA-HQ
    'trainset_path_ghosting': 'xxxx',  # Ghosting FFHQ
    'testset_path_ghosting': 'xxxx',  # Ghosting CelebA-HQ
}

model_paths = {
    'pSp': './pretrained_model/psp_ffhq_encode.pt',
	'ir_se50': './pretrained_model/model_ir_se50.pth',
    'styleGAN2_weight_url': 'http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth',
}


direction_path = './directions'
direction_list = ['Beard', 'Bushy_Eyebrows', 'Eyeglasses', 'Mouth_Open', 'Narrow_Eyes', 'Old', 'Smiling']

random_seed = 2024

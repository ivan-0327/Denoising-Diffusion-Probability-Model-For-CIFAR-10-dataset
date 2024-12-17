

import os
import torch
from tqdm import trange
from metrics.both import get_inception_and_fid_score
from models.model import UNet
from training.diffusion import GaussianDiffusionSampler
from torchvision.utils import make_grid, save_image

def eval(arg):
    # model setup
    model = UNet(
        T=arg.T, ch=arg.ch, ch_mult=arg.ch_mult, attn=arg.attn,
        num_res_blocks=arg.num_res_blocks, dropout=arg.dropout)
    sampler = GaussianDiffusionSampler(
        model, arg.beta_1, arg.beta_T, arg.T, img_size=arg.img_size,
        mean_type=arg.mean_type, var_type=arg.var_type).to(arg.device)
    if arg.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(arg.logdir, 'ckpt.pt') ,  map_location = arg.device)

    model.load_state_dict(ckpt['ema_model'] )
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(arg.logdir, 'samples_ema.png'),
        nrow=16)

def evaluate(sampler, model ,arg):
    model.eval()
    
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, arg.num_images, arg.batch_size, desc=desc):
            batch_size = min(arg.batch_size, arg.num_images - i)
            x_T = torch.randn((batch_size, 3, arg.img_size, arg.img_size))
            batch_images = sampler(x_T.to(arg.device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, arg.fid_cache, num_images=arg.num_images,
        use_torch=arg.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images

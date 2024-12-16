import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import os 
from tqdm import trange
import copy
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
import json

from evaluating.eval_process import evaluate
from metrics.both import get_inception_and_fid_score
from models.model import UNet
from training.diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))
        
def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

def warmup_lr(step , warmup):
    return min(step, warmup) / warmup


# 將參數轉換為類似字串的格式
def flags_into_string(args):
    return ' '.join(f'--{key}={value} \n ' for key, value in vars(args).items())

def train( arg ):
    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=arg.batch_size, shuffle=True,
        num_workers=arg.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(
        T=arg.T, ch=arg.ch, ch_mult=arg.ch_mult, attn=arg.attn,
        num_res_blocks=arg.num_res_blocks, dropout=arg.dropout)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=arg.lr)
    #sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
    optim,
    lr_lambda=lambda step: warmup_lr(step, arg.warmup)
    )
    trainer = GaussianDiffusionTrainer(
        net_model, arg.beta_1, arg.beta_T, arg.T).to(arg.device)
    net_sampler = GaussianDiffusionSampler(
        net_model, arg.beta_1, arg.beta_T, arg.T, arg.img_size,
        arg.mean_type).to(arg.device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, arg.beta_1, arg.beta_T, arg.T, arg.img_size,
        arg.mean_type).to(arg.device)
    if arg.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    if not os.path.exists(os.path.join(arg.logdir, 'sample')):
        os.makedirs(os.path.join(arg.logdir, 'sample'))
    x_T = torch.randn(arg.sample_size, 3, arg.img_size, arg.img_size)
    x_T = x_T.to(arg.device)
    grid = (make_grid(next(iter(dataloader))[0][:arg.sample_size]) + 1) / 2
    writer = SummaryWriter(arg.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(arg.logdir, "flagfile.txt"), 'w') as f:
        f.write(flags_into_string(arg) )
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    start_step = 0
    ckpt_path = os.path.join(arg.logdir, 'ckpt.pt')
    #loading checkpoint
    if os.path.exists(ckpt_path) :
        print(f"loading check point from {ckpt_path}.")
        ckpt = torch.load(ckpt_path , map_location=arg.device )

        net_model.load_state_dict( ckpt['net_model'])
        ema_model.load_state_dict( ckpt['ema_model'])
        sched    .load_state_dict( ckpt['sched']    )
        optim    .load_state_dict( ckpt['optim'])
        start_step = ckpt['step']
        x_T        = ckpt['x_T']
        print(f"Check point loaded, starting from step {start_step}.")
    else:
        print(f"No found check point file, starting from ground.")
    # start training
    with trange(start_step , arg.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(arg.device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), arg.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, arg.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if arg.sample_step > 0 and step % arg.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        arg.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if arg.save_step > 0 and step % arg.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(arg.logdir, 'ckpt.pt'))

            # evaluate
            if arg.eval_step > 0 and step % arg.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, arg.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(arg.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()

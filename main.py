import os ,argparse
import torch
from evaluating.eval_process import eval
from training.train_process import train

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(arg):
    if arg.train == True :
        train(arg)
    elif arg.eval == True:
        eval(arg)
    else:
        print(f"Alert : Set --train or --eval to execute corresponding tasks")
if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="DDPM augment")
   parser.add_argument("--train" ,default=True , type=str2bool, help="train for DDPM.")
   parser.add_argument("--eval" , default=False , type=str2bool ,help="evaluate performance.")

   #UNet
   parser.add_argument("--ch"             , default=128       , type=int   , help="base channel of UNet.")
   parser.add_argument("--ch_mult"        , default=[1,2,2,2] , type=int   , help="channel multiplier")
   parser.add_argument("--attn"           , default=[1]       , type=int   , help=" add attention to these levels. ")
   parser.add_argument("--num_res_blocks" ,default= 2         , type=int   , help= "#resblock in each level.")
   parser.add_argument("--dropout"        ,default=0.1        , type=float , help="dropout rate of resblock.")

   #Gaussian Diffussion 
   parser.add_argument("--beta_1"    , default=1e-4      , type=float       , help= "start beta value. ")
   parser.add_argument("--beta_T"    , default=0.02      ,type=float        , help="end beta value.")
   parser.add_argument("--T"         , default=1000      , type= int        , help="total diffusion steps.")
   parser.add_argument("--mean_type" , default="epsilon"                    , help= "predict variable.")
   
   #Training
   parser.add_argument("--lr"            , default=2e-4     , type=float    , help="target learning rate.")
   parser.add_argument("--grad_clip"     , default=1.       , type = float  , help= "gradient norm clipping.")
   parser.add_argument("--total_steps"   , default=800000   , type=int      , help="total training steps .")
   parser.add_argument("--img_size"      , default=32       , type=int      , help="image size.")
   parser.add_argument("--warmup"        , default=5000     , type=int      , help= "learning rate warm up .")
   parser.add_argument("--batch_size"    , default=128      , type=int      , help="batch size ")
   parser.add_argument("--num_workers"   , default=4        , type=int      , help="workers of Dataloader.")
   parser.add_argument("--ema_decay"     , default= 0.9999  , type=float    , help="ema decay rate ")
   parser.add_argument("--parallel"      , default= False   , type=str2bool , help="multi gpu training ")

   #Logging & Sampling
   parser.add_argument("--logdir"       , default ='./logs/DDPM_CIFAR10_EPS'                , help='log directory')
   parser.add_argument("--sample_size"  , default =64                       , type=int      , help="sampling size of images")
   parser.add_argument("--sample_step"  ,default =1000                      , type=int      , help='frequency of sampling')

   #Evaluation
   parser.add_argument("--save_step"     , default=5000         , type=int                  , help="requency of saving checkpoints, 0 to disable during training")
   parser.add_argument("--eval_step"     , default= 0           , type=int                  , help= "frequency of evaluating model, 0 to disable during training")
   parser.add_argument("--num_images"    , default=50000        , type=int                  , help="the number of generated images for evaluation")
   parser.add_argument("--fid_use_torch" , default=False                                    , help= " calculate IS and FID on gpu")
   parser.add_argument("--fid_cache"     , default='./stats/cifar10.train.npz'              , help='FID cache')

   arg = parser.parse_args()
   arg.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
   # execute tasks
   main(arg =arg)


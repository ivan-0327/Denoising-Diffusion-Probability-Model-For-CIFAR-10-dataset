import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

                       
        # calculations for σ_t
        self.register_buffer(
            'sigma_t',
             torch.sqrt(( (1. - alphas_bar_prev) / (1. - alphas_bar) )* self.betas  )   ) 
        
        # epsilon θ coefficient
        self.register_buffer(
            'epsilon_coefficient',
              (1. - alphas) / torch.sqrt(1. - alphas_bar) ) 
        
        # 1 / square alphas_t 
        self.register_buffer(
            'mean_coefficient',
            1 / torch.sqrt(alphas) )
        

    def p_mean(self,eps,  x_t, t):
        """
        Compute the mean e of the diffusion posterior
        p(x_{t-1} | x_t)
        """
        
        posterior_mean = (
            extract(self.mean_coefficient, t, x_t.shape) * (x_t - extract(self.epsilon_coefficient, t, x_t.shape) *  eps )            
        )
        
        return posterior_mean


    def p_mean_sigma(self, x_t, t):
        # below: only log_variance is used in the KL computations
        
        model_sigma = extract(self.sigma_t, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'epsilon':   # the model predicts epsilon
            eps        = self.model(x_t, t)
            model_mean = self.p_mean(eps, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        

        return model_mean, model_sigma

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, sigma = self.p_mean_sigma(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + sigma * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

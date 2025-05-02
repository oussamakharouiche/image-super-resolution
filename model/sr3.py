import torch 
import torch.nn as nn


def linear_beta_schedule(T):
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    cumulative_alphas = torch.cumprod(alphas, dim=0)
    return betas, cumulative_alphas


class Gaussiendiffusion(nn.Module):
    def __init__(self, denoise_fn, T, device, loss_type = "l2"):
        super(Gaussiendiffusion, self).__init__()
        self.T = T
        self.device = device
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.cumulative_alphas = linear_beta_schedule(T)[1]
        self.register_buffer("sqrt_alpha_t", torch.sqrt(self.cumulative_alphas))
        self.register_buffer("sqrt_one_minus_alpha_t", torch.sqrt(1 - self.cumulative_alphas))

    def Loss(self, device):
        if self.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduce="mean").to(device=device)
        elif self.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduce="mean").to(device=device)

    def forward(self, x_in, x_cond):
        t = torch.randint(0, self.T, (x_in.size(0),))

        noise = torch.randn_like(x_in).to(self.device)
        x = self.sqrt_alpha_t[t].view(-1, 1, 1, 1) * x_in + self.sqrt_one_minus_alpha_t[t].view(-1, 1, 1, 1) * noise
        self.Loss(x_in.device)
        return self.loss_func(noise, self.denoise_fn(torch.cat([x_cond,x], dim=1),t))
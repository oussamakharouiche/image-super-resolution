import torch
import torch.nn as nn

class CustumSequentiel(nn.Module):
    def __init__(self, *models):
        super(CustumSequentiel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, noise_emb):
        for model in self.models:
            if isinstance(model, FeatureWiseAffine) or isinstance(model, ResBlock):
                x = model(x, noise_emb)
            else:
                x = model(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim_out):
        super(PositionalEncoding, self).__init__()
        self.dim_out = dim_out
        self.n = 10000
        
    def forward(self, t):
        pos  = (torch.arange(self.dim_out, dtype=torch.float32, device=t.device) // 2 )*2
        denom = torch.pow(self.n, pos / self.dim_out)

        t = t.view(-1,1)
        
        angles = t / denom

        out = torch.zeros_like(angles, device = t.device)

        out[:,0::2] = torch.sin(angles[:,0::2])
        out[:,1::2] = torch.cos(angles[:,1::2])

        return out

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine=False):
        super(FeatureWiseAffine, self).__init__()

        self.use_affine = use_affine

        self.func = nn.Linear(in_channels, out_channels*(1+self.use_affine))

    def forward(self, x, noise_emb):
        noise_emb = noise_emb.to(x.device)

        if self.use_affine:
            alpha, bias = self.func(noise_emb).view(x.size(0), -1, 1, 1).chunck(2, dim=1)
            return (1+alpha) * x + bias
        else:
            return x + self.func(noise_emb).view(x.size(0), -1, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, noise_emb_dim, use_affine):
        super(ResBlock, self).__init__()
        self.model_ultim = CustumSequentiel(
            nn.GroupNorm(input_channels//16 if input_channels>=16 else 1, input_channels),
            nn.SiLU(),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, padding='same'),
            FeatureWiseAffine(noise_emb_dim, input_channels, use_affine=use_affine),
            nn.GroupNorm(input_channels//16 if input_channels>=16 else 1, input_channels),
            nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding='same'),
            FeatureWiseAffine(noise_emb_dim, output_channels, use_affine=use_affine),
            nn.GroupNorm(output_channels//16 if output_channels>=16 else 1, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding='same'),
            FeatureWiseAffine(noise_emb_dim, output_channels, use_affine=use_affine),
            nn.GroupNorm(output_channels//16 if output_channels>=16 else 1, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=1, padding='same')
        )
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding='same') if input_channels != output_channels else nn.Identity()


    def forward(self, x, noise_emb):
        h = self.model_ultim(x, noise_emb)
        
        return (self.conv(x) / torch.sqrt(torch.tensor(2.0))) + h
    
class Unet_encoder(nn.Module):
    def __init__(self, input_channels, noise_emb_dim, base_channels = 64, use_affine = False):
        super(Unet_encoder, self).__init__()
        self.enc1 = CustumSequentiel(
            ResBlock(input_channels, base_channels, noise_emb_dim, use_affine),
            ResBlock(base_channels,  base_channels, noise_emb_dim, use_affine),
            ResBlock(base_channels,  base_channels, noise_emb_dim, use_affine),
        )
        self.enc2 = CustumSequentiel(
            ResBlock(base_channels, base_channels*2, noise_emb_dim, use_affine),
            ResBlock(base_channels*2, base_channels*2, noise_emb_dim, use_affine),
            ResBlock(base_channels*2, base_channels*2, noise_emb_dim, use_affine)
        )
        self.enc3 = CustumSequentiel(
            ResBlock(base_channels*2, base_channels*4, noise_emb_dim, use_affine),
            ResBlock(base_channels*4, base_channels*4, noise_emb_dim, use_affine),
            ResBlock(base_channels*4, base_channels*4, noise_emb_dim, use_affine)
        )
        self.enc4_1 = CustumSequentiel(
            ResBlock(base_channels*4, base_channels*8, noise_emb_dim, use_affine),
            ResBlock(base_channels*8, base_channels*8, noise_emb_dim, use_affine),
            ResBlock(base_channels*8, base_channels*8, noise_emb_dim, use_affine)
        )
        self.enc4_2 = CustumSequentiel(
            ResBlock(base_channels*8, base_channels*8, noise_emb_dim, use_affine),
            ResBlock(base_channels*8, base_channels*8, noise_emb_dim, use_affine),
            ResBlock(base_channels*8, base_channels*8, noise_emb_dim, use_affine)
        )
        
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, noise_emb):
        x1 = self.enc1(x, noise_emb)
        x2 = self.enc2(self.pooling(x1), noise_emb)
        x3 = self.enc3(self.pooling(x2), noise_emb)
        x4 = self.enc4_2(self.enc4_1(self.pooling(x3), noise_emb), noise_emb)
        return x1, x2, x3, x4
    
class Bottleneck(nn.Module):
    def __init__(self, input_channels, noise_emb_dim, use_affine = False):
        super(Bottleneck, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = CustumSequentiel(
            ResBlock(input_channels, input_channels*2, noise_emb_dim, use_affine),
            ResBlock(input_channels*2, input_channels*2, noise_emb_dim, use_affine),
            ResBlock(input_channels*2, input_channels*2, noise_emb_dim, use_affine)
        )
        self.layer2 = CustumSequentiel(
            ResBlock(input_channels*2, input_channels*2, noise_emb_dim, use_affine),
            ResBlock(input_channels*2, input_channels*2, noise_emb_dim, use_affine),
            ResBlock(input_channels*2, input_channels*2, noise_emb_dim, use_affine)
        )
        self.up4 = nn.ConvTranspose2d(input_channels*2, input_channels, kernel_size=2, stride=2)

    def forward(self, x, noise_emb):
        x = self.pooling(x)
        x = self.layer1(x, noise_emb)
        x = self.layer2(x, noise_emb)
        x = self.up4(x)
        return x
    
class Unet_decoder(nn.Module):
    def __init__(self, noise_emb_dim, output_channels, base_channels = 64, use_affine = False):
        super(Unet_decoder, self).__init__()
        self.dec1_1 = CustumSequentiel(
            ResBlock(base_channels*16, base_channels*8, noise_emb_dim, use_affine),
            ResBlock(base_channels*8, base_channels*8, noise_emb_dim, use_affine),
            ResBlock(base_channels*8, base_channels*8, noise_emb_dim, use_affine),
        )
        self.dec1_2 = CustumSequentiel(
            ResBlock(base_channels*8, base_channels*8, noise_emb_dim, use_affine),
            ResBlock(base_channels*8, base_channels*8, noise_emb_dim, use_affine),
            ResBlock(base_channels*8, base_channels*8, noise_emb_dim, use_affine),
        )
        self.up1 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec2 = CustumSequentiel(
            ResBlock(base_channels*8, base_channels*4, noise_emb_dim, use_affine),
            ResBlock(base_channels*4, base_channels*4, noise_emb_dim, use_affine),
            ResBlock(base_channels*4, base_channels*4, noise_emb_dim, use_affine),
        )
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec3 = CustumSequentiel(
            ResBlock(base_channels*4, base_channels*2, noise_emb_dim, use_affine),
            ResBlock(base_channels*2, base_channels*2, noise_emb_dim, use_affine),
            ResBlock(base_channels*2, base_channels*2, noise_emb_dim, use_affine),
        )
        self.up3 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec4 = CustumSequentiel(
            ResBlock(base_channels*2, base_channels, noise_emb_dim, use_affine),
            ResBlock(base_channels, base_channels, noise_emb_dim, use_affine),
            ResBlock(base_channels, base_channels, noise_emb_dim, use_affine),
        )
        self.last = CustumSequentiel(
            ResBlock(base_channels, output_channels, noise_emb_dim, use_affine),
            ResBlock(output_channels, output_channels, noise_emb_dim, use_affine),
            ResBlock(output_channels, output_channels, noise_emb_dim, use_affine),
        )

    def forward(self, x, noise_emb, x1, x2, x3, x4):
        x = self.up1(self.dec1_2(self.dec1_1(torch.cat([x4,x], dim=1), noise_emb), noise_emb))

        x = self.up2(self.dec2(torch.cat([x3,x], dim=1), noise_emb))

        x = self.up3(self.dec3(torch.cat([x2,x], dim=1), noise_emb))

        
        x = self.dec4(torch.cat([x1,x], dim=1), noise_emb)

        x = self.last(x, noise_emb)

        return x
    

class Unet(nn.Module):
    def __init__(self, noise_emb_dim, input_channels, output_channels, base_channels = 64, use_affine = False):
        super(Unet, self).__init__()
        self.pos_encoder = PositionalEncoding(noise_emb_dim)
        self.encoder = Unet_encoder(input_channels, noise_emb_dim, base_channels=base_channels, use_affine=use_affine)
        self.bottleneck = Bottleneck(base_channels*8, noise_emb_dim, use_affine=use_affine)
        self.decoder = Unet_decoder(noise_emb_dim, output_channels, base_channels=base_channels, use_affine=use_affine)

    def forward(self, x, t):
        noise_emb = self.pos_encoder(t)

        x1, x2, x3, x4 = self.encoder(x, noise_emb)

        x = self.bottleneck(x4, noise_emb)

        x = self.decoder(x, noise_emb, x1, x2, x3, x4)

        return x

    



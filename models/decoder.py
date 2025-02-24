
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(
                self,
                in_dim,
                h_dim,
                n_res_layers,
                res_h_dim,
                out_dim=3,
                cov_configs=None
                ):
        super(Decoder, self).__init__()
        kernel_configs = [4, 4, 3]
        stride_configs = [2, 2, 1]
        padding_configs = [1, 1, 1]
        
        kernel_configs.reverse()
        stride_configs.reverse()
        padding_configs.reverse()

        res_kernel = 3
        res_stride = 1
        res_padding = 1



        if cov_configs is not None:
            kernel_configs = cov_configs['kernel'].copy()
            stride_configs = cov_configs['stride'].copy()
            padding_configs = cov_configs['padding'].copy()
            
            kernel_configs.reverse()
            stride_configs.reverse()
            padding_configs.reverse()

            res_kernel = cov_configs['res_kernel']
            res_stride = cov_configs['res_stride']
            res_padding = cov_configs['res_padding']
            print(f'Using custom kernel configs: {kernel_configs}')

        self.inverse_conv_stack = nn.ModuleList(
            [
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel_configs[0], stride=stride_configs[0], padding=padding_configs[0]),
            
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, res_kernel, res_stride, res_padding),
            ]
        )
        
        if len(kernel_configs) > 3:
            left_kernel = kernel_configs[1:-2]
            left_stride = stride_configs[1:-2]
            left_padding = padding_configs[1:-2]
            print(left_kernel)
            for i in range(len(left_kernel)):
                self.inverse_conv_stack.append(nn.ReLU())
                self.inverse_conv_stack.append(nn.ConvTranspose2d(h_dim, h_dim, kernel_size=kernel_configs[i+3],
                               stride=left_stride[i], padding=left_padding[i]))
        
        
        self.inverse_conv_stack.extend([
            nn.ReLU(), # added compared to original repo
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel_configs[1], stride=stride_configs[1], padding=padding_configs[1]),
            
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, out_dim, kernel_size=kernel_configs[2],
                               stride=stride_configs[2], padding=padding_configs[2])
            ]
        )
        

    def forward(self, x):
        for layer in self.inverse_conv_stack:
            x = layer(x)
            # print(x.shape)
        return x

    # def get_cov_configs(self, para_configs):
    #     if isinstance(para_configs, list):
    #         config = para_configs
    #     else:
    #         print(f'Using default conv configs: {para_configs}, It is better to be a list')
    #         config = [para_configs, para_configs, para_configs - 1 if para_configs - 1 >= 0 else para_configs]
    #     return config

if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 3, 64, 3)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)

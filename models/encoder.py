import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, cov_configs=None):
        super(Encoder, self).__init__()
        kernel_configs = [4, 4, 3]
        stride_configs = [2, 2, 1]
        padding_configs = [1, 1, 1]
        
        if cov_configs is not None:
            kernel_configs = cov_configs['kernel']
            stride_configs = cov_configs['stride']
            padding_configs = cov_configs['padding']
            res_kernel = cov_configs['res_kernel']
            res_stride = cov_configs['res_stride']
            res_padding = cov_configs['res_padding']
            print(f'Using custom conv configs: {cov_configs}')

        self.conv_stack = nn.ModuleList([   
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel_configs[0],
                      stride=stride_configs[0], padding=padding_configs[0]),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel_configs[1],
                      stride=stride_configs[1], padding=padding_configs[1]),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel_configs[2],
                      stride=stride_configs[2], padding=padding_configs[2]),
        ])

        if len(kernel_configs) > 3:
            left_kernel = kernel_configs[3:]
            left_stride = stride_configs[3:]
            left_padding = padding_configs[3:]
            for i in range(len(left_kernel)):
                self.conv_stack.append(nn.ReLU())
                self.conv_stack.append(
                    nn.Conv2d(h_dim, h_dim, kernel_size=left_kernel[i],
                          stride=left_stride[i], padding=left_padding[i])
                )
                
        self.conv_stack.append(
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers, res_kernel, res_stride, res_padding), # 7 * 7 for mnist 
        )



    def forward(self, x):
        for layer in self.conv_stack:
            x = layer(x)
        return x

#    def get_cov_configs(self, para_configs):
#        if isinstance(para_configs, list):
#            config = para_configs
#        else:
#            print(f'Using default conv configs: {para_configs}, It is better to be a list')
#            config = [para_configs, para_configs, para_configs - 1 if para_configs - 1 >= 0 else para_configs]
#        return config


# if __name__ == "__main__":
    # random data
#x = np.random.random_sample((256, 3, 32, 32))
#x = torch.tensor(x).float()

    # test encoder
#encoder = Encoder(3, 128, 2, 32)
#encoder_out = encoder(x)
#print('Encoder out shape:', encoder_out.shape)
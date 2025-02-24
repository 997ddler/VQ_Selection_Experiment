from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer_EMA, VectorQuantizer
from torchinfo import summary
import torch.nn as nn

class VQVAE(nn.Module):
    def __init__(self, vq_params, backbone_configs, use_vq=True, init_size=0, use_compressed=False, projection_dim=32):
        super(VQVAE, self).__init__()
        self.encoder, self.decoder = self.get_backbone(
            backbone_configs=backbone_configs,
            embedding_dim=vq_params['embedding_dim']
        )

        summary(self.encoder, input_size=(128, 3, 32, 32))
        

        self.use_vq = use_vq

        
        if 'Encoder_configs' in backbone_configs:
            enc_configs = backbone_configs['Encoder_configs']
            h_dim = enc_configs['encoder_h_dim'] if 'encoder_h_dim' in enc_configs else backbone_configs['h_dim']
        else:
            h_dim = backbone_configs['h_dim']
            print('change pre conv size')


        # encode image into continuous latent space
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, vq_params['embedding_dim'], kernel_size=1, stride=1)
        
        
        self.use_compressed = use_compressed
        decoder_h_dim = vq_params['embedding_dim']
        if use_compressed:
            print(f'Use compressed to project the latent space to lower dimension{projection_dim}')
            self.projection_dim = projection_dim
            self.linear_project_lower = nn.Linear(vq_params['embedding_dim'], projection_dim)
            self.linear_project_upper = nn.Linear(projection_dim, vq_params['embedding_dim'])
            vq_params['embedding_dim'] = projection_dim
        
        summary(self.decoder, input_size=(128, decoder_h_dim, 8, 8))
        
        if self.use_vq:
            self.quantizer = VectorQuantizer_EMA(**vq_params, init_size=init_size)



    def set_ema_decay(self, ema_decay):
        self.quantizer.set_ema_decay(ema_decay)

    def initialize_codebook(self, inputs):
        self.quantizer.initialize_codebook(inputs)


    def get_backbone(self, backbone_configs, embedding_dim):
        cov_configs = None if 'cov_configs' not in backbone_configs else backbone_configs['cov_configs']
        if 'Encoder_configs' in backbone_configs:
            enc_configs = backbone_configs['Encoder_configs']
            h_dim = enc_configs['encoder_h_dim'] if 'encoder_h_dim' in enc_configs else backbone_configs['h_dim']
            encoder_layer = enc_configs['encoder_layer'] if 'encoder_layer' in enc_configs else backbone_configs['n_res_layers']
            res_h_dim = enc_configs['encoder_res_h_dim'] if 'encoder_res_h_dim' in enc_configs else backbone_configs['res_h_dim']

            encoder = Encoder(
                    backbone_configs['in_dim'], 
                    h_dim, 
                    encoder_layer, 
                    res_h_dim,
                    cov_configs
                )
            print('change size of Encoder')
        else:
            encoder = Encoder(
                                backbone_configs['in_dim'], 
                                backbone_configs['h_dim'], 
                                backbone_configs['n_res_layers'], 
                                backbone_configs['res_h_dim'],
                                cov_configs
                            )
        decoder = Decoder(
                            embedding_dim, 
                            backbone_configs['h_dim'], 
                            backbone_configs['n_res_layers'], 
                            backbone_configs['res_h_dim'],
                            backbone_configs['in_dim'],
                            cov_configs
                        )
        return encoder, decoder

    def encode(self, x):

        # if self.segment_input:  
        #    z_e = []
        #    for i in range(self.segment_part_num):
        #        for j in range(self.segment_part_num):
        #            z_e.append(
        #                    self.encoder[i * self.segment_part_num + j](
        #                        x[:, :, i * 32 // self.segment_part_num : (i + 1) * 32 // self.segment_part_num, 
        #                        j * 32 // self.segment_part_num : (j + 1) * 32 // self.segment_part_num]
        #                    )
        #                )
        #    z_e = torch.cat(z_e, dim=1)
        #else:
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        if self.use_compressed:
            z_e = z_e.permute(0, 2, 3, 1)
            z_e = self.linear_project_lower(z_e)
            z_e = z_e.permute(0, 3, 1, 2)

        return z_e

    def forward(self, x):
        z_e = self.encoder(x)
        
        
        z_e = self.pre_quantization_conv(z_e)
        
        
        if self.use_compressed:
            z_e = z_e.permute(0, 2, 3, 1)
            z_e = self.linear_project_lower(z_e)
            z_e = z_e.permute(0, 3, 1, 2)
        
        if self.use_vq:
            loss, z_q, perplexity, encodings, indices = self.quantizer(z_e)
        else:
            loss, z_q, perplexity, encodings, indices = 0, z_e, 0, None, None
        
        
            
        if self.use_compressed:
            z_q = z_q.permute(0, 2, 3, 1)
            z_q = self.linear_project_upper(z_q)
            z_q = z_q.permute(0, 3, 1, 2)        
            
        x_recon = self.decoder(z_q)


            
        return x_recon, loss, indices, perplexity, z_e.detach(), encodings
    
    def get_codebook(self):
        if self.use_vq:
            return self.quantizer.get_codebook()
        else:
            return None
        
    def get_ori_decay(self):
        if self.use_vq:
            return self.quantizer.get_ori_decay()
        else:
            return None

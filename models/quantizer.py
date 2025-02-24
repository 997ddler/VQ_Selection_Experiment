import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer_EMA(nn.Module):
    def __init__(self, embedding_dim, n_embeddings, decay, eps=1e-5, beta=0.25,kmean_init=False, init_size=0):
        super(VectorQuantizer_EMA, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.decay = decay
        self.eps = eps
        self.beta = beta
        
        self.initialized = False
        self.init_size = init_size


        self.kmean_init = kmean_init

        self.codebook = nn.Parameter(torch.randn(n_embeddings, embedding_dim), requires_grad=False)
        self.register_buffer('cluster_size', torch.ones(n_embeddings))
        self.register_buffer('embed_avg', self.codebook.clone())

   # initialize codebook with first batch data
    def _initialize_codebook(self, x, kmean_init=False):
        with torch.no_grad():
            # flatten the input
            flat_x = x.reshape(-1, self.embedding_dim)
            #if self.init_size > 0:
            #    print('data size: {}'.format(flat_x.shape[0]))
            #    flat_x = flat_x[:self.init_size]
            print('initialization shape {}'.format(flat_x.shape))
            if kmean_init:
                # use kmean to initialize the codebook
                print('kmeans initialization!')
                kmeans = KMeans(n_clusters=self.n_embeddings, random_state=0).fit(flat_x.cpu().numpy())
                centers = torch.tensor(
                    kmeans.cluster_centers_,
                    dtype=self.codebook.dtype,
                )
                self.codebook.data.copy_(centers)
                self.embed_avg.data.copy_(self.codebook.data)

            else:
                # randomly sampling from encoder output and update the codebook
                #indices = torch.randperm(flat_x.shape[0])[:self.n_embeddings]
                #self.codebook.data.copy_(flat_x[indices])
                #self.embed_avg.data.copy_(self.codebook.data)
                print('Still use random initialization of codebook')
        self.initialized = True
        print("Codebook initialized with first batch data.")
        
    def get_codebook(self):
        return self.codebook.data


    def initialize_codebook(self, z_e):
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_flatten = z_e.view(-1, self.embedding_dim)

        if z_flatten.shape[0] > self.init_size:
            indices = torch.randperm(z_flatten.shape[0])[:self.init_size]
            z_flatten = z_flatten[indices]
            print('random select {} data from {} data to initialize codebook'.format(self.init_size, z_flatten.shape[0]))
        elif z_flatten.shape[0] < self.init_size:
            z_flatten = z_flatten.repeat(self.init_size // z_flatten.shape[0] + 1, 1)
            z_flatten = z_flatten[:self.init_size]
            print('repeat {} times to initialize codebook'.format(self.init_size // z_flatten.shape[0] + 1))
        print('manual init')
        self._initialize_codebook(z_flatten, kmean_init=self.kmean_init)

    def forward(self, z_e):
        # permute the input to (B, C, H, W) -> (B, H, W, C)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        shape = z_e.shape
        
        # if high dimension, flatten the input like (B, H, W, C) -> (B*H*W, C)
        z_flatten = z_e.view(-1, self.embedding_dim)
        # first foward in training, initialize codebook
        if self.training and not self.initialized:
            self._initialize_codebook(z_flatten, kmean_init=self.kmean_init)
            self.initialized = True

        # calculate the distance between the input and the codebook
        dis = z_flatten.pow(2).sum(dim=1, keepdim=True) - 2 * torch.einsum('ik,jk->ij', z_flatten, self.codebook) + self.codebook.pow(2).sum(dim=1, keepdim=True).T
    
        # find the nearest neighbor's index
        # Add unsqueeze(1) to match the shape of indices
        indices = torch.argmin(dis, dim=1)

        # get the nearest neighbor's feature
        z_q = F.embedding(indices, self.codebook)
        

        # get the one hot encoding # (B, ) -> (B, n_embeddings) 
        encodings = F.one_hot(indices, num_classes=self.n_embeddings) #

        # if training, update the codebook
        if self.training:
            with torch.no_grad():
                # update the cluster size and the embed_avg
                
                # cluster_size = cluster_size * decay + encodings.sum(0) * (1 - decay)
                self.cluster_size.data.mul_(self.decay)
                self.cluster_size.data.add_(encodings.sum(0), alpha=1 - self.decay)

                # embed_avg = embed_avg * decay + encodings.transpose(0, 1) X  z_flatten * (1 - decay)
                self.embed_avg.data.mul_(self.decay)
                new_sum_embed = encodings.transpose(0, 1).type(z_flatten.dtype) @ z_flatten

                self.embed_avg.data.add_(new_sum_embed, alpha=1 - self.decay)

                # smooth the cluster size
                n = self.cluster_size.sum()
                smoothed_cluster_size = ((self.cluster_size + self.eps) / 
                                         (n + self.n_embeddings * self.eps) ) * n
                

                # compute the codebook information and update it
                codebook_updated = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
                self.codebook.data.copy_(codebook_updated)
    
        # align with another quantizer
        indices = indices.unsqueeze(1)
    
        # perplexity
        e_mean = torch.mean(encodings.float(), dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
    

    
        # reshape the output to the original shape
        z_q = z_q.view(shape)
        
        # loss
        loss = self.beta * F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()


        # reshape the output to the original shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, encodings, indices

    def set_ema_decay(self, ema_decay):
        self.decay = ema_decay

    def get_ori_decay(self):
        return self.decay

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, embedding_dim, n_embeddings, beta, eps=1e-5, kmean_init=True):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.beta = beta
        self.eps = eps
        self.kmean_init = kmean_init
        
        self.embedding = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embeddings, 1.0 / self.n_embeddings)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_embeddings).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

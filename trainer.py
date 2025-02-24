from pickletools import optimize
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
from visualize_util import Visualize_Util
from models.VQ_VAE import VQVAE
from tqdm import tqdm
from torchvision.utils import make_grid


class Resnet_Trainer:
    def __init__(self, device, model_configs, train_configs):
        self.device = device
        self.dataset = train_configs['dataset']

        self.use_vq = model_configs['use_vq']
        
        self.use_compressed = False
        self.projection_dim = 32
        if 'projection_dim' in model_configs:
            self.use_compressed = True
            self.projection_dim = model_configs['projection_dim']
            
        if 'init_size' in train_configs:
            self.init_size = train_configs['init_size']
        else:
            self.init_size = None
        self.model_configs = model_configs
        self.train_config = train_configs
        
        if 'init_batch' in train_configs:
            self.init_batch = train_configs['init_batch']
        else:
            self.init_batch = 0
        
        self.model = self.get_model()
        self.optimizer = self.get_optimizer(train_configs['optimizer'], self.model)
        self.criterion = self.get_criterion(train_configs['criterion'])
        
        self.fine_tune = False
        if 'fine_tune_configs' in train_configs:
            print('Use fine tune')
            self.fine_tune = True
            self.fine_tune_configs = train_configs['fine_tune_configs']
            self.warmup_epoch = self.fine_tune_configs['warmup_epoch']
            self.lr_scheduler = self.get_lr_scheduler(
                                                        self.fine_tune_configs['schedule_method'], 
                                                        self.optimizer, 
                                                        self.fine_tune_configs['schedule_configs']
                                                    )
            
            self.ema_decay_end = self.model.get_ori_decay()
            self.ema_decay_start = self.fine_tune_configs['ema_finetune']
            self.model.set_ema_decay(self.fine_tune_configs['ema_finetune'])


        self.iter = 0
        self.epoch = 0
        
        self.save_dir = train_configs['save_dir']
        
        self.visualize_util = Visualize_Util(log_dir=train_configs['log_dir'])

    def linear_warmup(self, epoch):
        if epoch < self.warmup_epoch:
            return epoch / self.warmup_epoch
        else:
            return 1.0



    def get_lr_scheduler(self, scheduler_name, optimizer, configs):
        if scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['lr_step_size'], gamma=configs['lr_gamma'])
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs['T_max']) # have some problem
        elif scheduler_name == 'linear':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.linear_warmup)
        elif scheduler_name == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs['lr_exp_gamma'])
        else:
            raise ValueError("Invalid scheduler name")
        return scheduler


    def get_criterion(self, criterion_name):
        if criterion_name == 'mse':
            criterion = nn.MSELoss()
        elif criterion_name == 'l1':
            criterion = nn.L1Loss()
        else:
            raise ValueError("Invalid criterion name")
        return criterion

    def get_model(self):
        model = VQVAE(
                    self.model_configs['vq_params'],
                    self.model_configs['backbone_configs'],
                    self.use_vq, 
                    self.init_size, 
                    self.use_compressed, 
                    self.projection_dim
                )
        if self.train_config['pretrain']:
            print('loading pretrain model...')
            model.load_state_dict(torch.load(self.train_config['pretrain_model_path'], map_location='cpu', weights_only=True))
        elif 'partial_pretrain' in self.train_config and self.train_config['partial_pretrain']:
            print('loading partial pretrain model...')
            pretrain_model = torch.load(self.train_config['pretrain_model_path'], map_location='cpu', weights_only=True)
            encoder_dict = {k: v for k, v in pretrain_model.items() if k.startswith('encoder.')}
            pre_quantize_dict = {k: v for k, v in pretrain_model.items() if k.startswith('pre_quantization_conv.')}
            print(encoder_dict.keys())
            print(pre_quantize_dict.keys())
            model.load_state_dict(encoder_dict, strict=False)
            model.pre_quantization_conv.load_state_dict(pre_quantize_dict, strict=False)
        model = model.to(self.device)
        return model


    def get_optimizer(self, optimizer_name, model):
        if optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=self.train_config['lr'], weight_decay=self.train_config['weight_decay'])
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=self.train_config['lr'], amsgrad=True)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=self.train_config['lr'])
        else:
            raise ValueError("Invalid optimizer name")
        return optimizer

    def get_dataset(self):
        print('loading dataset...')
        batch_size = self.train_config['batch_size']
        num_workers = self.train_config['num_workers']
        if self.dataset == 'cifar10':
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])
            train_dataset = torchvision.datasets.CIFAR10(root='/data/zwh', train=True, download=False, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(root='/data/zwh', train=False, download=False, transform=transform)

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
            
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        elif self.dataset == 'mnist':
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Resize((32, 32)),
                                            transforms.Normalize((0.5), (0.5))
                                            ])
            train_dataset = torchvision.datasets.MNIST(root='/data3/zwh/mnist', train=True, download=False, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root='/data3/zwh/mnist', train=False, download=False, transform=transform)
            
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            print(len(train_loader))
        else:
            raise ValueError("Invalid dataset name")
        return train_loader, val_loader, test_loader


    def fit(self):
        epochs = self.train_config['max_epoch']
        train_loader, val_loader, test_loader = self.get_dataset()
        bar = tqdm(range(epochs), desc='Epochs')   
        
        if self.init_batch > 0:
            with torch.no_grad():
                inputs = []
                for batch_idx, (input, target) in enumerate(train_loader):
                    if batch_idx >= self.init_batch:
                        break
                    input = input.to(self.device)
                    inputs.append(input)
                inputs = torch.cat(inputs, dim=0)
                z_e = self.model.encode(inputs)
                
                self.model.initialize_codebook(z_e)
                print('Init batch done')


        for epoch in bar:
            train_loss, train_perplexity = self.train(train_loader)
            val_loss, val_perplexity = self.validate(val_loader)

            if self.fine_tune and epoch < self.warmup_epoch:
                self.lr_scheduler.step()
                self.model.set_ema_decay(self.linear_warmup(epoch) * (self.ema_decay_end - self.ema_decay_start) + self.ema_decay_start)
                
            self.visualize_util.log_scalar('Train/Epoch_loss', train_loss, self.epoch)
            self.visualize_util.log_scalar('Train/Epoch_perplexity', train_perplexity, self.epoch)
            self.visualize_util.log_scalar('Val/Epoch_loss', val_loss, self.epoch)
            self.visualize_util.log_scalar('Val/Epoch_perplexity', val_perplexity, self.epoch)
            self.epoch += 1

        test_loss, test_perplexity = self.validate(test_loader)
        print(f'Test Loss: {test_loss}, Test Perplexity: {test_perplexity}')
        
        os.makedirs(self.train_config['save_dir'], exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.train_config['save_dir'], 'model.pth'))
    
    def train(self, train_loader):
        self.model.train()
        cum_loss = 0
        cum_perplexity = 0
        
        for batch_idx, (input, target) in enumerate(train_loader):
            input, target = input.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            recon, quantize_loss, _, perplexity, _, _ = self.model(input)
            recon_loss = self.criterion(recon, input)
            
            loss = recon_loss + quantize_loss
            
            loss.backward()
            self.optimizer.step()
            
            self.iter += 1
            
            self.visualize_util.log_scalar('Train/Iter_loss', loss.item(), self.iter)
            if self.use_vq:
                self.visualize_util.log_scalar('Train/Iter_quantize_loss', quantize_loss.item(), self.iter)
                self.visualize_util.log_scalar('Train/Iter_perplexity', perplexity.item(), self.iter)
                cum_perplexity += perplexity.cpu().item()
                
            cum_loss += recon_loss.cpu().item()
            
            
        return cum_loss / len(train_loader), cum_perplexity / len(train_loader)
    
    
    def validate(self, val_loader):
        self.model.eval()
        
        cum_loss = 0
        cum_perplexity = 0
        
        ori_data = []
        encoder_data = []
        target_data = []
        recon_data = []
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(val_loader):
                input, target = input.to(self.device), target.to(self.device)   
                
                recon, quantize_loss, _, perplexity, encoder_output, _ = self.model(input)
                
                recon_loss = self.criterion(recon, input)
                loss = recon_loss + quantize_loss
                
                B, C, H, W = encoder_output.shape
                # B, C, H, W -> B, H*W, C
                encoder_output = encoder_output.permute(0, 2, 3, 1).view(B, -1, C)
                
                encoder_data.append(encoder_output.cpu().numpy())
                
                ori_data.append(input.cpu().numpy())
                recon_data.append(recon.cpu().numpy())
                target_data.append(target.cpu().numpy())
                
                cum_loss += recon_loss.cpu().item()
                if self.use_vq:
                    cum_perplexity += perplexity.cpu().item()
                
            codebook = self.model.get_codebook()
            if codebook is not None:
                codebook = codebook.cpu().numpy()
    
        ori_data = np.concatenate(ori_data, axis=0)
        recon_data = np.concatenate(recon_data, axis=0)
        target_data = np.concatenate(target_data, axis=0)
        encoder_data = np.concatenate(encoder_data, axis=0)
        
        
        # if self.epoch % 50 == 0:
            # self.visualize_util.visualize_code_distribution(codebook, encoder_data, target_data, self.epoch, 'Codebook Distribution')
        
        images, _ = next(iter(val_loader))
        images = images[ : 64]
        recon_images, _, _, _, _, _ = self.model(images.to(self.device))
        
        recon_images = recon_images.detach()
        
        ori_images = make_grid(images, normalize=True, nrow=8 )
        recon_images = make_grid(recon_images, normalize=True, nrow=8)
        
        ori_images = ori_images.cpu().numpy()
        recon_images = recon_images.cpu().numpy()
        
        if self.epoch % 10 == 0:
            self.visualize_util.plot_recon(ori_images, recon_images, self.epoch)
        
        return cum_loss / len(val_loader), cum_perplexity / len(val_loader)
            
                
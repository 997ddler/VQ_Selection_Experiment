import argparse
import torch
import numpy as np
import random
import yaml
import os
import time
from trainer import Resnet_Trainer

def main(args, trainer, configs):
    trainer.fit()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3072, help="fix seed")
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--config", type=str, default="config.yaml", help="config file")
    args = parser.parse_args()

    yaml_path = args.config
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['seed'] = args.seed
    config['device_id'] = args.device_id
    
    torch.cuda.set_device(args.device_id)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    config['train_configs']['save_dir'] = os.path.join(config['train_configs']['save_dir'], current_time)
    config['train_configs']['log_dir'] = os.path.join(config['train_configs']['log_dir'], current_time)

    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # save  config file
    os.makedirs(config['train_configs']['save_dir'], exist_ok=True)
    with open(os.path.join(config['train_configs']['save_dir'], 'saved_config.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    
    trainer = Resnet_Trainer(device, config['model_configs'], config['train_configs'])
    
    main(args, trainer, config)


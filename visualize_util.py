from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os



class Visualize_Util:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.tensorboard = SummaryWriter(log_dir)
        
    def log_scalar(self, tag, scalar_value, global_step):
        self.tensorboard.add_scalar(tag, scalar_value, global_step)

    def log_image(self, tag, image, global_step, dataformats='HWC'):
        self.tensorboard.add_image(tag, image, global_step, dataformats=dataformats)

    def decompose_data(self, data, compress_dim=2, method='tsne'):
        if method == 'tsne':
            tsne = TSNE(n_components=compress_dim, random_state=0)
            data = tsne.fit_transform(data)
        elif method == 'pca':
            pca = PCA(n_components=compress_dim, random_state=0)
            data = pca.fit_transform(data)
        # elif method == 'umap':
        #    umap = UMAP(n_components=compress_dim, random_state=0)
        #    data = umap.fit_transform(data)
        else:
            raise ValueError("Invalid method")
        return data

    def plot_scatter_target(self, data, target, ax, title):
        marker = ['o', 's', 'D', 'v', 'x', 'p', 'h', '8', '4', '2', '1', '3', '5', '7', '6', '9', '0']
        color = ['blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'olive', 'lime', 'teal', 'navy', 'maroon', 'indigo']
        for i in range(len(np.unique(target))):
            ax.scatter(data[target==i, 0], data[target==i, 1], marker=marker[i], label=f'Class {i}', alpha=0.3, color=color[i])
        ax.set_title(title)
        ax.legend()
        
    def plot_recon(self, ori_images, recon_images, global_step):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 64))
        
        ori_images = np.transpose(ori_images, (1, 2, 0))
        recon_images = np.transpose(recon_images, (1, 2, 0))
        
        ax1.imshow(ori_images)
        ax2.imshow(recon_images)
        
        
        ax1.set_title('Input')
        ax2.set_title('Recon')
        
        ax1.axis('off')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Log image to TensorBoard
        self.log_image('Reconstruction Comparison', image, global_step)
        # print(f'Log image to TensorBoard: {global_step}')
        
        plt.close(fig)
        
        
        
    def visualize_code_distribution(
                                    self,
                                    codebook,
                                    encoder_output,
                                    encoder_target,
                                    global_step,
                                    title,
                                    compress_dim=2,
                                    method='tsne'
                                ):
        # codebook, encoder should be numpy array

        # encoder_output is (B, h * w, C) -> (B * h * w, C) and record h * w

        repeat_num = encoder_output.shape[1]
        encoder_output = encoder_output.reshape(-1, encoder_output.shape[2])
        
        # repeat encoder_target to match encoder_output shape
        encoder_target = encoder_target.repeat(repeat_num)
        
        print(encoder_output.shape)
        # random sample 10000 data from encoder_output
        index = np.random.choice(encoder_output.shape[0], 1000, replace=False)
        encoder_output = encoder_output[index]
        encoder_target = encoder_target[index]
        
        if codebook is not None:
            if codebook.shape[1] == 1:
                return
            data = np.concatenate([codebook, encoder_output], axis=0)
            data = self.decompose_data(data, compress_dim=compress_dim, method=method)
            codebook_decomposed = data[:codebook.shape[0]]
            encoder_output_decomposed = data[codebook.shape[0]:]
        else:
            data = encoder_output
            data = self.decompose_data(data, compress_dim=compress_dim, method=method)
            encoder_output_decomposed = data
        
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        self.plot_scatter_target(encoder_output_decomposed, encoder_target, ax1, 'Codebook')
        if codebook is not None:
            ax1.scatter(codebook_decomposed[:, 0], codebook_decomposed[:, 1], marker='*', label='Codebook', color='red', alpha=0.3)
        ax1.legend()
        
        # conver plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        #os.makedirs('./ae_code_distribution', exist_ok=True)
        
        #plt.savefig(f'./ae_code_distribution/ae_code_distribution_{global_step}.png')
        
        # Log image to TensorBoard
        self.log_image(title, image, global_step)
        plt.close(fig)
        

        
        
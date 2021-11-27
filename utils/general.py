import os
import matplotlib.pyplot as plt
import numpy as np


def plot_result(gen_image, num_epoch, num_item=0, save=False, save_dir='CelebA_DCGAN_results/', show=False, fig_size=(5, 5)):
    n_rows = np.sqrt(gen_image.shape[0]).astype(np.int32)
    n_cols = np.sqrt(gen_image.shape[0]).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.set_adjustable('box')
        # Scale to 0-255
        img = img.transpose(1, 2, 0).astype(np.uint8)
        # ax.imshow(img.cpu().data.view(image_size, image_size, 3).numpy(), cmap=None, aspect='equal')
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch+1)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = os.path.join(save_dir, 'epoch_{:d}-item_{:d}'.format(num_epoch, num_item) + '.png')
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


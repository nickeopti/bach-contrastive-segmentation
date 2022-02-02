import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plotable(image):
    return image.squeeze()


def plot_selected_crops(data, path=None):
    if len(data) <= 1:
        return

    n_channels = len(data[1][1])
    fig, axs = plt.subplots(1 + 2*n_channels, len(data), figsize=(len(data), 1 + 2*n_channels))
    for i, (image, attention_map, attended_image, selections) in enumerate(data):
        axs[0][i].imshow(plotable(image))
        for j in range(n_channels):
            axs[1+j][i].imshow(plotable(attention_map[j]))
            axs[1+n_channels+j][i].imshow(plotable(attended_image[j]))

        for j in range(1 + 2*n_channels):
            for n, (_, row, col, size) in enumerate(selections):
                axs[j][i].add_patch(Rectangle((col, row), size, size, linewidth=(0.5 if n < 2 else 0.1), edgecolor=('g' if n < 2 else 'r'), facecolor='none'))
                axs[j][i].axis('off')

    if path is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=300)
    plt.close()

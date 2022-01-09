import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plotable(image):
    return image.squeeze()


def plot_selected_crops(data, path=None):
    fig, axs = plt.subplots(3, len(data), figsize=(len(data), 3))
    for i, (image, attention_map, selections) in enumerate(data):
        axs[0][i].imshow(plotable(image))
        axs[1][i].imshow(plotable(attention_map))
        axs[2][i].imshow(plotable(image * attention_map))

        for j in range(3):
            for n, (row, col, size) in enumerate(selections):
                axs[j][i].add_patch(Rectangle((col, row), size, size, linewidth=(0.5 if n < 2 else 0.1), edgecolor=('g' if n < 2 else 'r'), facecolor='none'))
                axs[j][i].axis('off')

    if path is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=300)
    plt.close()

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


COLOURS = ['green', 'red', 'darkorange', 'black', 'white']

def class_index_from_n(n, class_indices):
    return next(i for i, (from_index, to_index) in enumerate(class_indices) if from_index <= n < to_index)


def plotable(image):
    return image.squeeze()


def plot_selected_crops(data, path=None):
    if len(data) <= 1:
        return

    n = min(len(data), 10)
    n_channels = len(data[1][1])
    fig, axs = plt.subplots(
        1 + 2 * n_channels, n, figsize=(n, 1 + 2 * n_channels)
    )
    for i, (image, attention_map, attended_image, regions, class_indices) in enumerate(data[:n]):
        axs[0][i].imshow(plotable(image))
        for j in range(n_channels):
            axs[1 + j][i].imshow(plotable(attention_map[j]))
            axs[1 + n_channels + j][i].imshow(plotable(attended_image[j]))

        for j in range(1 + 2 * n_channels):
            for n, region in enumerate(regions):
                ci = class_index_from_n(n, class_indices)
                if j == 0 or (j-1) % n_channels == ci:
                    axs[j][i].add_patch(
                        Rectangle(
                            (region.col, region.row),
                            region.size,
                            region.size,
                            # linewidth=(0.5 if n < class_indices[0][1] else 0.1),
                            linewidth=0.5,
                            edgecolor=COLOURS[ci],
                            facecolor="none",
                        )
                    )
                axs[j][i].axis("off")

    if path is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=300)
    plt.close()

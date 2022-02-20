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
    for i, (image, attention_map, attended_image, positive_regions, negative_regions) in enumerate(data[:n]):
        axs[0][i].imshow(plotable(image))
        for j in range(n_channels):
            axs[1 + j][i].imshow(plotable(attention_map[j]))
            axs[1 + n_channels + j][i].imshow(plotable(attended_image[j]))

        for j in range(1, 1 + 2 * n_channels):
            for region in positive_regions[(j-1) % n_channels]:
                axs[j][i].add_patch(
                    Rectangle(
                        (region.col, region.row),
                        region.size,
                        region.size,
                        linewidth=0.5,
                        edgecolor='green',
                        facecolor="none",
                    )
                )
            for region in negative_regions[(j-1) % n_channels]:
                axs[j][i].add_patch(
                    Rectangle(
                        (region.col, region.row),
                        region.size,
                        region.size,
                        linewidth=0.5,
                        edgecolor='red',
                        facecolor="none",
                    )
                )
        
        for j in range(1 + 2 * n_channels):
            axs[j][i].axis("off")

    if path is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=300)
    plt.close()


def plot_histograms(data, path=None):
    if len(data) <= 1:
        return

    n = min(len(data), 10)
    n_channels = len(data[1][1])
    fig, axs = plt.subplots(
        1 + 2 * n_channels, n, figsize=(n, 1 + 2 * n_channels)
    )
    for i, (image, attended_image, attentions) in enumerate(data[:n]):
        axs[0][i].imshow(plotable(image))
        axs[0][i].axis("off")
        for j, attended_channel in enumerate(attended_image, start=1):
            axs[j][i].imshow(plotable(attended_channel))
            axs[j][i].axis("off")
        for j, channel_attentions in enumerate(attentions, start=1+n_channels):
            axs[j][i].hist(channel_attentions, bins=20)
            axs[j][i].axis("off")

    if path is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=300)
    plt.close()

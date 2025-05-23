import matplotlib.pyplot as plt
from skimage.transform import resize


def plot_pr_curve(precisions, recalls):
    """绘制PR曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.', label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.show()


def display_results(query_img, top_images, top_similarities):
    """显示查询结果"""
    query_resized = resize(query_img, (200, 200))
    resized_images = [resize(img, (200, 200)) for img in top_images]

    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes[0, 0].imshow(query_resized)
    axes[0, 0].set_title("Query Image")
    axes[0, 0].axis('off')

    for col in range(1, 5):
        axes[0, col].axis('off')

    for i in range(5):
        axes[1, i].imshow(resized_images[i])
        axes[1, i].set_title(f"Rank {i+1}\n{top_similarities[i]:.3f}")
        axes[1, i].axis('off')

    for i in range(5, 10):
        axes[2, i-5].imshow(resized_images[i])
        axes[2, i-5].set_title(f"Rank {i+1}\n{top_similarities[i]:.3f}")
        axes[2, i-5].axis('off')

    plt.tight_layout()
    plt.show()
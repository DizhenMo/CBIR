import numpy as np
import cv2
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 优先使用的中文字体列表
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_image_database(root_path):
    """
    加载图像数据库
    root_path: 数据库根目录，包含多个子文件夹，每个子文件夹代表一个类别
    返回：
    - images: 所有图像列表
    - labels: 对应的标签列表
    - class_names: 类别名称列表
    """
    images = []
    labels = []
    class_names = []

    root_path = Path(root_path)

    # 遍历所有子文件夹
    for class_idx, class_dir in enumerate(sorted(root_path.iterdir())):
        if not class_dir.is_dir():
            continue

        class_names.append(class_dir.name)

        # 遍历当前类别下的所有图像
        for img_path in class_dir.glob('*.[jJ][pP][gG]'):  # 支持jpg和JPG
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    # 统一调整图像大小
                    img = cv2.resize(img, (64, 64))
                    images.append(img)
                    labels.append(class_idx)
            except Exception as e:
                print(f"加载图像出错 {img_path}: {e}")

    return np.array(images), np.array(labels), class_names


def create_ground_truth_matrix(query_labels, db_labels):
    """
    创建ground truth矩阵
    返回一个矩阵，其中每行表示一个查询图像与所有数据库图像的匹配情况
    """
    ground_truth = np.zeros((len(query_labels), len(db_labels)), dtype=int)
    for i, q_label in enumerate(query_labels):
        ground_truth[i] = (db_labels == q_label).astype(int)
    return ground_truth


def extract_hog_features(image):
    """HOG特征提取"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(image, orientations=16,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   visualize=False)
    return features


def whole_image_similarity(img1, img2):
    """全图比对相似度计算"""
    img1 = cv2.resize(img1, (450, 450))
    img2 = cv2.resize(img2, (450, 450))

    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return cosine_similarity(img1.reshape(1, -1), img2.reshape(1, -1))[0][0]


def calculate_precision_recall(predictions, ground_truth, k):
    """计算precision和recall"""
    relevant = sum([1 for pred, true in zip(predictions[:k], ground_truth[:k]) if pred == true])
    total_relevant = sum(ground_truth)

    precision = relevant / k if k > 0 else 0
    recall = relevant / total_relevant if total_relevant > 0 else 0
    return precision, recall


def evaluate_retrieval(query_images, db_images, ground_truth, save_path=None):
    """评估检索性能并生成PR曲线"""
    avg_hog_precisions = np.zeros(len(db_images))
    avg_hog_recalls = np.zeros(len(db_images))
    avg_whole_precisions = np.zeros(len(db_images))
    avg_whole_recalls = np.zeros(len(db_images))

    for i in range(len(query_images)):
        query_img = query_images[i]
        query_feat = extract_hog_features(query_img)

        hog_similarities = []
        whole_img_similarities = []

        for j in range(len(db_images)):
            db_img = db_images[j]

            # HOG特征方法
            db_feat = extract_hog_features(db_img)
            hog_sim = cosine_similarity(query_feat.reshape(1, -1), db_feat.reshape(1, -1))[0][0]
            hog_similarities.append(hog_sim)

            # 全图比对方法
            whole_sim = whole_image_similarity(query_img, db_img)
            whole_img_similarities.append(whole_sim)

        # 计算不同k值下的precision和recall
        k_values = range(1, len(db_images) + 1)
        hog_precisions = []
        hog_recalls = []
        whole_precisions = []
        whole_recalls = []

        for k in k_values:
            # HOG特征方法的评估
            hog_top_k = np.argsort(hog_similarities)[-k:]
            hog_precision, hog_recall = calculate_precision_recall(
                hog_top_k, ground_truth[i], k)
            hog_precisions.append(hog_precision)
            hog_recalls.append(hog_recall)

            # 全图比对方法的评估
            whole_top_k = np.argsort(whole_img_similarities)[-k:]
            whole_precision, whole_recall = calculate_precision_recall(
                whole_top_k, ground_truth[i], k)
            whole_precisions.append(whole_precision)
            whole_recalls.append(whole_recall)

        # 累加每次的结果用于计算平均值
        avg_hog_precisions += np.array(hog_precisions)
        avg_hog_recalls += np.array(hog_recalls)
        avg_whole_precisions += np.array(whole_precisions)
        avg_whole_recalls += np.array(whole_recalls)

        # 绘制单张图像的PR曲线
        plt.figure(figsize=(10, 6))
        plt.plot(hog_recalls, hog_precisions, 'b-', label='HOG特征检索')
        plt.plot(whole_recalls, whole_precisions, 'r--', label='全图比对检索')
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title(f'查询图像 {i + 1} 的PR曲线')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(os.path.join(save_path, f'pr_curve_query_{i + 1}.png'))
        plt.close()

    # 计算平均PR曲线
    avg_hog_precisions /= len(query_images)
    avg_hog_recalls /= len(query_images)
    avg_whole_precisions /= len(query_images)
    avg_whole_recalls /= len(query_images)

    # 绘制平均PR曲线
    plt.figure(figsize=(10, 6))
    plt.plot(avg_hog_recalls, avg_hog_precisions, 'b-', label='HOG特征检索')
    plt.plot(avg_whole_recalls, avg_whole_precisions, 'r--', label='全图比对检索')
    plt.xlabel('平均召回率 (Average Recall)')
    plt.ylabel('平均精确率 (Average Precision)')
    plt.title('所有查询的平均PR曲线')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(os.path.join(save_path, 'average_pr_curve.png'))
    plt.close()

    # 计算平均精确率（AP）
    ap_hog = np.mean(avg_hog_precisions)
    ap_whole = np.mean(avg_whole_precisions)

    # 保存评估结果到文本文件
    if save_path:
        with open(os.path.join(save_path, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
            f.write(f'HOG特征检索平均精确率 (AP): {ap_hog:.4f}\n')
            f.write(f'全图比对检索平均精确率 (AP): {ap_whole:.4f}\n')


def main():
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, r"/dataset/image")  # 数据库图像根目录
    query_path = os.path.join(current_dir, r"/test")  # 查询图像根目录
    results_path = os.path.join(current_dir, r"G:\CS\IR\IR\results")  # 结果保存目录

    # 创建结果保存目录
    os.makedirs(results_path, exist_ok=True)

    print("正在加载数据库图像...")
    db_images, db_labels, db_class_names = load_image_database(db_path)
    print(f"已加载 {len(db_images)} 张数据库图像")

    print("正在加载查询图像...")
    query_images, query_labels, _ = load_image_database(query_path)
    print(f"已加载 {len(query_images)} 张查询图像")

    # 创建ground truth矩阵
    ground_truth = create_ground_truth_matrix(query_labels, db_labels)

    print("正在评估检索性能...")
    evaluate_retrieval(query_images, db_images, ground_truth, results_path)

    print(f"评估完成。结果保存在: {results_path}")


if __name__ == "__main__":
    main()
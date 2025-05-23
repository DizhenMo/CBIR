import cv2
import os
import numpy as np
from PIL import Image
from skimage.feature import hog as skimage_hog
from skimage import transform
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist


# 图像加载函数
def load_images(folder_path):
    images = []
    labels = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    with Image.open(img_path) as img:
                        img_array = np.array(img)
                        images.append(img_array)
                        labels.append(class_folder)
    return images, labels


# 特征提取函数
def featExtra(image, featCate, size=(450, 450)):
    image = resize(image, size, anti_aliasing=True)
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    if featCate == 'HOG':
        return _hog_features(gray)
    elif featCate == 'Haar':
        return _haar_features(gray)
    elif featCate == 'LBP':
        return _lbp_features(gray)
    elif featCate == 'SIFT':
        return _sift_features(image)
    elif featCate == 'SURF':
        return _surf_features(image)
    else:
        raise ValueError(f"不支持的特征类型：{featCate}")


def _hog_features(gray):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        hog_features, _ = skimage_hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=False,
            transform_sqrt=True,
            visualize=True
        )
    return hog_features.reshape(-1, hog_features.shape[-1])


def _haar_features(gray):
    from skimage.transform import wavedec2
    coeffs = wavedec2(gray, 'haar', level=2)
    features = []
    for arr in coeffs:
        if isinstance(arr, np.ndarray):
            block_size = 8
            rows = arr.shape[0] // block_size
            cols = arr.shape[1] // block_size
            for i in range(rows):
                for j in range(cols):
                    block = arr[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    features.append(block.ravel())
    return np.array(features)


def _lbp_features(gray):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    block_size = 8
    features = []
    for i in range(0, lbp.shape[0], block_size):
        for j in range(0, lbp.shape[1], block_size):
            block = lbp[i:i + block_size, j:j + block_size]
            hist, _ = np.histogram(block.ravel(), bins=256, range=(0, 256))
            features.append(hist)
    return np.array(features)


def _sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors if descriptors is not None else np.empty((0, 128))


def _surf_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    surf = cv2.xfeatures2d.SURF_create(4000)
    keypoints, descriptors = surf.detectAndCompute(gray, None)
    return descriptors if descriptors is not None else np.empty((0, 64))


# 特征编码函数（修正后支持TF-IDF）
def desExtra(features, codebook, idf=None, desCate='BoF'):
    if desCate == 'BoF':
        if features.size == 0:
            return np.zeros(len(codebook))
        distances = np.linalg.norm(features[:, np.newaxis] - codebook, axis=2)
        labels = np.argmin(distances, axis=1)
        hist, _ = np.histogram(labels, bins=np.arange(len(codebook) + 1), density=False)
        if idf is not None:
            tf_idf = hist * idf
            return tf_idf / (np.sum(tf_idf) + 1e-8)
        else:
            return hist.astype(float) / (hist.sum() + 1e-8)
    elif desCate == 'VLAD':
        if features.size == 0:
            return np.zeros(len(codebook) * codebook.shape[1])
        K = codebook.shape[0]
        d = codebook.shape[1]
        residuals = np.zeros((K, d))
        distances = dist.cdist(features, codebook, 'euclidean')
        nearest_indices = np.argmin(distances, axis=1)
        for i in range(K):
            mask = (nearest_indices == i)
            if np.sum(mask) > 0:
                residuals[i] = np.sum(features[mask] - codebook[i], axis=0)
        vlad = residuals.flatten()
        vlad /= (np.linalg.norm(vlad) + 1e-8)
        return vlad
    else:
        raise ValueError(f"不支持的编码类型：{desCate}")


# 生成码书（新增IDF计算）
def build_codebook(db_images, featCate='SIFT', n_clusters=100):
    all_features = []
    for img in db_images:
        feats = featExtra(img, featCate)
        if feats.size > 0:
            all_features.append(feats)
    if not all_features:
        raise ValueError("没有足够的特征用于生成码书")
    all_features = np.concatenate(all_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    codebook = kmeans.fit(all_features).cluster_centers_

    doc_freq = np.zeros(n_clusters)
    N = len(db_images)
    for img in db_images:
        feats = featExtra(img, featCate)
        if feats.size == 0:
            continue
        distances = np.linalg.norm(feats[:, np.newaxis] - codebook, axis=2)
        labels = np.argmin(distances, axis=1)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            doc_freq[label] += 1
    idf = np.log(N / (doc_freq + 1))
    return codebook, idf


# 相似度计算
def compute_similarity(query_hist, db_hist):
    return cosine_similarity([query_hist], [db_hist])[0][0]




# 合并后的主函数
if __name__ == "__main__":
    # 加载图像
    db_images, _ = load_images(r"/dataset/image")
    query_images, _ = load_images(r"/test")

    # 测试特征提取（原主函数1的内容）
    print("测试HOG特征提取:")
    for idx, img in enumerate(query_images):
        features_hog = featExtra(img, 'HOG')
        print(f"图像 {idx + 1} 的 HOG 特征维度: {features_hog.shape}")

    print("\n测试LBP特征提取:")
    for idx, img in enumerate(query_images):
        features_lbp = featExtra(img, 'LBP')
        print(f"图像 {idx + 1} 的 LBP 特征维度: {features_lbp.shape}")

    # 主流程（原主函数2的内容）
    feat_type = 'HOG'  # 特征类型
    des_type = 'BoF'  # 编码类型
    n_clusters = 128  # 码书大小

    # 生成码书
    print("生成视觉词汇表...")
    codebook, idf = build_codebook(db_images, featCate=feat_type, n_clusters=n_clusters)

    # 处理查询图像（第一个查询图像）
    query_img = query_images[0]
    query_features = featExtra(query_img, feat_type)
    query_hist = desExtra(query_features, codebook, idf=idf, desCate=des_type)

    # 预处理数据库图像
    print("预处理数据库图像...")
    db_hists = []
    for db_img in db_images:
        db_features = featExtra(db_img, feat_type)
        db_hist = desExtra(db_features, codebook, idf=idf, desCate=des_type)
        db_hists.append(db_hist)

    # 计算相似度并获取前10名
    print("计算相似度...")
    similarities = [compute_similarity(query_hist, db_hist) for db_hist in db_hists]
    top_k_indices = np.argsort(similarities)[::-1][:10]
    top_similarities = [similarities[i] for i in top_k_indices]
    top_images = [db_images[i] for i in top_k_indices]


    # 显示结果
    def display_results(query_img, top_images, top_similarities):
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
            axes[1, i].set_title(f"Rank {i + 1}\n{top_similarities[i]:.3f}")
            axes[1, i].axis('off')

        for i in range(5, 10):
            axes[2, i - 5].imshow(resized_images[i])
            axes[2, i - 5].set_title(f"Rank {i + 1}\n{top_similarities[i]:.3f}")
            axes[2, i - 5].axis('off')

        plt.tight_layout()
        plt.show()


    print("显示结果...")
    display_results(query_img, top_images, top_similarities)
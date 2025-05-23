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
import pickle

# 保存和加载中间结果
def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_cache_filename(prefix, feat_type, n_clusters, des_type=None):
    if des_type:
        return f"{prefix}_{feat_type}_{n_clusters}_{des_type}.pkl"
    else:
        return f"{prefix}_{feat_type}_{n_clusters}.pkl"




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
            orientations=9, # 方向数
            pixels_per_cell=(8, 8), # 每个单元格的像素数
            cells_per_block=(2, 2), # 每个块的单元格数
            feature_vector=False,
            transform_sqrt=True,
            visualize=True
        )
    return hog_features.reshape(-1, hog_features.shape[-1]) # 展平特征

# Haar小波特征
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
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform') # 计算LBP
    block_size = 8
    features = []
    for i in range(0, lbp.shape[0], block_size):
        for j in range(0, lbp.shape[1], block_size):
            block = lbp[i:i + block_size, j:j + block_size]
            hist, _ = np.histogram(block.ravel(), bins=256, range=(0, 256)) # 计算直方图
            features.append(hist)
    return np.array(features)


def _sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create() # 创建SIFT对象
    keypoints, descriptors = sift.detectAndCompute(gray, None) # 检测关键点并计算描述符
    return descriptors if descriptors is not None else np.empty((0, 128))


def _surf_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    surf = cv2.xfeatures2d.SURF_create(4000) # 创建SURF对象
    keypoints, descriptors = surf.detectAndCompute(gray, None) # 检测关键点并计算描述符
    return descriptors if descriptors is not None else np.empty((0, 64))


# 特征编码函数（修正后支持TF-IDF）
def desExtra(features, codebook, idf=None, desCate='BoF'):
    if desCate == 'BoF':
        if features.size == 0:
            return np.zeros(len(codebook))
        distances = np.linalg.norm(features[:, np.newaxis] - codebook, axis=2) # 计算特征与码书的距离
        labels = np.argmin(distances, axis=1) # 找到最近的码字
        hist, _ = np.histogram(labels, bins=np.arange(len(codebook) + 1), density=False) # 统计直方图
        if idf is not None: # 如果提供IDF，则应用TF-IDF加权
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
        feats = featExtra(img, featCate)  # 提取特征
        if feats.size > 0:
            all_features.append(feats)
    if not all_features:
        raise ValueError("没有足够的特征用于生成码书")
    all_features = np.concatenate(all_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42) # 使用KMeans聚类生成码书
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
            doc_freq[label] += 1 # 统计文档频率
    idf = np.log(N / (doc_freq + 1)) # 计算IDF
    return codebook, idf


# 相似度计算（余弦相似度）
def compute_similarity(query_hist, db_hist):
    return cosine_similarity([query_hist], [db_hist])[0][0]



# 合并后的主函数（完整版）
if __name__ == "__main__":
    # 参数定义
    feat_type = 'HOG'  # 可选项：HOG, LBP, SIFT, SURF, Haar
    des_type = 'BoF'  # 可选项：BoF, VLAD
    n_clusters = 128  # 码书大小（建议值：64-256）
    db_folder = r"G:\CS\IR\IR\dataset\image"  # 数据库图像路径
    query_folder = r"G:\CS\IR\IR\test"  # 查询图像路径

    # --- 加载图像 ---
    if not os.path.exists(db_folder) or not os.path.exists(query_folder):
        raise FileNotFoundError("图像路径不存在，请检查路径配置")

    print("加载图像数据...")
    db_images, _ = load_images(db_folder)
    query_images, _ = load_images(query_folder)
    print(f"数据库图像数量：{len(db_images)}，查询图像数量：{len(query_images)}")

    # --- 码书和IDF缓存逻辑 ---
    codebook_file = get_cache_filename("codebook", feat_type, n_clusters)
    idf_file = get_cache_filename("idf", feat_type, n_clusters)

    if os.path.exists(codebook_file) and os.path.exists(idf_file):
        print("加载缓存的码书和IDF...")
        codebook = load_data(codebook_file)
        idf = load_data(idf_file)
    else:
        print("生成视觉词汇表（可能需要较长时间）...")
        codebook, idf = build_codebook(db_images, featCate=feat_type, n_clusters=n_clusters)
        save_data(codebook_file, codebook)
        save_data(idf_file, idf)
    print(f"码书维度：{codebook.shape}，IDF维度：{idf.shape}")

    # --- 数据库特征直方图缓存逻辑 ---
    db_hists_file = get_cache_filename("db_hists", feat_type, n_clusters, des_type)
    if os.path.exists(db_hists_file):
        print("加载缓存的数据库特征直方图...")
        db_hists = load_data(db_hists_file)
    else:
        print("预处理数据库图像（可能需要较长时间）...")
        db_hists = []
        for idx, db_img in enumerate(db_images):
            db_features = featExtra(db_img, feat_type)
            db_hist = desExtra(db_features, codebook, idf=idf, desCate=des_type)
            db_hists.append(db_hist)
            if (idx + 1) % 50 == 0:
                print(f"已处理 {idx + 1}/{len(db_images)} 张数据库图像")
        save_data(db_hists_file, db_hists)
    print(f"数据库特征直方图数量：{len(db_hists)}")

    # --- 处理查询图像 ---
    query_idx = 0  # 选择第一个查询图像
    query_img = query_images[query_idx]
    print("\n处理查询图像...")
    query_features = featExtra(query_img, feat_type)
    query_hist = desExtra(query_features, codebook, idf=idf, desCate=des_type)

    # --- 相似度计算与排序 ---
    print("计算相似度...")
    similarities = []
    for db_hist in db_hists:
        sim = compute_similarity(query_hist, db_hist)
        similarities.append(sim)
    similarities = np.array(similarities)

    # 获取前10相似结果
    top_k = 10
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    top_similarities = similarities[top_k_indices]
    top_images = [db_images[i] for i in top_k_indices]

    # --- 显示结果 ---
    print("\n相似度排名结果：")
    for rank, (idx, sim) in enumerate(zip(top_k_indices, top_similarities)):
        print(f"Rank {rank + 1}: 相似度={sim:.4f} (数据库索引={idx})")

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

    print("可视化结果...")
    display_results(query_img, top_images, top_similarities)
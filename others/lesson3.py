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

# 图像load函数
def load_images(folder_path):
    """
    加载图像数据，返回图像列表和对应的类别标签列表
    :param folder_path: 文件夹路径
    :return: (images, labels) 图像数组列表和类别标签列表
    """
    images = []
    labels = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    # 读取并转换为numpy数组
                    with Image.open(img_path) as img:
                        img_array = np.array(img)
                        # 确保图像尺寸为450x450（可选）
                        # if img_array.shape != (450, 450, 3):
                        #     img_array = img_array.resize((450,450))
                        images.append(img_array)
                        labels.append(class_folder)
    return images, labels

# 特征提取函数featExtra
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
    """HOG局部特征（每个cell的HOG向量）"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # 添加 visualize=True，确保返回两个值（特征和可视化图）
        hog_features, hog_image = skimage_hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=False,  # 返回每个cell的HOG向量
            transform_sqrt=True,
            channel_axis=None,
            visualize=True          # 新增：必须设置此参数
        )
    # 展平每个cell的特征
    return hog_features.reshape(-1, hog_features.shape[-1])

def _haar_features(gray):
    """Haar小波分解后的局部特征（每个系数层的区域）"""
    from skimage.transform import wavedec2
    coeffs = wavedec2(gray, 'haar', level=2)
    features = []
    for arr in coeffs:
        if isinstance(arr, np.ndarray):
            # 将系数分割为块（例如8x8）
            block_size = 8
            rows = arr.shape[0] // block_size
            cols = arr.shape[1] // block_size
            for i in range(rows):
                for j in range(cols):
                    block = arr[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    features.append(block.ravel())
    return np.array(features)

def _lbp_features(gray):
    """LBP局部特征（每个8x8块的LBP直方图）"""
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    # 分割为8x8块
    block_size = 8
    features = []
    for i in range(0, lbp.shape[0], block_size):
        for j in range(0, lbp.shape[1], block_size):
            block = lbp[i:i+block_size, j:j+block_size]
            hist, _ = np.histogram(block.ravel(), bins=256, range=(0, 256))
            features.append(hist)
    return np.array(features)

def _sift_features(image):
    """SIFT局部特征（描述符）"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors if descriptors is not None else np.empty((0, 128))

def _surf_features(image):
    """SURF局部特征（描述符）"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    surf = cv2.xfeatures2d.SURF_create(4000)
    keypoints, descriptors = surf.detectAndCompute(gray, None)
    return descriptors if descriptors is not None else np.empty((0, 64))

# 特征编码函数（BoF）
def desExtra(features, codebook, desCate='BoF'):
    if desCate == 'BoF':
        if features.size == 0:
            return np.zeros(len(codebook))
        distances = np.linalg.norm(features[:, np.newaxis] - codebook, axis=2)
        labels = np.argmin(distances, axis=1)
        hist, _ = np.histogram(labels, bins=np.arange(len(codebook)+1), density=False)
        return hist.astype(float) / (hist.sum() + 1e-8)  # 避免除以零
    else:
        raise ValueError(f"不支持的编码类型：{desCate}")

# 生成码书（视觉词汇表）
def build_codebook(db_images, featCate='SIFT', n_clusters=100):
    all_features = []
    for img in db_images:
        feats = featExtra(img, featCate)
        if feats.size > 0:
            all_features.append(feats)
    if not all_features:
        raise ValueError("没有足够的特征用于生成码书")
    all_features = np.concatenate(all_features)
    codebook = KMeans(n_clusters=n_clusters, random_state=42).fit(all_features).cluster_centers_
    return codebook

# 相似度计算
def compute_similarity(query_hist, db_hist):
    return cosine_similarity([query_hist], [db_hist])[0][0]

# 主函数1
if __name__ == "__main__":
    # 图片载入路径
    db_images, _ = load_images(r"/dataset/image")  # 数据库图像路径
    query_images, _ = load_images(r"/test")  # 查询图像路径

    # 测试特征提取
    print("测试HOG特征提取:")
    for idx, img in enumerate(query_images):
        features_hog = featExtra(img, 'HOG')
        print(f"图像 {idx + 1} 的 HOG 特征维度: {features_hog.shape}")

    print("\n测试LBP特征提取:")
    for idx, img in enumerate(query_images):
        features_lbp = featExtra(img, 'LBP')
        print(f"图像 {idx + 1} 的 LBP 特征维度: {features_lbp.shape}")

# 主函数2
if __name__ == "__main__":
    # 1. 加载数据库和查询图像
    db_images, _ = load_images(r"/dataset/image")  # 数据库路径
    query_images, _ = load_images(r"/test")       # 查询路径

    # 2. 参数设置
    feat_type = 'HOG'        # 特征类型（HOG/SIFT/LBP等）
    des_type = 'BoF'         # 编码类型（BoF）
    n_clusters = 100         # 码书大小

    # 3. 生成码书
    print("生成视觉词汇表...")
    codebook = build_codebook(db_images, featCate=feat_type, n_clusters=n_clusters)

    # 4. 处理查询图像（假设处理第一个查询图像）
    query_img = query_images[0]
    query_features = featExtra(query_img, feat_type)
    query_hist = desExtra(query_features, codebook, des_type)

    # 5. 预计算所有数据库图像的编码
    print("预处理数据库图像...")
    db_hists = []
    for db_img in db_images:
        db_features = featExtra(db_img, feat_type)
        db_hist = desExtra(db_features, codebook, des_type)
        db_hists.append(db_hist)

    # 6. 计算相似度并获取前10名
    print("计算相似度...")
    similarities = [compute_similarity(query_hist, db_hist) for db_hist in db_hists]
    top_k_indices = np.argsort(similarities)[::-1][:10]  # 前10名索引
    top_similarities = [similarities[i] for i in top_k_indices]
    top_images = [db_images[i] for i in top_k_indices]

    # 7. 显示结果（使用matplotlib）
    def display_results(query_img, top_images, top_similarities):
        # 调整图像尺寸以便显示
        query_resized = resize(query_img, (200, 200))
        resized_images = [resize(img, (200, 200)) for img in top_images]

        # 创建画布（1行11列）
        fig, axes = plt.subplots(1, 11, figsize=(22, 4))
        axes[0].imshow(query_resized)
        axes[0].set_title("Query Image")
        axes[0].axis('off')

        for i in range(10):
            axes[i+1].imshow(resized_images[i])
            axes[i+1].set_title(f"Rank {i+1}\n{top_similarities[i]:.3f}")
            axes[i+1].axis('off')

        plt.tight_layout()
        plt.show()

    print("显示结果...")
    display_results(query_img, top_images, top_similarities)
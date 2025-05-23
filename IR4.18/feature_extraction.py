import cv2
import numpy as np
from skimage.feature import hog as skimage_hog
from skimage.transform import resize
import pywt
from skimage.feature import local_binary_pattern
import warnings


def featExtra(image, featCate, size=(450, 450)):
    """特征提取主函数"""
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
        raise ValueError(f"Unsupported feature type: {featCate}")


def _hog_features(gray):
    """
    HOG特征提取
    :param gray: 输入灰度图像 (2D numpy array)
    :return: 提取的特征 (2D numpy array, 每行是一个特征向量)
    """
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
    """
    Haar小波特征提取
    :param gray: 输入灰度图像 (2D numpy array)
    :return: 提取的特征 (2D numpy array, 每行是一个特征向量)
    """
    # 使用二维 Haar 小波分解，level=2 表示分解层数为 2
    # coeffs = wavedec2(gray, 'haar', level=2) # <-- 修改前
    coeffs = pywt.wavedec2(gray, 'haar', level=2)  # <-- 修改后

    features = []
    block_size = 8  # 定义分块大小

    # 遍历小波系数
    # 注意：wavedec2 返回的是一个包含不同层级系数的列表，
    # 第一个是近似系数 (cA)，后面是不同方向的细节系数 (cH, cV, cD)
    # 原始代码的遍历方式可能需要根据 wavedec2 的实际输出来调整
    # 这里我们先保持原样，但请注意 coeffs 的结构

    # 改进的遍历方式，处理 wavedec2 的输出结构:
    # coeffs = [cA, (cH_level2, cV_level2, cD_level2), (cH_level1, cV_level1, cD_level1)]

    # 方式一：处理所有系数矩阵 (包括近似和各层级细节)
    all_coeffs_arrays = [coeffs[0]]  # 加入 cA
    for detail_coeffs_level in coeffs[1:]:  # 加入各层级细节 (cH, cV, cD)
        all_coeffs_arrays.extend(detail_coeffs_level)

    for arr in all_coeffs_arrays:
        if isinstance(arr, np.ndarray):
            rows = arr.shape[0] // block_size
            cols = arr.shape[1] // block_size

            # 对每个块进行展平并添加到特征列表中
            for i in range(rows):
                for j in range(cols):
                    # 确保块索引不越界 (如果图像尺寸不是 block_size 的整数倍)
                    row_start, row_end = i * block_size, (i + 1) * block_size
                    col_start, col_end = j * block_size, (j + 1) * block_size
                    if row_end <= arr.shape[0] and col_end <= arr.shape[1]:
                        block = arr[row_start:row_end, col_start:col_end]
                        features.append(block.ravel())  # 展平块并添加到特征列表

    # 如果没有提取到特征（例如图像太小），返回一个空的 numpy 数组
    if not features:
        # 你可能需要根据下游处理定义一个合适的空特征形状
        # 例如，如果下游期望固定数量的特征或固定长度的向量，这里需要调整
        print(f"Warning: No Haar features extracted for image size {gray.shape} with block size {block_size}")
        # 返回一个形状为 (0, block_size*block_size) 的数组作为示例
        return np.empty((0, block_size * block_size))

    return np.array(features)  # 返回所有特征的数组形式


def _lbp_features(gray):
    """
    LBP特征提取
    :param gray: 输入灰度图像 (2D numpy array)
    :return: 提取的特征 (2D numpy array, 每行是一个直方图特征向量)
    """
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    block_size = 8

    def extract_hist(block):
        hist, _ = np.histogram(block.ravel(), bins=np.arange(0, n_points + 2), range=(0, n_points + 1))
        return hist

    features = [
        extract_hist(lbp[i:i+block_size, j:j+block_size])
        for i in range(0, lbp.shape[0], block_size)
        for j in range(0, lbp.shape[1], block_size)
    ]
    return np.array(features)


def _sift_features(image):
    """
    SIFT特征提取
    :param image: 输入图像 (3D numpy array, RGB格式)
    :return: 提取的特征描述符 (2D numpy array, 每行是一个128维的特征向量)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors if descriptors is not None else np.empty((0, 128))


def _surf_features(image):
    """
    SURF特征提取
    :param image: 输入图像 (3D numpy array, RGB格式)
    :return: 提取的特征描述符 (2D numpy array, 每行是一个64维的特征向量)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    surf = cv2.xfeatures2d.SURF_create(4000)
    keypoints, descriptors = surf.detectAndCompute(gray, None)
    return descriptors if descriptors is not None else np.empty((0, 64))
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


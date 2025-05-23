import numpy as np
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from feature_extraction import featExtra


def desExtra(features, codebook, idf=None, desCate='BoF'):
    """特征编码"""
    if desCate == 'BoF':
        # 提取特征到码书的最近邻标签（生成词频直方图）
        if features.size == 0:
            return np.zeros(len(codebook))
        distances = np.linalg.norm(features[:, np.newaxis] - codebook, axis=2)
        labels = np.argmin(distances, axis=1)
        hist, _ = np.histogram(labels, bins=np.arange(len(codebook)+1), density=False)
        # 如果提供 IDF，则计算 TF-IDF
        if idf is not None:
            tf_idf = hist * idf # 将词频（hist）乘以 IDF，得到 TF-IDF 值
            return tf_idf / (np.sum(tf_idf) + 1e-8) # 归一化 tf-idf 值
        else:
            return hist.astype(float) / (hist.sum() + 1e-8) # 如果没有 IDF，则直接归一化词频
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
        # 引入IDF加权
        if idf is not None:
            residuals *= idf[:, np.newaxis]  # 按类别加权
        vlad = residuals.flatten()
        vlad /= (np.linalg.norm(vlad) + 1e-8)
        return vlad
    else:
        raise ValueError(f"Unsupported encoding type: {desCate}")


def build_codebook(db_images, featCate='SIFT', n_clusters=100):
    """生成视觉码书"""
    all_features = []
    for img in db_images:
        feats = featExtra(img, featCate)
        if feats.size > 0:
            all_features.append(feats)
    if not all_features:
        raise ValueError("Insufficient features for codebook generation")
    all_features = np.concatenate(all_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    codebook = kmeans.fit(all_features).cluster_centers_

    doc_freq = np.zeros(n_clusters) # 初始化文档频率数组，记录每个聚类中心出现在多少张图像中
    N = len(db_images) # 数据库图像总数
    for img in db_images:
        feats = featExtra(img, featCate)
        if feats.size == 0:
            continue
        distances = np.linalg.norm(feats[:, np.newaxis] - codebook, axis=2) # 计算特征点到码书的距离
        labels = np.argmin(distances, axis=1) # 找到最近的聚类中心索引
        unique_labels = np.unique(labels) # 获取当前图像中出现的唯一聚类中心索引
        for label in unique_labels:
            doc_freq[label] += 1 # 统计每个聚类中心出现的文档数

    # 计算 IDF：log(总文档数 / (文档频率 + 1))
    idf = np.log(N / (doc_freq + 1))  # 加 1 避免除零错误
    return codebook, idf
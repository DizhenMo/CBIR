# QE.py
import numpy as np
from feature_extraction import featExtra
from feature_encoding import desExtra
from metrics import compute_similarity


def expand_query(query_img, db_images, db_hists, codebook, idf, top_indices, des_type='VLAD', qe_top_k=5):
    """
    执行 Query Expansion，对前 top-k 结果进行特征融合并生成新的查询向量

    参数:
        query_img: 原始查询图像 (numpy array)
        db_images: 数据库图像列表
        db_hists: 数据库图像的特征直方图列表
        codebook: 视觉码书
        idf: IDF 权重数组
        top_indices: 初始排序前 top-k 索引列表
        des_type: 特征编码类型 ('BoF' 或 'VLAD')
        qe_top_k: 拓展查询使用的 top-k 数量（默认为5）

    返回:
        expanded_similarities: 使用拓展查询后的相似度得分
    """
    expanded_features = []

    # 添加原始查询图像的特征
    query_features_qe = featExtra(query_img, 'SIFT')  # 假设使用 SIFT 提取局部特征
    if query_features_qe.size > 0:
        expanded_features.append(query_features_qe)

    # 收集 top-k 图像的局部特征
    for idx in top_indices[:qe_top_k]:  # ✅ 正确使用传入的 top_indices
        img = db_images[idx]
        features = featExtra(img, 'SIFT')
        if features.size > 0:
            expanded_features.append(features)

    # 如果没有有效特征，返回 None 表示跳过 QE
    if not expanded_features:
        return None

    # 合并所有特征并重新编码
    expanded_features = np.vstack(expanded_features)
    expanded_hist = desExtra(expanded_features, codebook, idf=idf, desCate=des_type)

    # 计算新相似度
    expanded_similarities = np.array([compute_similarity(expanded_hist, db_hist) for db_hist in db_hists])

    return expanded_similarities

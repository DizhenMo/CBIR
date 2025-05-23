import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(query_hist, db_hist):
    """计算余弦相似度"""
    return cosine_similarity([query_hist], [db_hist])[0][0]


def compute_precision_recall(similarities, relevant_indices):
    """计算PR曲线数据"""
    # 生成100个均匀分布的阈值（0到1之间）
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    for threshold in thresholds:
        # 根据阈值筛选预测为正例的索引（相似度≥阈值的样本）
        predicted_indices = np.where(similarities >= threshold)[0]
        # 计算真正例数量（预测正例与真实正例的交集）
        true_positives = len(set(predicted_indices) & set(relevant_indices))

        # 计算精确率：真正例数 / 预测正例总数
        if len(predicted_indices) == 0:
            precision = 1.0 # 无预测时视为精确率1（避免除零错误）
        else:
            precision = true_positives / len(predicted_indices)
        # 计算召回率：真正例数 / 真实正例总数
        recall = true_positives / len(relevant_indices)

        precisions.append(precision)
        recalls.append(recall)
        
    return precisions, recalls
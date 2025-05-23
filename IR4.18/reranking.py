import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def cluster_reranking(query_hist, db_hists, initial_similarities, top_k=100, n_clusters=5):
    """
    基于聚类的重排序方法
    
    参数:
        query_hist: 查询图像的特征直方图
        db_hists: 数据库图像的特征直方图列表
        initial_similarities: 初始相似度得分
        top_k: 重排序考虑的前k个结果数量（默认100）
        n_clusters: 聚类数量（默认5）
    
    返回:
        reranked_similarities: 重排序后的相似度得分
    """
    # 获取初始排序的前top_k个结果的索引
    top_indices = np.argsort(initial_similarities)[::-1][:top_k]
    
    # 提取前top_k个结果的特征
    top_features = np.array([db_hists[i] for i in top_indices])
    
    # 对前top_k个结果进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(top_features)
    
    # 找到查询图像最近的聚类中心
    query_cluster = kmeans.predict([query_hist])[0]
    
    # 计算每个聚类的权重（基于与查询图像的相似度）
    cluster_weights = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_mask = (cluster_labels == i)
        if np.any(cluster_mask):
            cluster_features = top_features[cluster_mask]
            cluster_sim = cosine_similarity([query_hist], cluster_features).mean()
            cluster_weights[i] = cluster_sim
    
    # 归一化聚类权重
    cluster_weights = cluster_weights / cluster_weights.sum()
    
    # 计算重排序得分
    reranked_similarities = initial_similarities.copy()
    
    # 对前top_k个结果进行重排序
    for idx, orig_idx in enumerate(top_indices):
        # 基础分数（初始相似度）
        base_score = initial_similarities[orig_idx]
        # 聚类加权分数
        cluster_score = cluster_weights[cluster_labels[idx]]
        # 位置权重（越靠前权重越大）
        position_weight = 1.0 - (idx / top_k)
        
        # 综合计算新的相似度得分
        # 参数根据450x450车辆图片特点调整:
        # - 初始相似度权重较大(0.6)保持基础排序的稳定性
        # - 聚类权重适中(0.3)考虑视觉相似性
        # - 位置权重较小(0.1)略微考虑初始排序位置
        new_score = (0.6 * base_score + 
                    0.3 * cluster_score + 
                    0.1 * position_weight)
        
        reranked_similarities[orig_idx] = new_score
    
    return reranked_similarities

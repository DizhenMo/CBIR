import os
import numpy as np
from matplotlib import pyplot as plt

from utils import save_data, load_data, get_cache_filename
from image_loader import load_images
from feature_extraction import featExtra
from feature_encoding import desExtra, build_codebook
from metrics import compute_similarity, compute_precision_recall
from visualization import plot_pr_curve, display_results
from reranking import cluster_reranking
from QE import expand_query
'''
加入重排序后的main函数
加入QE功能
'''

if __name__ == "__main__":
    # 参数定义
    feat_type = 'SIFT'  # 可选项：HOG, LBP, SIFT, Haar, SURF(版权有问题，记得装opencv-contrib-python)
    des_type = 'VLAD'  # 可选项：BoF, VLAD
    n_clusters = 128
    db_folder = r"G:\CS\IR\IR\dataset\image"  # 数据库路径
    query_folder = r"G:\CS\IR\IR\test"  # 查询图像路径
    cache_dir = r"G:\CS\IR\IR\IR4.18\file"  # 中间文件存储路径

    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)

    # 加载图像
    print("加载图像数据...")
    db_images, db_labels = load_images(db_folder)  # 加载数据库图像及其标签
    query_images, _ = load_images(query_folder)
    print(f"数据库图像数量：{len(db_images)}，查询图像数量：{len(query_images)}")

    # 加载或生成码书和IDF
    codebook_file = os.path.join(cache_dir, get_cache_filename("codebook", feat_type, n_clusters))
    idf_file = os.path.join(cache_dir, get_cache_filename("idf", feat_type, n_clusters))
    if os.path.exists(codebook_file) and os.path.exists(idf_file):
        print("加载缓存的码书和IDF...")
        codebook = load_data(codebook_file)
        idf = load_data(idf_file)
    else:
        print("生成码书...")
        codebook, idf = build_codebook(db_images, featCate=feat_type, n_clusters=n_clusters)
        save_data(codebook_file, codebook)
        save_data(idf_file, idf)
    print(f"码书维度：{codebook.shape}，IDF维度：{idf.shape}")

    # 加载或生成数据库特征直方图
    db_hists_file = os.path.join(cache_dir, get_cache_filename("db_hists", feat_type, n_clusters, des_type))
    if os.path.exists(db_hists_file):
        print("加载缓存的数据库特征直方图...")
        db_hists = load_data(db_hists_file)
    else:
        print("处理数据库图像...")
        db_hists = []
        for idx, db_img in enumerate(db_images):
            db_features = featExtra(db_img, feat_type)
            db_hist = desExtra(db_features, codebook, idf=idf, desCate=des_type)
            db_hists.append(db_hist)
            if (idx + 1) % 50 == 0:
                print(f"已处理 {idx + 1}/{len(db_images)} 张数据库图像")
        save_data(db_hists_file, db_hists)
    print(f"数据库特征直方图数量：{len(db_hists)}")

    # 存储所有查询图像的真实标签和相似度分数
    all_true_labels = []
    all_similarities = []

    # 处理所有查询图像
    for query_idx, query_img in enumerate(query_images):
        print(f"\n处理查询图像 {query_idx + 1}/{len(query_images)}...")
        query_features = featExtra(query_img, feat_type)
        query_hist = desExtra(query_features, codebook, idf=idf, desCate=des_type)

        # 计算相似度
        print("计算相似度...")
        similarities = [compute_similarity(query_hist, db_hist) for db_hist in db_hists]
        similarities = np.array(similarities)

        # 执行 Query Expansion
        print("执行 Query Expansion...")
        expanded_similarities = expand_query(
            query_img=query_img,
            db_images=db_images,
            db_hists=db_hists,
            top_indices=np.argsort(similarities)[::-1][:10],  # 初始 top-k
            codebook=codebook,
            idf=idf,
            des_type=des_type,
            qe_top_k=7  # 使用前5个图像拓展查询
        )
        # 不使用QE，(mAP) = 0.7045
        # top_k = 1 (mAP) = 0.7045
        # top_k = 2 (mAP) = 0.7003
        # top_k = 3 (mAP) = 0.7336
        # top_k = 4 (mAP) = 0.7245
        # top_k = 5 (mAP) = 0.7401
        # top_k = 6 (mAP) = 0.7344
        # top_k = 7 (mAP) = 0.7424
        # top_k = 8 (mAP) = 0.7250
        # top_k = 9 (mAP) = 0.7121
        # top_k = 10 (mAP) = 0.7017

        # 替换为 QE 后的相似度结果
        if expanded_similarities is not None:
            similarities = expanded_similarities
        else:
            print("QE 警告：未生成新相似度，使用初始排序")

        # 应用重排序
        print("应用重排序...")
        reranked_similarities = cluster_reranking(
            query_hist=query_hist,
            db_hists=db_hists,
            initial_similarities=similarities,
            top_k=100,  # 考虑前100个结果进行重排序
            n_clusters=10   # 聚类数量，针对车辆图片优化 SIFT+VLAD下5:0.6864 8:0.7021 10:0.7045 23:0.7030
        )

        # 获取前10结果（用于显示结果）
        top_k = 10
        top_k_indices = np.argsort(reranked_similarities)[::-1][:top_k]
        top_similarities = reranked_similarities[top_k_indices]
        top_images = [db_images[i] for i in top_k_indices]

        # 获取 label 为 "A0C573" 的相关图像索引
        relevant_indices = [idx for idx, label in enumerate(db_labels) if label == "A0C573"]
        print(f"label 为 A0C573 的相关图像数量：{len(relevant_indices)}")

        # 如果没有找到相关图像，给出警告并跳过当前查询图像
        if len(relevant_indices) == 0:
            print(f"警告：未找到任何 label 为 A0C573 的相关图像，请检查数据集或标签！")
            continue

        # 收集真实标签和相似度分数
        true_labels = np.array([1 if i in relevant_indices else 0 for i in range(len(db_labels))])
        all_true_labels.extend(true_labels)
        all_similarities.extend(reranked_similarities)

        # 显示单个查询图像的结果
        display_results(query_img, top_images, top_similarities)

    # 绘制总 PR 曲线
    from sklearn.metrics import precision_recall_curve, average_precision_score

    print("\n绘制总 PR 曲线...")
    all_true_labels = np.array(all_true_labels)
    all_similarities = np.array(all_similarities)

    # 计算总 PR 数据和 mAP
    precision, recall, _ = precision_recall_curve(all_true_labels, all_similarities)
    ap = average_precision_score(all_true_labels, all_similarities)

    # 绘制总 PR 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Average Precision (mAP) = {ap:.4f}", color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"所有查询图像的平均精度 (mAP) = {ap:.4f}")


'''
给UI调用的函数
'''


def process_query_image(query_image, feat_type='SIFT', des_type='VLAD', n_clusters=128, apply_reranking=True, apply_qe=True):
    """
    处理单个查询图像并执行特征提取、相似度计算、QE 和重排序流程，返回检索结果和 PR 曲线数据。
    
    参数:
        query_image (PIL.Image): 输入的查询图像对象
        feat_type (str): 特征提取方法，支持 'SIFT', 'HOG', 'LBP', 'Haar'
        des_type (str): 特征编码方法，支持 'BoF', 'VLAD'
        n_clusters (int): 聚类数量，用于码书生成，默认为 128
        apply_reranking (bool): 是否应用基于聚类的重排序，默认为 True
        apply_qe (bool): 是否应用 Query Expansion，默认为 True
    
    返回:
        results (List[Tuple[np.ndarray, float]]): 检索到的前10个图像及其相似度得分，
                                              每项格式为 (image_array, similarity_score)
        pr_curve_data (Tuple[List[float], List[float]]): PR曲线数据，包含两个列表：
                                                  - precisions: 各阈值下的精确率
                                                  - recalls: 各阈值下的召回率
    """
    from utils import load_data
    from feature_extraction import featExtra
    from feature_encoding import desExtra, build_codebook
    from metrics import compute_similarity, compute_precision_recall
    from reranking import cluster_reranking
    from image_loader import load_images

    # 参数定义
    db_folder = r"G:\CS\IR\IR\dataset\image"  # 数据库路径
    cache_dir = r"G:\CS\IR\IR\IR4.18\file"  # 中间文件存储路径

    # 加载数据库图像
    db_images, db_labels = load_images(db_folder)

    # 加载或生成码书和IDF
    codebook_file = os.path.join(cache_dir, get_cache_filename("codebook", feat_type, n_clusters))
    idf_file = os.path.join(cache_dir, get_cache_filename("idf", feat_type, n_clusters))
    if os.path.exists(codebook_file) and os.path.exists(idf_file):
        codebook = load_data(codebook_file)
        idf = load_data(idf_file)
    else:
        codebook, idf = build_codebook(db_images, featCate=feat_type, n_clusters=n_clusters)

    # 加载或生成数据库特征直方图
    db_hists_file = os.path.join(cache_dir, get_cache_filename("db_hists", feat_type, n_clusters, des_type))
    if os.path.exists(db_hists_file):
        db_hists = load_data(db_hists_file)
    else:
        db_hists = [desExtra(featExtra(img, feat_type), codebook, idf=idf, desCate=des_type) for img in db_images]

    # 查询图像处理
    query_features = featExtra(np.array(query_image), feat_type)
    query_hist = desExtra(query_features, codebook, idf=idf, desCate=des_type)

    # 计算相似度
    similarities = np.array([compute_similarity(query_hist, db_hist) for db_hist in db_hists])

    # 执行 Query Expansion（如果启用）
    if apply_qe:
        print("执行 Query Expansion...")
        expanded_similarities = expand_query(
            query_img=np.array(query_image),
            db_images=db_images,
            db_hists=db_hists,
            top_indices=np.argsort(similarities)[::-1][:10],  # 初始 top-k
            codebook=codebook,
            idf=idf,
            des_type=des_type,
            qe_top_k=5
        )

        if expanded_similarities is not None:
            similarities = expanded_similarities
        else:
            print("QE 警告：未生成新相似度，使用初始排序")

    # 应用重排序（如果启用）
    if apply_reranking:
        reranked_similarities = cluster_reranking(
            query_hist=query_hist,
            db_hists=db_hists,
            initial_similarities=similarities,
            top_k=100,
            n_clusters=10
        )
    else:
        reranked_similarities = similarities

    # 获取前10结果
    top_k_indices = np.argsort(reranked_similarities)[::-1][:10]
    top_similarities = reranked_similarities[top_k_indices]
    top_images = [db_images[i] for i in top_k_indices]

    # 收集相关图像索引
    relevant_indices = [idx for idx, label in enumerate(db_labels) if label == "A0C573"]

    # 计算PR曲线数据
    precisions, recalls = compute_precision_recall(similarities, relevant_indices)

    # 返回结果
    results = list(zip(top_images, top_similarities))
    pr_curve_data = (precisions, recalls)
    return results, pr_curve_data

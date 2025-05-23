import os
import numpy as np
from matplotlib import pyplot as plt

from utils import save_data, load_data, get_cache_filename
from image_loader import load_images
from feature_extraction import featExtra
from feature_encoding import desExtra, build_codebook
from metrics import compute_similarity, compute_precision_recall
from visualization import plot_pr_curve, display_results
from QE import expand_query
from reranking import cluster_reranking


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

        # 获取前10结果（用于显示结果）
        top_k = 10
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_k_indices]
        top_images = [db_images[i] for i in top_k_indices]

        # 执行 Query Expansion
        expanded_similarities = expand_query(
            query_img=query_img,
            db_images=db_images,
            db_hists=db_hists,  # 👈 新增这一行：传入 db_hists
            top_indices=top_k_indices,
            codebook=codebook,
            idf=idf,
            des_type=des_type,
            qe_top_k=5
        )
        # 不使用QE时 (mAP) = 0.6520
        # top_k = 1时，(mAP) = 0.6516
        # top_k = 2时，(mAP) = 0.6898
        # top_k = 3时，(mAP) = 0.7111
        # top_k = 4时，(mAP) = 0.7189
        # top_k = 5时，(mAP) = 0.7280
        # top_k = 6时，(mAP) = 0.7267
        # top_k = 7时，(mAP) = 0.7194
        # top_k = 8时，(mAP) = 0.7040
        # top_k = 9时，(mAP) = 0.7107
        # top_k = 10时，(mAP) = 0.7091

        # 替换为 QE 后的相似度结果
        if expanded_similarities is not None:
            similarities = expanded_similarities
            top_k_indices = np.argsort(similarities)[::-1][:top_k]  # 更新排序
            top_similarities = similarities[top_k_indices]
            top_images = [db_images[i] for i in top_k_indices]
        else:
            print("QE 警告：未生成新相似度，使用初始排序")

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
        all_similarities.extend(similarities)

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
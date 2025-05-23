# 以图搜图图像检索系统
## 1.项目介绍
这是一个基于内容的图像检索（Content-Based Image Retrieval, CBIR）系统，主要用于通过输入查询图像，在数据库中找到相似的图像，并展示结果。项目结合了多种特征提取和编码技术、Query Expansion（QE）、以及重排序（Re-ranking）等方法来提升检索准确率。
## 2.核心功能
### 特征提取与编码  
支持多种特征提取方法：HOG, LBP, SIFT, Haar, SURF  
支持两种特征编码方式：BoF (Bag of Features)，VLAD (Vector of Locally Aggregated Descriptors)  
### 查询扩展（Query Expansion）
使用初始检索出的 Top-k 图像进行特征融合，生成新的查询向量。  
可以有效提高检索的平均精度（mAP）。
### 基于聚类的重排序（Cluster-based Re-ranking）
对初始检索结果进行二次优化，进一步提升排序准确性。
### 效果评估
每次查询后绘制 PR 曲线，并计算平均精度（mAP），用于评估检索效果。
## 3.脚本描述
### main2.py
主程序逻辑，包含完整的图像检索流程，支持 QE 和重排序。
### UI.py
Tkinter 实现的图形用户界面，用于上传查询图像并显示结果。
### image_loader.py
加载图像数据集，按文件夹组织图像及其标签。
### feature_extraction.py
提取局部特征和全局描述子。
### feature_encoding.py
编码为 BoF/VLAD 向量，并构建码书（Codebook）。
### QE.py
实现 Query Expansion，融合多个图像特征提升检索性能。
### reranking.py
实现基于聚类的重排序算法。
### metrics.py
计算相似度，mAP和PR曲线
### visualization.py
显示检索结果和 PR 曲线。
### utils.py
工具函数：缓存加载/保存、路径处理等。
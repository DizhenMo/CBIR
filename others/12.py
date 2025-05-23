import os
import time
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import joblib
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from math import log


class ImageRetrievalApp:
    def __init__(self, root):
        """初始化图像检索应用程序"""
        self.root = root
        self.root.title("图像检索系统 (TF-IDF倒排索引版)")
        self.root.geometry("1200x800")

        # 初始化系统变量
        self.train_folder = None  # 训练集路径
        self.test_folder = None  # 测试集路径
        self.dictionary = None  # 视觉词典（聚类中心）
        self.idf = None  # IDF权重向量（numpy数组）
        self.train_features = []  # 训练集特征向量（TF-IDF）
        self.test_features = []  # 测试集特征向量
        self.train_images_paths = []  # 训练图像路径列表
        self.test_images_paths = []  # 测试图像路径列表
        self.train_labels = []  # 训练图像类别标签
        self.ap_history = []  # 保存每次检索的AP值
        self.recall_history = []  # 保存每次检索的召回率
        self.inverted_index = defaultdict(list)  # 倒排索引结构：{word: [(img_idx, weight)]}

        # 自动加载已有模型和数据
        self.load_saved_models()

        # 初始化界面
        self.create_widgets()

    def create_widgets(self):
        """创建GUI界面组件"""
        # 主容器
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 指标显示面板
        metrics_frame = tk.Frame(main_frame, relief=tk.RAISED, bd=1)
        metrics_frame.pack(fill=tk.X, pady=5, padx=5)

        # 定义指标显示样式
        metric_config = {'font': ('宋体', 10), 'width': 15}
        self.ap_label = tk.Label(metrics_frame, text="AP: 0.0000", **metric_config)
        self.ap_label.pack(side=tk.LEFT, padx=10)

        self.recall_label = tk.Label(metrics_frame, text="召回率: 0.0000", **metric_config)
        self.recall_label.pack(side=tk.LEFT, padx=10)

        self.map_label = tk.Label(metrics_frame, text="MAP: 0.0000", **metric_config)
        self.map_label.pack(side=tk.LEFT, padx=10)

        self.mar_label = tk.Label(metrics_frame, text="MAR: 0.0000", **metric_config)
        self.mar_label.pack(side=tk.LEFT, padx=10)

        # 控制按钮面板
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10, fill=tk.X)

        # 功能按钮配置
        buttons = [
            ("选择训练集", self.select_train_folder),
            ("选择测试集", self.select_test_folder),
            ("生成视觉词典", self.generate_dictionary),
            ("编码训练集", self.encode_train_images),
            ("编码测试集", self.encode_test_images),
            ("图像检索", self.select_image_and_search),
            ("绘制PR曲线", self.evaluate_all_queries_and_plot_pr)
        ]

        # 排列按钮
        for idx, (text, cmd) in enumerate(buttons):
            btn = tk.Button(button_frame, text=text, command=cmd, width=12)
            btn.grid(row=0, column=idx, padx=5)

        # 查询图像显示区域
        self.query_frame = tk.Frame(main_frame, height=180)
        self.query_frame.pack(fill=tk.X, pady=5)

        # 检索结果展示区域
        result_container = tk.Frame(main_frame)
        result_container.pack(fill=tk.BOTH, expand=True)

        # 滚动条和画布配置
        self.canvas = tk.Canvas(result_container)
        self.scrollbar = ttk.Scrollbar(result_container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 结果展示内部框架
        self.result_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.result_frame, anchor="nw")

        # 布局组件
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 状态栏
        self.status_bar = tk.Label(main_frame, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 绑定画布滚动事件
        self.result_frame.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))

    # 核心功能方法 ------------------------------------------------------------

    def evaluate_all_queries_and_plot_pr(self):
        """对测试集所有图像执行检索并输出PR曲线、P@K、R@K、MAP等指标"""
        if not self.test_features or not self.test_images_paths:
            messagebox.showerror("错误", "请先编码测试集！")
            return

        all_labels = []
        all_scores = []

        # 统计指标
        ap_list = []
        recall_list = []
        precision_at_k = {1: [], 5: [], 10: []}
        recall_at_k = {1: [], 5: [], 10: []}

        for test_vector, test_path in zip(self.test_features, self.test_images_paths):
            query_class = os.path.basename(os.path.dirname(test_path))
            total_relevant = sum(1 for label in self.train_labels if label == query_class)

            scores = defaultdict(float)
            query_words = np.where(test_vector > 1e-6)[0]

            for word in query_words:
                for img_idx, weight in self.inverted_index.get(word, []):
                    scores[img_idx] += weight * test_vector[word]

            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            retrieved_indices = [i for i, _ in sorted_scores]
            retrieved_scores = [s for _, s in sorted_scores]

            relevance = [1 if self.train_labels[i] == query_class else 0 for i in retrieved_indices]
            all_labels.extend(relevance)
            all_scores.extend(retrieved_scores)

            # 计算AP和Recall
            relevance_array = np.array(relevance)
            cumsum = np.cumsum(relevance_array)
            precision = cumsum / (np.arange(len(relevance_array)) + 1)
            if total_relevant > 0:
                ap = np.sum(precision * relevance_array) / total_relevant
                recall = np.sum(relevance_array) / total_relevant
            else:
                ap = 0
                recall = 0
            ap_list.append(ap)
            recall_list.append(recall)

            for k in [1, 5, 10]:
                top_k = relevance[:k]
                retrieved_k = min(len(top_k), total_relevant) if total_relevant > 0 else 1
                precision_at_k[k].append(sum(top_k) / k)
                recall_at_k[k].append(sum(top_k) / retrieved_k)

        # 计算整体PR曲线
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_scores)
        ap_total = average_precision_score(all_labels, all_scores)

        # 绘图
        plt.figure(figsize=(6, 5))
        plt.plot(recall_curve, precision_curve, label=f'AP = {ap_total:.4f}', color='blue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("pr_curve.png")
        plt.show()

        # 控制台输出结果
        print("=" * 50)
        print("📊 检索性能评估指标：")
        print(f"MAP (Mean Average Precision): {np.mean(ap_list):.4f}")
        print(f"MAR (Mean Average Recall):    {np.mean(recall_list):.4f}")
        for k in [1, 5, 10]:
            print(f"Precision@{k}: {np.mean(precision_at_k[k]):.4f}")
            print(f"Recall@{k}:    {np.mean(recall_at_k[k]):.4f}")
        print("=" * 50)

        self.status_bar.config(
            text=f"PR曲线完成 | MAP: {np.mean(ap_list):.4f} | P@10: {np.mean(precision_at_k[10]):.4f}")

    def load_saved_models(self):
        """加载已保存的模型和数据"""
        try:
            # 加载视觉词典
            if os.path.exists('../visual_dictionary.pkl'):
                self.dictionary = joblib.load('../visual_dictionary.pkl')

            # 加载训练数据（包含倒排索引和IDF）
            if os.path.exists('../train_data.pkl'):
                data = joblib.load('../train_data.pkl')
                # 验证数据完整性
                if len(data) != 5:
                    raise ValueError("训练数据格式不兼容")

                # 解包数据：确保顺序正确
                (self.train_features,
                 self.train_images_paths,
                 self.train_labels,
                 self.inverted_index,
                 self.idf) = data

                # 强制类型转换确保计算安全
                self.idf = self.idf.astype(np.float64)
                self.train_features = [v.astype(np.float64) for v in self.train_features]

        except Exception as e:
            messagebox.showerror("加载错误", f"数据加载失败：{str(e)}")
            # 删除损坏文件
            if os.path.exists('../train_data.pkl'):
                os.remove('../train_data.pkl')
            self.dictionary = None
            self.idf = None

    def select_train_folder(self):
        """选择训练集目录"""
        self.train_folder = filedialog.askdirectory()
        self.status_bar.config(text=f"训练集路径: {self.train_folder}")

    def select_test_folder(self):
        """选择测试集目录"""
        self.test_folder = filedialog.askdirectory()
        self.status_bar.config(text=f"测试集路径: {self.test_folder}")

    def generate_dictionary(self):
        """生成视觉词典（视觉单词聚类中心）"""
        # 检查已有词典
        if os.path.exists('../visual_dictionary.pkl'):
            self.dictionary = joblib.load('../visual_dictionary.pkl')
            messagebox.showinfo("提示", "已加载现有视觉词典！")
            return

        # 初始化SIFT特征检测器
        sift = cv2.SIFT_create()
        descriptors_list = []

        # 遍历训练集提取特征
        for folder, _, filenames in os.walk(self.train_folder):
            for filename in filenames:
                if filename.lower().endswith(('jpg', 'jpeg', 'png')):
                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # 提取SIFT特征
                    keypoints, descriptors = sift.detectAndCompute(img, None)
                    if descriptors is not None:
                        descriptors_list.append(descriptors)

        # 检查特征有效性
        if not descriptors_list:
            messagebox.showerror("错误", "未提取到任何特征！")
            return

        # 合并所有描述符并进行K-means聚类
        descriptors_stack = np.vstack(descriptors_list)
        kmeans = KMeans(n_clusters=1000, random_state=42, n_init=10)
        kmeans.fit(descriptors_stack)

        # 保存视觉词典
        self.dictionary = kmeans.cluster_centers_
        joblib.dump(self.dictionary, '../visual_dictionary.pkl')
        messagebox.showinfo("提示", f"视觉词典已生成！共{len(self.dictionary)}个视觉单词")

    def encode_train_images(self):
        """编码训练集图像并构建倒排索引"""
        # 前置检查
        if self.dictionary is None:
            messagebox.showerror("错误", "请先生成视觉词典！")
            return
        if os.path.exists('../train_data.pkl'):
            messagebox.showinfo("提示", "已加载训练数据！")
            return

        sift = cv2.SIFT_create()
        df = np.zeros(self.dictionary.shape[0], dtype=np.int32)  # 文档频率统计
        tf_features = []  # 临时存储TF特征
        self.train_images_paths = []
        self.train_labels = []

        # 第一轮遍历：计算文档频率（DF）
        for root_dir, _, filenames in os.walk(self.train_folder):
            class_name = os.path.basename(root_dir)
            for filename in filenames:
                if filename.lower().endswith(('jpg', 'jpeg', 'png')):
                    img_path = os.path.join(root_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # 提取特征并量化到视觉词典
                    keypoints, descriptors = sift.detectAndCompute(img, None)
                    if descriptors is not None:
                        words, _ = vq(descriptors, self.dictionary)
                        # 统计每个视觉单词出现的文档数
                        present_words = np.unique(words)
                        df[present_words] += 1
                        # 记录词频（TF）
                        hist, _ = np.histogram(words, bins=np.arange(len(self.dictionary) + 1))
                        tf_features.append(hist)
                        self.train_images_paths.append(img_path)
                        self.train_labels.append(class_name)

        # 计算IDF（逆文档频率）
        N = len(tf_features)  # 总文档数
        self.idf = np.log(N / (df.astype(float) + 1e-6))  # 添加平滑项避免除零

        # 转换为TF-IDF并归一化
        self.train_features = []
        for tf in tf_features:
            tf = tf.astype(float)
            tf_idf = tf * self.idf  # 计算TF-IDF
            norm = np.linalg.norm(tf_idf)
            tf_idf_normalized = tf_idf / (norm + 1e-10)  # L2归一化
            self.train_features.append(tf_idf_normalized)

        # 构建倒排索引
        self.build_inverted_index()

        # 保存训练数据
        joblib.dump(
            (self.train_features,
             self.train_images_paths,
             self.train_labels,
             self.inverted_index,
             self.idf.astype(np.float32)),  # 压缩存储
            '../train_data.pkl'
        )
        self.status_bar.config(text=f"训练集编码完成！共编码{len(self.train_features)}张图像")

    def build_inverted_index(self):
        """构建倒排索引结构"""
        self.inverted_index = defaultdict(list)
        for img_idx, feature in enumerate(self.train_features):
            # 提取非零权重特征
            non_zero = np.where(feature > 1e-6)[0]
            for word in non_zero:
                # 存储格式：(图像索引, 权重)
                self.inverted_index[word].append((img_idx, feature[word]))

    def encode_test_images(self):
        """编码测试集图像"""
        # 前置检查
        if self.dictionary is None or self.idf is None:
            messagebox.showerror("错误", "请先编码训练集！")
            return

        sift = cv2.SIFT_create()
        self.test_features = []
        self.test_images_paths = []

        # 遍历测试集
        for root_dir, _, filenames in os.walk(self.test_folder):
            for filename in filenames:
                if filename.lower().endswith(('jpg', 'jpeg', 'png')):
                    img_path = os.path.join(root_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # 提取特征并量化
                    keypoints, descriptors = sift.detectAndCompute(img, None)
                    if descriptors is not None:
                        words, _ = vq(descriptors, self.dictionary)
                        # 计算TF-IDF
                        hist, _ = np.histogram(words, bins=np.arange(len(self.dictionary) + 1))
                        tf_idf = hist.astype(float) * self.idf
                        norm = np.linalg.norm(tf_idf)
                        tf_idf_normalized = tf_idf / (norm + 1e-10)
                        self.test_features.append(tf_idf_normalized)
                        self.test_images_paths.append(img_path)

        joblib.dump((self.test_features, self.test_images_paths), '../test_data.pkl')
        self.status_bar.config(text=f"测试集编码完成！共编码{len(self.test_features)}张图像")

    def select_image_and_search(self):
        """选择查询图像并执行检索"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.search_image(file_path)

    def search_image(self, image_path):
        """执行图像检索"""
        # 前置检查
        if not self.train_features:
            messagebox.showerror("错误", "请先编码训练集！")
            return

        start_time = time.time()

        # 提取查询图像特征
        sift = cv2.SIFT_create()
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None:
            messagebox.showerror("错误", "无法提取图像特征！")
            return

        # 生成查询向量
        words, _ = vq(descriptors, self.dictionary)
        hist, _ = np.histogram(words, bins=np.arange(len(self.dictionary) + 1))
        tf_idf_query = hist.astype(float) * self.idf
        norm = np.linalg.norm(tf_idf_query)
        query_vector = tf_idf_query / (norm + 1e-10)

        # 倒排索引检索 --------------------------------------------------------
        scores = defaultdict(float)  # 存储图像得分
        query_words = np.where(query_vector > 1e-6)[0]  # 查询包含的视觉单词


        for word in query_words:
            # 获取倒排列表中的(图像索引, 权重)对
            for img_idx, weight in self.inverted_index.get(word, []):
                # 累加相似度得分（余弦相似度分解计算）
                scores[img_idx] += weight * query_vector[word]
        search_time = time.time() - start_time

        # 获取Top10结果
        top_k = 10
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_candidates = [item[0] for item in sorted_scores]
        distances = [1 - item[1] for item in sorted_scores]  # 转换为距离值

        # 评估指标计算 --------------------------------------------------------
        query_class = os.path.basename(os.path.dirname(image_path))
        total_relevant = sum(1 for label in self.train_labels if label == query_class)
        ap = 0.0
        recall = 0.0

        if total_relevant > 0:
            # 计算检索结果的二值相关性
            retrieved_labels = [self.train_labels[i] for i in top_candidates]
            binary_relevance = np.array([1 if label == query_class else 0 for label in retrieved_labels])

            # 计算召回率
            relevant_retrieved = np.sum(binary_relevance)
            recall = relevant_retrieved / total_relevant

            # 计算平均精度（AP）
            cumsum = np.cumsum(binary_relevance)
            precision_at_k = cumsum / (np.arange(len(binary_relevance)) + 1)
            ap = np.sum(precision_at_k * binary_relevance) / top_k

            # 记录历史指标
            self.ap_history.append(ap)
            self.recall_history.append(recall)
        else:
            messagebox.showwarning("警告", "未找到相关类别图像！")

        # 更新全局指标
        map_value = np.mean(self.ap_history) if self.ap_history else 0.0
        mar_value = np.mean(self.recall_history) if self.recall_history else 0.0

        self.ap_label.config(text=f"AP: {ap:.4f}")
        self.recall_label.config(text=f"召回率: {recall:.4f}")
        self.map_label.config(text=f"MAP: {map_value:.4f}")
        self.mar_label.config(text=f"MAR: {mar_value:.4f}")

        # 显示结果
        self.show_query_image(image_path)
        self.show_results(top_candidates, distances, search_time)

    def show_query_image(self, image_path):
        """显示查询图像"""
        # 清空旧内容
        for widget in self.query_frame.winfo_children():
            widget.destroy()

        # 加载并显示图像
        img = Image.open(image_path)
        img = img.resize((160, 160), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        label = tk.Label(self.query_frame, image=img_tk)
        label.image = img_tk
        label.pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(self.query_frame, text="查询图片", font=('宋体', 10)).pack(side=tk.LEFT)

    def show_results(self, indices, distances, search_time):
        """显示检索结果"""
        # 清空旧结果
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # 显示每个结果图像
        for idx, (img_idx, distance) in enumerate(zip(indices, distances)):
            img_path = self.train_images_paths[img_idx]
            try:
                img = Image.open(img_path)
                img = img.resize((150, 150), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)

                # 创建结果条目容器
                frame = tk.Frame(self.result_frame)
                frame.grid(row=idx // 5, column=idx % 5, padx=10, pady=10)

                # 显示图像和相似度
                label = tk.Label(frame, image=img_tk)
                label.image = img_tk
                label.pack()
                tk.Label(frame, text=f"相似度: {1 - distance:.4f}", font=('宋体', 8)).pack()
            except Exception as e:
                print(f"加载图片失败: {img_path} - {str(e)}")

        # 更新状态栏
        self.status_bar.config(text=f"检索完成！耗时: {search_time:.4f}s | 返回结果: {len(indices)} 张")


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageRetrievalApp(root)
    root.mainloop()
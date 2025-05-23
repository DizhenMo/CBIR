# ui_tkinter.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from main2 import process_query_image  # 假设您将核心逻辑封装为函数


class ImageRetrievalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("以图搜图图像检索系统")
        self.root.resizable(False, False)  # 禁止调整窗口大小

        # 参数变量
        self.feat_type = tk.StringVar(value="SIFT")
        self.des_type = tk.StringVar(value="VLAD")
        self.query_image_path = tk.StringVar()
        self.apply_reranking = tk.BooleanVar(value=True)
        self.apply_qe = tk.BooleanVar(value=True)  # 默认启用 QE

        # 界面布局
        self.create_widgets()

    def create_widgets(self):
        # 标题
        tk.Label(self.root, text="以图搜图图像检索系统", font=("Arial", 16)).grid(
            row=0, column=0, columnspan=2, pady=10, sticky="w"
        )

        # 特征提取方法选择
        tk.Label(self.root, text="特征提取方法:").grid(
            row=1, column=0, sticky="w", padx=10, pady=5
        )
        feat_options = ["SIFT", "HOG", "LBP", "Haar"]
        tk.OptionMenu(self.root, self.feat_type, *feat_options).grid(
            row=1, column=1, sticky="w", padx=10, pady=5
        )

        # 特征编码方法选择
        tk.Label(self.root, text="特征编码方法:").grid(
            row=2, column=0, sticky="w", padx=10, pady=5
        )
        des_options = ["BoF", "VLAD"]
        tk.OptionMenu(self.root, self.des_type, *des_options).grid(
            row=2, column=1, sticky="w", padx=10, pady=5
        )

        # 查询图像路径选择
        tk.Label(self.root, text="查询图像路径:").grid(
            row=3, column=0, sticky="w", padx=10, pady=5
        )
        tk.Entry(self.root, textvariable=self.query_image_path, width=50).grid(
            row=3, column=1, sticky="w", padx=10, pady=5
        )
        tk.Button(self.root, text="浏览", command=self.select_image).grid(
            row=3, column=2, padx=5, pady=5
        )

        # 是否应用重排序
        tk.Checkbutton(self.root, text="应用重排序", variable=self.apply_reranking).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=10, pady=5
        )

        # 是否应用 QE
        tk.Checkbutton(self.root, text="应用 Query Expansion", variable=self.apply_qe).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=10, pady=5
        )

        # 开始检索按钮
        tk.Button(self.root, text="开始检索", command=self.start_retrieval).grid(
            row=6, column=0, columnspan=2, sticky="w", padx=10, pady=10
        )

        # 结果展示区域
        self.result_frame = tk.Frame(self.root)
        self.result_frame.grid(row=6, column=0, columnspan=3, pady=10, sticky="w")

    def select_image(self):
        """选择查询图像"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
            title="选择查询图像"
        )
        if file_path:
            self.query_image_path.set(file_path)

    def start_retrieval(self):
        """开始检索"""
        # 检查输入
        if not self.query_image_path.get():
            messagebox.showerror("错误", "请先选择查询图像！")
            return

        # 加载查询图像
        try:
            query_image = Image.open(self.query_image_path.get())
        except Exception as e:
            messagebox.showerror("错误", f"无法加载查询图像: {e}")
            return

        # 调用核心逻辑
        feat_type = self.feat_type.get()
        des_type = self.des_type.get()
        apply_reranking = self.apply_reranking.get()

        try:
            results, pr_curve_data = process_query_image(
                query_image,
                feat_type=feat_type,
                des_type=des_type,
                apply_reranking=apply_reranking
            )
        except Exception as e:
            messagebox.showerror("错误", f"检索失败: {e}")
            return

        # 显示结果
        self.display_results(query_image, results)

    def display_results(self, query_image, results):
        """显示检索结果"""
        from PIL import Image  # 导入 PIL.Image

        # 清空结果区域
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # 显示查询图像
        query_image.thumbnail((200, 200))
        query_photo = ImageTk.PhotoImage(query_image)
        tk.Label(self.result_frame, image=query_photo).grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        tk.Label(self.result_frame, text="查询图像").grid(
            row=1, column=0, padx=5, sticky="w"
        )

        # 显示前10个结果
        for rank, (img, similarity) in enumerate(results[:10], start=1):
            # 将 numpy.ndarray 转换为 PIL.Image
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)  # 转换为 PIL.Image 对象

            # 缩略图处理
            img.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(self.result_frame, image=photo)
            label.image = photo  # 防止垃圾回收
            label.grid(row=0, column=rank, padx=5, pady=5, sticky="w")
            tk.Label(self.result_frame, text=f"Rank {rank}\n{similarity:.3f}").grid(
                row=1, column=rank, padx=5, sticky="w"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRetrievalApp(root)
    root.mainloop()
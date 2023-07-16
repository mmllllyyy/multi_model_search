import torch
import clip
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import json

class MultiModel:
    """
    构造函数，读取模型

    参数:
        model_path:字符串，模型地址
    """
    def __init__(self, model_path):
        # 检查GPU
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device state: ", self.__device)
        print()
        
        # 数据加载标志，npy已加载为True，未加载为False
        self.__img_npy_loaded = False
        self.__text_npy_loaded = False
        
        # 创建两个列表，分别存储编码后的图片特征和图片文件名
        self.__image_features = []
        self.__image_names = []
        # 创建两个列表，分别存储编码后的文本特征和对应的信息（行号和图片文件名）
        self.__text_features = []
        self.__text_info = []
        # 存储当次搜索的文本或图片的特征值
        self.__text_feature = torch.zeros(1)
        self.__img_feature = torch.zeros(1)
        
        try:
            # 读取图片特征,将numpy数组转为Tensor
            self.__image_features = np.load('image_features.npy')
            self.__image_names = np.load('image_names.npy')
            self.__image_features = torch.tensor(self.__image_features).to(self.__device)
            # 标记为已加载
            self.__img_npy_loaded = True
            
            # 读取文本特征,将numpy数组转为Tensor
            self.__text_features = np.load('text_features.npy')
            with open('text_info.json', 'r') as f:
                lines = f.readlines()
                self.__text_info = [json.loads(line) for line in lines]
            self.__text_features = torch.tensor(self.__text_features).to(self.__device)
            # 标记为已加载
            self.__text_npy_loaded = True
            
        except Exception as e:
            print("something went wrong while loading npy, check the model path or load data first")
            print("details: ", str(e))
            print()
        
        try:
            # __model：预训练模型
            # __transformer：图像预处理函数
            #self.__model, self.__transform = clip.load("ViT-B/32", device=self.__device, download_root=model_path)
            self.__model, self.__transform = clip.load("ViT-B/32", device=self.__device, download_root=model_path)
            print("load model done")
            print()
        
        except Exception as e:
            print("something went wrong while loading model, check the model path")
            print("details: ", str(e))
            print()
            self.__model = None
            self.__transform = None
    
        
        
        
    """
    计算余弦相似度
    """
    def __cal_similarity(self, a, b):
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a).to(self.__device)
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b).to(self.__device)
        a = a.float()
        b = b.float()
        
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        similarity = a @ b.T
        return similarity
        
    
    
    
    """
    编码单张图片，返回这张图片的特征向量

    参数:
        img_path:字符串，图片地址
        
    返回：
        __image_feature:图片特征向量
    """
    def EncodeImg(self, img_path):
        try:
            # 加载图片
            image = Image.open(img_path)
        except Exception as e:
            print("something went wrong while loading image for encode it, check the image path")
            print("details: ", str(e))
            print()
        
        # 对图片进行预处理
        image = self.__transform(image).unsqueeze(0).to(self.__device)
        
        # 使用模型来编码图片
        self.__img_feature = self.__model.encode_image(image)
        
        return self.__img_feature
    
    
    
    
    """
    编码单句文本，返回这句文本的特征向量

    参数:
        text:字符串，文本内容
        
    返回：
        __text_feature:图片特征向量
    """
    def EncodeText(self, text):
        if len(text) > 300:
            # 如果文本长度超过300，进行裁剪或者分割
            print("your text is out of limit, the content longer than 77 will be ignored")
            print()
            text = text[:300]
        
        # 使用模型来编码文本
        text = clip.tokenize([text]).to(self.__device)
        self.__text_feature = self.__model.encode_text(text)
        
        return self.__text_feature




    """
    批量处理图片，返回图片的特征向量，存储成npy文件

    参数:
        images_path:字符串，数据集中的图片所在的地址
        
    返回：
        image_features:图片特征的列表
        image_names:图片名的列表，与图片特征一一对应
    """
    def LoadImgData(self, img_path):
        if self.__img_npy_loaded == True:
            print("image features are loaded already")
            print()
            return self.__image_features, self.__image_names
        
        
        try:
            # 获取目录中的所有文件
            files = os.listdir(img_path)
        except FileNotFoundError:
            print(f"File {img_path} not found.")
        else:
            # 筛选出图片文件，这里假设所有图片都是.jpg格式
            image_files = [f for f in files if f.endswith('.jpg')]
            
            # 创建完整的文件路径列表
            image_paths = [os.path.join(img_path, f) for f in image_files]
            

            # 对每个图像进行处理
            for path in image_paths:
                print(path)
                
                # 使用提取特征
                feature = self.EncodeImg(path)

                # 将特征添加到列表中
                self.__image_features.append(feature.detach().cpu().numpy())

                # 将图像名添加到列表中
                self.__image_names.append(os.path.basename(path))
                
                # 清空gpu缓存
                if self.__device == 'cuda':
                    torch.cuda.empty_cache()  

            try:
                # 保存特征和图像名
                np.save('image_features.npy', self.__image_features)
                np.save('image_names.npy', self.__image_names)
                
            except FileExistsError:
                print("The files already exists.")
                print()
            
            return self.__image_features, self.__image_names
    
    
    
    
    """
    处理标注文件，返回文本的特征向量等信息，存储成npy文件

    参数:
        token_path:字符串，数据集中文本标注文件的地址
        
    返回：
        text_features:文本特征的列表
        text_info:文本所在行数及文本描述的图片名称，与文本特征一一对应, [[line, img_name], [...], ...] 
    """
    def LoadTokenData(self, token_path):
        if self.__text_npy_loaded == True:
            print("text features are loaded already")
            print()
            return self.__text_features, self.__text_info
        
        try:
            # 打开文件并逐行读取
            with open(token_path, 'r') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    print(f"loading token line {i}")
                    # 根据'\t'分割得到图片名和编号以及标注文本
                    image_name_and_number, text = line.strip().split('\t')

                    # 再根据 '#' 分割图片名和编号
                    image_name, number = image_name_and_number.split('#')
                    
                    # 编码
                    encoded_text = self.EncodeText(text)

                    # 将编码后的文本添加到列表中
                    self.__text_features.append(encoded_text.detach().cpu().numpy())

                    # 将行号和图片文件名添加到列表中
                    print((i, image_name))
                    self.__text_info.append((i, image_name))
                    

        except FileNotFoundError:
            print(f"File {token_path} not found.")
            
        
        # 将特征和信息转为numpy数组
        self.__text_features = np.stack(self.__text_features)
        

        # 保存特征和信息
        try:
            np.save('text_features.npy', self.__text_features)
            
            with open('text_info.json', 'w') as f:
                for element in self.__text_info:
                    json.dump(element, f)
                    f.write('\n')
        except FileExistsError:
                print("The files already exists.")
            
        return self.__text_features, self.__text_info




    """
    用文本搜图片，获得要搜索的文本信息

    参数:
        text:字符串，要搜索的文本信息
        
    返回：
        img:搜索到的图片名
    """    
    def SearchImgWithText(self, text):
        # 获得文本特征
        self.__text_feature = self.EncodeText(text)
        
        # 计算相似度
        similarities = self.__cal_similarity(self.__image_features, self.__text_feature)
 
        # 将Tensor转回numpy数组以进行排序
        similarities = similarities.detach().cpu().numpy()
        
        # 获取最匹配的结果的索引
        similarities = np.squeeze(similarities)
        # 如果similarities是一个单元素的NumPy数组(只有一个样本时可能发生)
        if isinstance(similarities, np.ndarray) and similarities.size == 1:
            similarities = [similarities.item()]

        top_indices = np.argsort(similarities)[-1]
        
        print(f'Filename: {self.__image_names[top_indices]}, Similarity: {similarities[top_indices]}')
        
        return self.__image_names[top_indices]
        
        
        
        
        
    """
    用图片搜文本，获得要搜索的图片

    参数:
        img_path:要搜索的图片路径
        
    返回：
        text:搜索到的文本在token文件中的行数
    """    
    def SearchTextWithImg(self, img_path):
        
        
        
        # 获得图片特征
        self.__img_feature = self.EncodeImg(img_path)
        
        # 计算相似度
        similarities = self.__cal_similarity(self.__text_features, self.__img_feature)
        
        # 将Tensor转回numpy数组以进行排序
        similarities = similarities.detach().cpu().numpy()
        
        # 获取最匹配的结果的索引
        similarities = np.squeeze(similarities)
         # 如果similarities是一个单元素的NumPy数组(只有一个样本时可能发生)
        if isinstance(similarities, np.ndarray) and similarities.size == 1:
            similarities = [similarities.item()]
            
        top_indices = np.argsort(similarities)[-1]
        
        print(f'Token info: {self.__text_info[top_indices]} , Similarity: {similarities[top_indices]}')
        
        return self.__text_info[top_indices]
        



if __name__ == "__main__":
    # 初始化模型
    Searcher = MultiModel("/home/mmllllyyy/.cache/clip")
    Searcher.LoadImgData("/mnt/c/users/dwc20/pictures/dataset/search/flickr30k/flickr30k-images")
    Searcher.LoadTokenData("/mnt/c/users/dwc20/pictures/dataset/search/flickr30k/flickr30k/results_20130124.token")
    # Searcher.LoadTokenData("/mnt/c/users/dwc20/pictures/dataset/search/flickr30k/test_token/test.token")
    Searcher.SearchImgWithText("A man holds a coffee cup while in the bathroom .")
    Searcher.SearchTextWithImg("/mnt/c/users/dwc20/pictures/dataset/search/flickr30k/flickr30k-images/36979.jpg")
    
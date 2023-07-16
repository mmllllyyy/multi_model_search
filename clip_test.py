import torch
import clip
from PIL import Image

# 检查gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device state: ",device)

# 加载模型
model, transform = clip.load("ViT-B/32", device=device)

# 处理图片
image = transform(Image.open("/mnt/c/users/dwc20/pictures/dataset/search/flickr30k/test_img/36979.jpg")).unsqueeze(0).to(device)

texts = []
text_features = []
normalized_text_features = []
# 处理文字
texts.append(clip.tokenize(["A group of friends playing cards and trying to bluff each other into making a terrible mistake ."]).to(device))
texts.append(clip.tokenize(["A group of college students gathers to play texas hold em poker ."]).to(device))
texts.append(clip.tokenize(["Several men play cards while around a green table ."]).to(device))
texts.append(clip.tokenize(["A group of several men playing poker ."]).to(device))
texts.append(clip.tokenize(["Six white males playing poker ."]).to(device))
texts.append(clip.tokenize(["a lion is eating a rabbit."]).to(device))


# 使用模型
with torch.no_grad():
    image_feature = model.encode_image(image)
    for text in texts:
        text_features.append(model.encode_text(text))
    
# 计算图像和文字之间的相似性
# 首先，对特征进行标准化
image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
for text_feature in text_features:
    normalized_text_feature  = text_feature / text_feature.norm(dim=-1, keepdim=True)
    normalized_text_features.append(normalized_text_feature)


# 然后，计算余弦相似性
for text_feature in normalized_text_features:
    similarity = image_feature @ text_feature.T
    print(similarity)
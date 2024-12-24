import torch
from vit import ViT


# 实例化一个 ViT (Vision Transformer) 模型，用于图像分类任务
v = ViT(
    image_size=256,     # 输入图像的尺寸（高度和宽度），这里为256x256像素
    patch_size=32,      # 每个图像块的尺寸（高度和宽度），这里为32x32像素
    num_classes=1000,   # 分类任务的类别数量，这里为1000类
    dim=1024,           # Transformer 模型的特征维度，这里为1024
    depth=6,            # Transformer 层的数量，这里为6层
    heads=16,           # 注意力头的数量，这里为16个
    mlp_dim=2048,       # 前馈神经网络中隐藏层的维度，这里为2048
    dropout=0.1,        # Transformer 层中的 Dropout 概率，这里为0.1
    emb_dropout=0.1     # 嵌入层中的 Dropout 概率，这里为0.1
)

# 创建一个随机输入张量，形状为 (1, 3, 256, 256)
# 假设输入是一张RGB图像，批次大小为1，通道数为3，图像尺寸为256x256像素
img = torch.randn(1, 3, 256, 256)

# 前向传播，通过 ViT 模型处理输入图像
# 输出 preds 的形状为 (1, 1000)
# 表示模型对输入图像在1000个类别上的预测分数
preds = v(img) # (1, 1000)

# 输出结果
print(f"模型预测结果形状: {preds.shape}")

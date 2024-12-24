import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    """
    将输入转换为元组。如果输入已经是元组，则直接返回；否则返回由输入值重复两次组成的元组。

    参数:
        t: 输入值，可以是任意类型。

    返回:
        Tuple: 输入值转换后的元组。如果输入是元组，则返回原元组；否则返回 (t, t)。
    """
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    """
    FeedForward 类实现了一个前馈神经网络（Feedforward Neural Network）。
    该网络通常用于 Transformer 模型中的前馈部分，由两个线性层组成，中间使用激活函数和 Dropout 层。

    参数说明:
        dim (int): 输入和输出的特征维度。
        hidden_dim (int): 隐藏层的维度。
        dropout (float, 可选): Dropout 层的失活概率，默认为0。
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()

        # 定义前馈网络的层序列
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 第一个层是层归一化，用于稳定输入数据
            nn.Linear(dim, hidden_dim),  # 第一个线性层，将输入维度映射到隐藏层维度
            nn.GELU(),  # 使用高斯误差线性单元（GELU）作为激活函数
            nn.Dropout(dropout),  # Dropout 层，用于防止过拟合
            nn.Linear(hidden_dim, dim),  # 第二个线性层，将隐藏层维度映射回输入维度
            nn.Dropout(dropout)  # 另一个 Dropout 层
        )

    def forward(self, x):
        """
        前向传播方法，执行前馈神经网络的前向计算。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Attention 类实现了一个多头自注意力机制（Multi-Head Self-Attention）。
    该机制广泛应用于 Transformer 模型中，用于捕捉输入序列中不同位置之间的关系。

    参数说明:
        dim (int): 输入和输出的特征维度。
        heads (int, 可选): 注意力头的数量，默认为8。
        dim_head (int, 可选): 每个注意力头的维度，默认为64。
        dropout (float, 可选): Dropout 层的失活概率，默认为0。
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()

        # 计算每个注意力头的内部维度
        inner_dim = dim_head *  heads
        # 判断是否需要投影输出
        project_out = not (heads == 1 and dim_head == dim)

        # 注意力头的数量
        self.heads = heads
        # 缩放因子，用于缩放点积注意力
        self.scale = dim_head ** -0.5

        # 层归一化，用于稳定输入数据
        self.norm = nn.LayerNorm(dim)

        # Softmax 函数，用于计算注意力权重
        self.attend = nn.Softmax(dim = -1)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 线性层，用于将输入投影到查询（q）、键（k）和值（v）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # 如果需要投影输出，则使用线性层和 Dropout；否则使用恒等映射
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        前向传播方法，执行多头自注意力机制的前向计算。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。

        返回:
            Tensor: 输出张量，形状为 (batch_size, sequence_length, dim)。
        """
        # 对输入进行层归一化
        x = self.norm(x)

        # 将输入投影到查询（q）、键（k）和值（v）
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 重塑查询（q）、键（k）和值（v），以适应多头注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # 计算点积注意力
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 应用 Softmax 函数，得到注意力权重
        attn = self.attend(dots)
        # 应用 Dropout
        attn = self.dropout(attn)

        # 计算最终输出
        out = torch.matmul(attn, v)
        # 重塑输出，以适应原始的输入形状
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer 类实现了一个标准的 Transformer 编码器模块。
    Transformer 模型由多个相同的层组成，每个层包含多头自注意力机制和前馈神经网络。
    该类支持指定模型的维度、层数、注意力头数、每个头的维度以及 Dropout 概率。

    参数说明:
        dim (int): 输入和输出的特征维度。
        depth (int): Transformer 层的数量。
        heads (int): 注意力头的数量。
        dim_head (int): 每个注意力头的维度。
        mlp_dim (int): 前馈神经网络中隐藏层的维度。
        dropout (float, 可选): Dropout 层的失活概率，默认为0。
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()

        # 定义层归一化，用于稳定输入数据
        self.norm = nn.LayerNorm(dim)

        # 定义 Transformer 层列表
        self.layers = nn.ModuleList([])

        # 根据指定的深度，添加多个 Transformer 层
        for _ in range(depth):
            # 每个 Transformer 层包含一个多头自注意力机制和一个前馈神经网络
            self.layers.append(nn.ModuleList([
                # 多头自注意力机制
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                # 前馈神经网络
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        """
        前向传播方法，执行 Transformer 编码器的前向计算。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, sequence_length, dim)。
        """
        # 遍历每一个 Transformer 层
        for attn, ff in self.layers:
            # 多头自注意力机制的前向计算，并添加残差连接
            x = attn(x) + x
            # 前馈神经网络的前向计算，并添加残差连接
            x = ff(x) + x
        # 最后进行层归一化
        return self.norm(x)


class ViT(nn.Module):
    """
    ViT (Vision Transformer) 类实现了一个视觉Transformer模型。
    ViT 模型将输入图像分割成固定大小的图像块（patches），然后将这些图像块线性映射到Transformer的输入维度。
    通过添加位置嵌入和分类（CLS）token，模型能够处理图像分类任务。

    参数说明:
        image_size (int 或 tuple): 输入图像的尺寸（高度和宽度）。
        patch_size (int 或 tuple): 图像块的尺寸（高度和宽度）。
        num_classes (int): 分类任务的类别数量。
        dim (int): Transformer 模型的特征维度。
        depth (int): Transformer 层的数量。
        heads (int): 注意力头的数量。
        mlp_dim (int): 前馈神经网络中隐藏层的维度。
        pool (str, 可选): 池化类型，'cls' 表示使用分类token，'mean' 表示对序列进行平均池化。默认为 'cls'。
        channels (int, 可选): 输入图像的通道数，默认为3（RGB图像）。
        dim_head (int, 可选): 每个注意力头的维度，默认为64。
        dropout (float, 可选): Transformer 层中的 Dropout 概率，默认为0。
        emb_dropout (float, 可选): 嵌入层中的 Dropout 概率，默认为0。
    """
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        # 将输入图像尺寸和图像块尺寸转换为元组
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 确保图像尺寸可以被图像块尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算图像块的数量和每个图像块的维度
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 定义图像块嵌入层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), # 重塑张量形状
            nn.LayerNorm(patch_dim), # 层归一化
            nn.Linear(patch_dim, dim), # 线性映射到 Transformer 的维度
            nn.LayerNorm(dim), # 层归一化
        )

        # 定义位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 定义分类 token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 定义嵌入层中的 Dropout
        self.dropout = nn.Dropout(emb_dropout)

        # 定义 Transformer 编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 定义池化类型
        self.pool = pool
        # 定义到潜在空间的映射
        self.to_latent = nn.Identity()

        # 定义分类头
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        """
        前向传播方法，执行 ViT 模型的前向计算。

        参数:
            img (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。

        返回:
            torch.Tensor: 输出分类结果，形状为 (batch_size, num_classes)。
        """
        # 将输入图像嵌入到 Transformer 的维度
        x = self.to_patch_embedding(img)
        # 获取批次大小和图像块数量
        b, n, _ = x.shape

        # 重复分类 token 以匹配批次大小
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # 将分类 token 与图像块嵌入连接
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置嵌入
        x += self.pos_embedding[:, :(n + 1)]
        # 应用嵌入层中的 Dropout
        x = self.dropout(x)

        # 通过 Transformer 编码器
        x = self.transformer(x)

        # 根据池化类型选择分类方式
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # 通过潜在空间映射
        x = self.to_latent(x)
        # 通过分类头得到最终分类结果
        return self.mlp_head(x)

import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    """
    FeedForward 类实现了一个前馈神经网络（Feedforward Neural Network）。
    该网络通常用于 Transformer 模型中的前馈部分，由两个线性层组成，中间使用激活函数和 Dropout 层。
    这种结构也被称为多层感知机（MLP）。

    参数说明:
        dim (int): 输入和输出的特征维度。
        hidden_dim (int): 隐藏层的维度，通常大于输入维度以增加模型的表达能力。
        dropout (float, 可选): Dropout 层的失活概率，用于防止过拟合。默认为0，表示不使用 Dropout。
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()

        # 定义前馈网络的层序列
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 第一层是层归一化，用于稳定输入数据
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
    每个注意力头独立地计算查询（Query）、键（Key）和值（Value），然后将所有头的输出进行合并。

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

        # 定义层归一化，用于稳定输入数据
        self.norm = nn.LayerNorm(dim)
        # 定义 Softmax 函数，用于计算注意力权重
        self.attend = nn.Softmax(dim = -1)
        # 定义 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 定义线性层，用于将输入投影到查询（q）、键（k）和值（v）
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

        # 定义 Transformer 层列表
        self.layers = nn.ModuleList([])

        # 根据指定的深度，添加多个 Transformer 层
        for _ in range(depth):
            # 每个 Transformer 层包含一个多头自注意力机制和一个前馈神经网络
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), # 多头自注意力机制
                FeedForward(dim, mlp_dim, dropout = dropout) # 前馈神经网络
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
        return x


class ViT(nn.Module):
    """
    ViT (Vision Transformer) 类实现了一个视觉 Transformer 模型。
    该模型将输入序列分割成固定大小的块（patches），然后将这些块线性映射到 Transformer 的输入维度。
    通过添加位置嵌入和分类（CLS）token，模型能够处理序列分类任务。

    参数说明:
        seq_len (int): 输入序列的长度。
        patch_size (int): 每个图像块的长度（对于1D序列）。
        num_classes (int): 分类任务的类别数量。
        dim (int): Transformer 模型的特征维度。
        depth (int): Transformer 层的数量。
        heads (int): 注意力头的数量。
        mlp_dim (int): 前馈神经网络中隐藏层的维度。
        channels (int, 可选): 输入序列的通道数，默认为3。
        dim_head (int, 可选): 每个注意力头的维度，默认为64。
        dropout (float, 可选): Transformer 层中的 Dropout 概率，默认为0。
        emb_dropout (float, 可选): 嵌入层中的 Dropout 概率，默认为0。
    """
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # 确保序列长度可以被图像块大小整除
        assert (seq_len % patch_size) == 0

        # 计算图像块的数量和每个图像块的维度
        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        # 定义图像块嵌入层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size), # 重塑张量形状
            nn.LayerNorm(patch_dim), # 层归一化
            nn.Linear(patch_dim, dim), # 线性映射到 Transformer 的维度
            nn.LayerNorm(dim), # 层归一化
        )

        # 定义位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 定义分类 token
        self.cls_token = nn.Parameter(torch.randn(dim))
        # 定义嵌入层中的 Dropout
        self.dropout = nn.Dropout(emb_dropout)

        # 定义 Transformer 编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 定义分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), # 层归一化
            nn.Linear(dim, num_classes) # 线性层映射到类别数量
        )

    def forward(self, series):
        """
        前向传播方法，执行 ViT 模型的前向计算。

        参数:
            series (torch.Tensor): 输入序列张量，形状为 (batch_size, channels, seq_len)。

        返回:
            torch.Tensor: 输出分类结果，形状为 (batch_size, num_classes)。
        """
        # 将输入序列嵌入到 Transformer 的维度
        x = self.to_patch_embedding(series)
        # 获取批次大小和图像块数量
        b, n, _ = x.shape

        # 重复分类 token 以匹配批次大小
        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        # 打包分类 token 和图像块嵌入
        x, ps = pack([cls_tokens, x], 'b * d')

        # 添加位置嵌入
        x += self.pos_embedding[:, :(n + 1)]
        # 应用嵌入层中的 Dropout
        x = self.dropout(x)

        # 通过 Transformer 编码器
        x = self.transformer(x)

        # 解包分类 token
        cls_tokens, _ = unpack(x, ps, 'b * d')

        # 通过分类头得到最终分类结果
        return self.mlp_head(cls_tokens)


if __name__ == '__main__':

    # 实例化一个 ViT (Vision Transformer) 模型
    v = ViT(
        seq_len=256,  # 输入序列的长度
        patch_size=16,  # 每个图像块的长度（对于1D序列）
        num_classes=1000,  # 分类任务的类别数量
        dim=1024,  # Transformer 模型的特征维度
        depth=6,  # Transformer 层的数量
        heads=8,  # 注意力头的数量
        mlp_dim=2048,  # 前馈神经网络中隐藏层的维度
        dropout=0.1,  # Transformer 层中的 Dropout 概率
        emb_dropout=0.1  # 嵌入层中的 Dropout 概率
    )

    # 创建一个随机输入张量，形状为 (4, 3, 256)
    # 假设输入是一个时间序列数据，批次大小为4，通道数为3，序列长度为256
    time_series = torch.randn(4, 3, 256)
    
    # 前向传播，通过 ViT 模型处理输入数据
    # 输出 logits 的形状为 (4, 1000)
    logits = v(time_series)

    # 输出结果
    print(f"输出 logits 的形状: {logits.shape}")

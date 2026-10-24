import math
import torch.nn.functional as F
import torch
from torch import nn
from transformers import PretrainedConfig

# 继承Hugging Face transformers库中的 PretrainedConfig，可无缝使用 Hugging Face 的模型存储、加载、from_pretrained 等工具
class MicroMindConfig(PretrainedConfig):
    model_type = "micromind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
# 标准化层，使用RMSNorm代替LayerNorm
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        # 为每一个特征生成一个weight[i]，weight为1*n的矩阵，进行乘法操作时向右对齐，自动补全前两维
        self.weight=nn.Parameter(torch.ones(dim))
    # 在 Transformer 模型中，一个批次的数据通常表示为[batch_size, seq_len, d_model]
    # 归一化（无论是 LayerNorm 还是 RMSNorm）的目标是：让每个 token 的特征向量具有稳定的分布，
    # 即特征维度上的各个分量被重新缩放，使它们的均方根（或均值/方差）标准化。
    # 因此，归一化操作必须沿着特征维度（即最后一维 dim=-1） 进行统计计算（均值、方差、RMS 等）
    # 写 -1 的好处是：无论输入是二维、三维还是四维，都能直接指定最后一维。
    # 并把这个维度保留为大小 1，方便后续进行广播运算。
    def norm(self,x):
        return x*torch.rsqrt_(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x):
        return (self.weight * self.norm(x.float())).type_as(x)
# 通过YaRN算法实现对旋转位置编码（RoPE）频率的预计算与扩展,按照低中高维分类讨论
# 返回的freqs_cos 和 freqs_sin 的形状均为 (end, dim)
# 预先生成 RoPE 使用的 cos、sin 表，还没有用到旋转矩阵
# | `dim` | 每个注意力头参与旋转的维度,偶数
# | `end` | 提前计算多少个位置
# | `rope_base` | 控制各组旋转频率的分布
# | `rope_scaling` | 可选的 YaRN 配置
# 返回的是每个token 每个token的每一维度应旋转的sin和cos值
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 计算基础角频率公式,**幂运算
    freqs = 1.0 / (
            rope_base ** (torch.arange(0, dim, 2).float() / dim)
    )
    # 温度因子，调整旋转结果的幅度，影响注意力分数的尺度，放大注意力分数，softmax更容易集中在高分位置
    attn_factor=1.0
    # 加载核心参数
    # YaRN核心公式，将原始频率f(i) 通过一个线性插值因子 γ（范围 0~1）和缩放因子s（即 factor）来修改。
    if rope_scaling is not None:
        # 原始支持的最大位置数,缩放频率,高频低维阈值,低频高维阈值,温度因子
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0)
        )
        # 超过预设范围才会使用YaRN外推,YaRN的核心就是就将原本的freqs划分成低中高频，重新计算其freqs
        if end / orig_max > 1.0:
                    # 计算波长b对应的维度,匿名函数，波长=2π/θ越大,θ越小,频率越低,维度越高
                    # inv_dim表示：某一对分量在“原上下文长度”内，恰好旋转 b 圈，那么它应该是第几对分量？inv_dim(32)：大约哪一组在原长度内转 32 圈？
                    inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
                    # 计算高低维边界，即转32圈和转1圈的组数
                    low, high = (max(math.floor(inv_dim(beta_fast)), 0),
                                 min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1))
                    # 计算ramp缩放程度,低于low时为负,low~high之前线性增长，大于hign即大于1
                    # 使用clamp将值裁剪到0 1区间
                    # 小于 0 → 设为 0
                    # 大于 1 → 设为 1
                    # 其余值 → 保持原样
                    #因为我们希望得到一个 **“当前分组走到了过渡区间的百分之多少” ** 的数值：
                    #- 到达低边界时，进度是0
                    #- 到达高边界时，进度是1
                    #- 中间的位置，线性从0增加到1。
                    # PyTorch要求参与计算的张量在同一设备
                    ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
                    #计算新频率公式
                    freqs = freqs * (1 - ramp + ramp / factor)
    # 从0开始的位置编号，提前计算的位置，计算要旋转的n*频率
    t = torch.arange(end, device=freqs.device)
    # 计算每个位置、每对分量的旋转角度，freqs为一维数组长为dim/2，t为一维数组长度为end，结果freqs为end*dim/2长度的数组
    freqs = torch.outer(t, freqs).float()
    # 生成cos、sin查找表，torch.cos sin为表里每个值计算正余弦，为什么拼接两遍：
    # 在此代码中用的是前半段与后半段相匹配的方式，即一个token为[1,2,3,4,5,6],1 4为一组，25 36为一组，没有使用相邻匹配，torch实现更方便
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin



# 一个q对应多个k、v,如q大小为32*n,k大小为8*n,k需要扩展为32*n
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    # 只需要将KV头逻辑上复制n_rep倍,以便与Q的头数对齐,但并不需要提前产生物理上重复的数据
    return (x[:, :, :, None, :].
            expand(bs, slen, num_key_value_heads, n_rep, head_dim).
            reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
# 旋转位置编码
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # ...表示保留前面所有维度，只对最后一维做切片；x.shape[-1] // 2: 表示取最后一维的一半，并且从一半开始切一直到末尾
    # [1,2,3,4]处理后变为[-3,-4,1,2]，q*cos+rotate_half*sin=[cos-3sin 2cos-4sin 3cos+sin 4cos+2sin],第13项拿出来
    # 即为[cos-3sin 3cos+sin]相当于[1,3]*旋转矩阵R[[cos,-sin],[sin,cos]]
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # unsqueeze 的意思是：给张量插入一个大小为 1 的维度。数值不变，只改变形状，参数表示在哪里添加这个维度，从0开始
    # [2,4].unsqueeze(1)之后维度变为[2,1,4],unsqueeze(1)=[1,2,4]
    # 假设q为[2,3,8,6]=>[batch数, token数, 头数, 每头特征数]2条序列 3个token 8个注意力头 嵌入维度为6,
    # 原本cos为[3,6],cos.unsqueeze后变为[3,1,6],即同一个token的8个头，都使用该位置的同一行cos。
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed
class Attention(nn.Module):
    def __init__(self, config: MicroMindConfig):
        super().__init__()
        # 未指定num_key_value_heads即退化为标准MHA
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        # q头
        self.n_local_heads = config.num_attention_heads
        # kv头
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = config.head_dim
        self.is_causal = True
        # Wq、Wk、Wv,得到Q K V 矩阵,如Q的大小为n*维度，(n*hidden_size)*(hidden_size*维度)=n*维度,q_proj表示的就是hidden_size*维度矩阵
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # output拼接,将维度从config.num_attention_heads * self.head_dim变回hidden_size
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # 归一化
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        # 通过投影得到Q K V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 拆分最后一维
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # 来自precompute_freqs_cis预计算好的正余弦编码
        cos, sin = position_embeddings
        #RoPE
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # KV 缓存处理(增量推理)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2),
                      repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (
                attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0,
                                                    is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"),
                                                                        device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

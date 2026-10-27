import math
import torch.nn.functional as F
import torch
from sympy.parsing.sympy_parser import null
from torch import nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


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

# 旋转位置编码
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # ...表示保留前面所有维度，只对最后一维做切片；x.shape[-1] // 2: 表示取最后一维的一半，并且从一半开始切一直到末尾
    # 某一行[q1,q2,q3,q4]处理后变为[-q3,-q4,q1,q2]，q*cos+rotate_half*sin：1 3列为一组，旋转角r相同，只看1 3列：=[q1cosr-q3sinr  q3cosr+q1sinr],第13项拿出来
    # 相当于旋转矩阵R[[cos,-sin],[sin,cos]]*[[q1],[q3]]
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # unsqueeze 的意思是：给张量插入一个大小为 1 的维度。数值不变，只改变形状，参数表示在哪里添加这个维度，从0开始
    # [2,4].unsqueeze(1)之后维度变为[2,1,4],unsqueeze(1)=[1,2,4]
    # 假设q为[2,3,8,6]=>[batch数, token数, 头数, 每头特征数]2条序列 3个token 8个注意力头 嵌入维度为6,
    # 原本cos为[3,6],cos.unsqueeze后变为[3,1,6],即同一个token的8个头，都使用该位置的同一行cos。
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


# 在gqa中，一个q对应多个k、v以减小内存压力
# 方法接收的 x 是 K 或 V 张量，形状为：[批次大小, token 数量, KV 头数, 每个头的维度]
# 如x.shape = [2, 10, 4, 96]表示：一批有 2 条序列，每条有 10 个 token，每个 token 有 4 个 KV 头，每个头是 96 维向量。
# n_rep表示一个q对应多少kv
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    # x[:, :, :, None, :]增加一维，None 表示在这个位置增加一个大小为 1 的维度。；
    # expand将增加的一维扩展为n_rep大小,但expand本身共享原来的存储，不会实际复制这几份数据。
    # reshape合并
    return (x[:, :, :, None, :].
            expand(bs, slen, num_key_value_heads, n_rep, head_dim).
            reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


class Attention(nn.Module):
    def __init__(self, config: MicroMindConfig):
        super().__init__()
        # 未指定num_key_value_heads即退化为标准MHA,一个q对应一个kv
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        # q头
        self.n_local_heads = config.num_attention_heads
        # kv头
        self.n_local_kv_heads = self.num_key_value_heads
        # kv需要复制的个数,n_rep个q共享kv
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度=hiddenSize/q头数即num_attention_heads
        self.head_dim = config.head_dim
        # 是否开启因果掩码，掩盖后续token对之前token的影响
        self.is_causal = True
        # 参数，in_feature（列数） out_feature（输出的行数） 是否使用可训练的偏置值
        # Linear对最后一维进行变换保留前面的维度，因此输入q[batch_size,seq_length,hidden_size]经过变换后得到[batch_size,seq_length,linear中的out_feature]
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出投影，学会如何组合不同头的信息.前面的拼接只是把各个头的结果放在一起；o_proj 通过可学习的矩阵，把不同头的信息重新组合
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # 归一化
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
    #params：
    #x：RMSNorm处理后后的输入，x.shape=[batch_size, seq_len, hidden_size]
    # position_embeddings:已处理好的正余弦编码
    # past_key_value:过去的kv_cache,默认无
    # use_cache:是否返回更新后的kv_cache
    # attention_mask:是否使用padding掩码
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        #batch_size sequence_length,x=>[batch_size, seq_len, hidden_size]
        bsz, seq_len, _ = x.shape
        # 通过投影得到Q K V，Q.shape=[batch_size,seq_len,config.num_attention_heads * self.head_dim]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 拆分最后一维成为多个头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        #Q、K 的数值大小会影响 softmax 的注意力分布，而 V 不参与 softmax，它负责提供被加权的信息，因此只对 Q、K 额外归一化。
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # 预先计算余弦正弦编码
        cos, sin = position_embeddings
        #应用RoPE，YaRN可选项
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # KV 缓存处理(增量推理)，dim=1 表示沿着序列长度维度拼接。
        # past_key_value是一个二元组,past_key_value = (
        #     历史K,
        #     历史V,
        # )
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # 后面的@会对张量的最后两个维度做矩阵乘法，希望计算的是“同一个头中，不同 token 之间的匹配分数”。
        # 现在的维度是[batch_size,seq_length,head,hidden_size],使用transpose=>[batch_size,head,seq_length,hidden_size]
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2),
                      repeat_kv(xv, self.n_rep).transpose(1, 2))
        # 使用Attention注意力计算公式计算，k先转置最后两维，前面维度相同会广播
        scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
        # 前一个屏蔽未来 token
        # 加上因果掩码，torch.full((seq_len, seq_len), float("-inf"),device=scores.device).triu(1)生成负无穷的矩阵后再保留右上角
        if self.is_causal:
            # 为什么只选最后 seq_len 列？K 已经拼接了历史缓存，而 seq_len 是本轮输入的 token 数量，不包含历史缓存。
            # 只对本轮新增的q*kt部分加因果掩码
            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"),device=scores.device).triu(1)
        # 屏蔽用来补齐长度的PAD，一个 batch 中，不同文本的长度可能不同：
        # 第一条：A B C D
        # 第二条：E F，一个batch中，不同文本的长度可能不同，为了放进同一个形状规则的张量，需要补齐。例如在左边添加特殊的PAD token，PAD只是占位，不是真正的文本内容。但它也会经过Embedding和Linear，产生K、V，不会天然被注意力忽略。
        if attention_mask is not None:
            # 加一个padding掩码矩阵，attention_mask初始大小为[batch_size,seq_len],unsqueeze扩大后成为[batch_size,1,1,seq_len]，seq_leng为kv序列总长度
            # score大小为[bs,attention_head,seq_len,完整长度(缓存kv的seq+本轮seq)]
            # 相加时，会在头维度和查询位置维度广播。attention_mask中为0的数表示padding位置，score会加上负无穷
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        #output[bs,a_h,seq_len,hidden_size]=>output[bs,seq_len,a_h,hidden_size]=>output[bs,seq_len,所有的hidden_size]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        #输出投影，学会如何组合不同头的信息
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
# FFN层，引入非线性能力
# 输入为Attention处理后的结果,output[bs,seq_len,hidden_size]
class FeedForward(nn.Module):
    def __init__(self, config: MicroMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MicroMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MicroMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)
    # hidden_states	本轮输入的 token 向量，形状 [B,T,768]
    # position_embeddings	本轮位置对应的 (cos, sin)
    # past_key_value	当前这个 Block 的历史 (K,V)
    # use_cache	是否返回更新后的 KV cache
    # attention_mask	标记有效位置与 padding 的掩码
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 残差，保存原有数据
        residual = hidden_states
        #Attention层处理
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # 残差连接
        hidden_states += residual
        # 残差连接+FFN
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        #返回的是通过该层后的状态及kv cache
        return hidden_states, present_key_value
#创建 Embedding、多个 Block、最终 RMSNorm 和 RoPE 表，然后在 forward 中按顺序调用它们。
class MicroMindModel(nn.Module):
    def __init__(self, config: MicroMindConfig):
        super().__init__()
        self.config = config
        #模型词表大小与隐藏层个数，一共给6400个词打分
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        #创建词嵌入矩阵，随训练一起更新
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        #　attention + ffn为一层，layers一共8层
        self.layers = nn.ModuleList([MicroMindBlock(l, config)
                                     for l in range(self.num_hidden_layers)])
        #这是八个 Block 全部执行完之后使用的 Norm，与每个 Block 内部的 Norm 是不同实例。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #所有attention层共享旋转角度，但是旋转后结果不同由各自的qkv控制
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        # self.register_buffer将cos和sin表交给pytorch统一管理,运行时保留并使用，随模型迁移设备；保存 state_dict() 时省略，因为是直接按照公式计算出来的不用保存，重新创建模型时再计算。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # input_ids:传入token_id及在词表中的编号，[batch_size, seq_length]，此时还没做嵌入
    # attention_mask:padding掩码，0表示无效，对应位置注意力分数会被置于负无穷
    # past_key_values；各层kv cache缓存
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        # 识别并丢弃当前实现没有适配的缓存格式，pytorch存储kvcache的属性有layers，识别到直接丢弃
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # 第一次没有历史缓存时，得到[
        #     None,  # Block 0
        #     None,  # Block 1
        #     None,  # Block 2
        #     None,  # Block 3
        #     None,  # Block 4
        #     None,  # Block 5
        #     None,  # Block 6
        #     None,  # Block 7
        # ]
        past_key_values = past_key_values or [None] * len(self.layers)
        # 确定：本轮输入的 token，位置编号应该从哪里开始。past_key_values[0][0]即缓存的key，shape为[bs,sl,kv,hidden_size]，这里是在取sequence length
        # 为什么要取：因为 RoPE 表提前存好了很多位置的旋转系数，而本轮只需要用到当前 token 对应的几行。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 取对应位置的sin cos表
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        # zip() 的作用是：把多个序列中相同位置的元素配成一组，将layer层 以及对应的past kv cache配置成一组
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents

#标准化封装模型，PreTrainedModel 提供模型配置、保存和加载等框架能力；GenerationMixin 提供生成相关接口。
class MicroMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MicroMindConfig
    def __init__(self, config: MicroMindConfig = None):
        self.config = config or MicroMindConfig()
        super().__init__(self.config)
        self.model = MicroMindModel(self.config)
        # 语言层映射，将隐藏向量映射为词表中各 token 的原始分数 logits
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享，输出层和输入层的词嵌入矩阵共享
        self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()
    # logits_to_keep:要保留的logits数量，默认为0
    # labels:正确答案，用于计算loss，二维矩阵，batch*seq_length
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0,labels=None,
                 **kwargs):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs)
        # 经过 lm_head 后，每个位置都会得到一组词表分数，如序列ABCD：
        # hA → 用于预测 B
        # hB → 用于预测 C
        # hC → 用于预测 D
        # hD → 用于预测下一个 token
        # 训练时，我们希望同时训练多个位置的预测，所以通常需要保留全部位置的 logits。
        # 生成下一个 token 时，只需要最后一个位置 hD 对应的 logits，就可以选择 D 后面接什么。
        # 因此，可以只计算：self.lm_head(hD)
        # 而不必把 hA、hB、hC 都送入输出层。这能减少输出层的计算和 logits 占用的内存，尤其词表很大时。
        # 如果传入的logits_to_keep是int类型即数字，就保留后logits_to_keep个数字（0表示无规则，-1表示取最后1个）；如果传入的是规则则使用
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        # 计算训练误差，logits先去掉最后一个预测，labels去掉第一个正确答案，因为位置0的词表分数做的是第一个词的预测
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # shift_logits和shift_labels都被展平，shift_logits形状为[bs*se,词表大小],shift_labels大小为[bs*se]
            # F.cross_entropy计算得的是正确答案的概率对应的损失，等价于:
            # 1. 计算每个 token 的对数概率
            # log_probs = F.log_softmax(logits, dim=-1)
            # # 2. 根据 labels 取出正确 token 的对数概率，取负并求平均
            # loss = F.nll_loss(log_probs, labels, ignore_index=-100)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
        return output




import math

import torch
from torch import nn
from transformers import PretrainedConfig

# 继承Hugging Face transformers库中的 PretrainedConfig，可无缝使用 Hugging Face 的模型存储、加载、from_pretrained 等工具
class MicroMindConfig(PretrainedConfig):
    model_type = "micromind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
# 标准化层，使用RMSNorm代替LayerNorm
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    # 在 Transformer 模型中，一个批次的数据通常表示为[batch_size, seq_len, d_model]
    # 归一化（无论是 LayerNorm 还是 RMSNorm）的目标是：让每个 token 的特征向量具有稳定的分布，
    # 即特征维度上的各个分量被重新缩放，使它们的均方根（或均值/方差）标准化。
    # 因此，归一化操作必须沿着特征维度（即最后一维 dim=-1） 进行统计计算（均值、方差、RMS 等）
    def norm(self,x):
        return x*torch.rsqrt_(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x):
        return (self.weight * self.norm(x.float())).type_as(x)
#通过YaRN算法实现对旋转位置编码（RoPE）频率的预计算与扩展
# dim:模型 hidden size（通常是 attention head 的维度，必须是偶数）
# end:预计算的最大位置索引，默认 32768。
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 计算基础角频率,**幂运算,//地板除法
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 温度因子
    attn_factor=1.0
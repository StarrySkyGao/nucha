#pragma once

#include "common.h"
#include "Tensor.h"
#include "Module.h"
#include "Linear.h"
#include "layernorm.h"
#include <pybind11/functional.h>
namespace pybind11 {
class function;
}

// 枚举类，表示注意力机制的实现方式
enum class AttentionImpl {
    // FlashAttention2实现方式
    FlashAttention2 = 0,
    // NunchakuFP16实现方式
    NunchakuFP16,
};

class AdaLayerNormZeroSingle : public Module {
public:
    // 定义是否使用4位
    static constexpr bool USE_4BIT = true;
    // 根据是否使用4位，选择不同的GEMM类型
    using GEMM                     = std::conditional_t<USE_4BIT, GEMV_AWQ, GEMM_W8A8>;

    // 定义输出结构体
    struct Output {
        Tensor x;
        Tensor gate_msa;
    };

public:
    // 构造函数，传入维度、数据类型和设备
    AdaLayerNormZeroSingle(int dim, Tensor::ScalarType dtype, Device device);
    // 前向传播函数，传入输入张量和嵌入张量
    Output forward(Tensor x, Tensor emb);

public:
    // 维度
    const int dim;

private:
    // GEMM类型
    GEMM linear;
    // LayerNorm类型
    LayerNorm norm;
};

class AdaLayerNormZero : public Module {
public:
    // 定义是否使用4位
    static constexpr bool USE_4BIT = true;
    // 根据是否使用4位，选择不同的GEMM类型
    using GEMM                     = std::conditional_t<USE_4BIT, GEMV_AWQ, GEMM_W8A8>;

    // 定义输出结构体
    struct Output {
        Tensor x;
        Tensor gate_msa;
        Tensor shift_mlp;
        Tensor scale_mlp;
        Tensor gate_mlp;
    };

public:
    // 构造函数
    AdaLayerNormZero(int dim, bool pre_only, Tensor::ScalarType dtype, Device device);
    // 前向传播函数
    Output forward(Tensor x, Tensor emb);

public:
    // 维度
    const int dim;
    // 是否只使用前向传播
    const bool pre_only;

private:
    // GEMM线性层
    GEMM linear;
    // LayerNorm层
    LayerNorm norm;
};

class Attention : public Module {
public:
    // 定义池化大小
    static constexpr int POOL_SIZE = 128;

    // 构造函数，初始化num_heads和dim_head
    Attention(int num_heads, int dim_head, Device device);
    // 前向传播函数，输入qkv张量
    Tensor forward(Tensor qkv);
    // 前向传播函数，输入qkv和pool_qkv张量，以及稀疏度比例
    Tensor forward(Tensor qkv, Tensor pool_qkv, float sparsityRatio);

    // 设置强制使用FP16的函数
    static void setForceFP16(Module *module, bool value);

public:
    // 定义头数
    const int num_heads;
    // 定义头维度
    const int dim_head;
    // 是否强制使用FP16
    bool force_fp16;

private:
    // 定义cu_seqlens_cpu张量
    Tensor cu_seqlens_cpu;
    // 定义headmask_type张量
    Tensor headmask_type;
};

class FluxSingleTransformerBlock : public Module {
public:
    // 定义是否使用4位浮点数的常量
    static constexpr bool USE_4BIT = true;
    // 根据是否使用4位浮点数，选择不同的GEMM类型
    using GEMM                     = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;

    // 构造函数，初始化参数
    FluxSingleTransformerBlock(int dim,
                               int num_attention_heads,
                               int attention_head_dim,
                               int mlp_ratio,
                               bool use_fp4,
                               Tensor::ScalarType dtype,
                               Device device);
    // 前向传播函数，输入hidden_states、temb、rotary_emb，输出Tensor
    Tensor forward(Tensor hidden_states, Tensor temb, Tensor rotary_emb);

public:
    // 定义参数
    const int dim;
    const int dim_head;
    const int num_heads;
    const int mlp_hidden_dim;

    // 定义AttentionImpl类型
    AttentionImpl attnImpl = AttentionImpl::FlashAttention2;

private:
    // 定义AdaLayerNormZeroSingle类型
    AdaLayerNormZeroSingle norm;
    // 定义GEMM类型
    GEMM mlp_fc1;
    GEMM mlp_fc2;
    GEMM qkv_proj;
    // 定义RMSNorm类型
    RMSNorm norm_q, norm_k;
    // 定义Attention类型
    Attention attn;
    // 定义GEMM类型
    GEMM out_proj;
};

class JointTransformerBlock : public Module {
public:
    static constexpr bool USE_4BIT = true;
    using GEMM                     = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;

    JointTransformerBlock(int dim,
                          int num_attention_heads,
                          int attention_head_dim,
                          bool context_pre_only,
                          bool use_fp4,
                          Tensor::ScalarType dtype,
                          Device device);
    std::tuple<Tensor, Tensor> forward(Tensor hidden_states,
                                       Tensor encoder_hidden_states,
                                       Tensor temb,
                                       Tensor rotary_emb,
                                       Tensor rotary_emb_context,
                                       float sparsityRatio);

public:
    const int dim;
    const int dim_head;
    const int num_heads;
    const bool context_pre_only;
    AdaLayerNormZero norm1;

    AttentionImpl attnImpl = AttentionImpl::FlashAttention2;

private:
    AdaLayerNormZero norm1_context;
    GEMM qkv_proj;
    GEMM qkv_proj_context;
    RMSNorm norm_q, norm_k;
    RMSNorm norm_added_q, norm_added_k;
    Attention attn;
    GEMM out_proj;
    GEMM out_proj_context;
    LayerNorm norm2;
    LayerNorm norm2_context;
    GEMM mlp_fc1, mlp_fc2;
    GEMM mlp_context_fc1, mlp_context_fc2;
};

class FluxModel : public Module {
public:
    FluxModel(bool use_fp4, bool offload, Tensor::ScalarType dtype, Device device);
    Tensor forward(Tensor hidden_states,
                   Tensor encoder_hidden_states,
                   Tensor temb,
                   Tensor rotary_emb_img,
                   Tensor rotary_emb_context,
                   Tensor rotary_emb_single,
                   Tensor controlnet_block_samples,
                   Tensor controlnet_single_block_samples,
                   bool skip_first_layer = false);
    std::tuple<Tensor, Tensor> forward_layer(size_t layer,
                                             Tensor hidden_states,
                                             Tensor encoder_hidden_states,
                                             Tensor temb,
                                             Tensor rotary_emb_img,
                                             Tensor rotary_emb_context,
                                             Tensor controlnet_block_samples,
                                             Tensor controlnet_single_block_samples);
    void setAttentionImpl(AttentionImpl impl);

    void set_residual_callback(std::function<Tensor(const Tensor &)> cb);

public:
    const Tensor::ScalarType dtype;

    std::vector<std::unique_ptr<JointTransformerBlock>> transformer_blocks;
    std::vector<std::unique_ptr<FluxSingleTransformerBlock>> single_transformer_blocks;

    std::function<Tensor(const Tensor &)> residual_callback;

private:
    bool offload;
};

# mistral.rs 二次开发指南

> 本文档面向需要对 mistral.rs 进行二次开发的工程师，详细说明了代码架构、核心模块、执行流程，以及 CPU/NPU 扩展支持的实现路径。

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. Workspace 结构与 Crate 依赖关系](#2-workspace-结构与-crate-依赖关系)
- [3. 核心模块详解](#3-核心模块详解)
  - [3.1 Pipeline 架构](#31-pipeline-架构)
  - [3.2 Engine 引擎与请求处理](#32-engine-引擎与请求处理)
  - [3.3 Scheduler 调度器](#33-scheduler-调度器)
  - [3.4 模型加载流程](#34-模型加载流程)
  - [3.5 Attention 注意力机制](#35-attention-注意力机制)
  - [3.6 KV Cache 管理](#36-kv-cache-管理)
  - [3.7 PagedAttention 分页注意力](#37-pagedattention-分页注意力)
  - [3.8 量化系统 (mistralrs-quant)](#38-量化系统-mistralrs-quant)
- [4. 硬件抽象层与设备管理](#4-硬件抽象层与设备管理)
  - [4.1 Candle 张量框架](#41-candle-张量框架)
  - [4.2 设备映射 (DeviceMapper)](#42-设备映射-devicemapper)
  - [4.3 Feature Flags 特性门控](#43-feature-flags-特性门控)
- [5. CPU 后端实现详解](#5-cpu-后端实现详解)
  - [5.1 CPU 上的张量运算](#51-cpu-上的张量运算)
  - [5.2 CPU 上的量化支持](#52-cpu-上的量化支持)
  - [5.3 CPU 上的注意力计算](#53-cpu-上的注意力计算)
  - [5.4 CPU 性能优化选项](#54-cpu-性能优化选项)
- [6. NPU 扩展开发指南](#6-npu-扩展开发指南)
  - [6.1 架构概览：从 CUDA/Metal 模式学习](#61-架构概览从-cudametal-模式学习)
  - [6.2 第一步：扩展 Candle 张量框架](#62-第一步扩展-candle-张量框架)
  - [6.3 第二步：实现 NPU 量化内核](#63-第二步实现-npu-量化内核)
  - [6.4 第三步：实现 NPU PagedAttention](#64-第三步实现-npu-pagedattention)
  - [6.5 第四步：设备映射集成](#65-第四步设备映射集成)
  - [6.6 第五步：构建系统集成](#66-第五步构建系统集成)
  - [6.7 第六步：Feature Flag 注册](#67-第六步feature-flag-注册)
  - [6.8 渐进式实现策略](#68-渐进式实现策略)
- [7. 对外接口层](#7-对外接口层)
  - [7.1 CLI 命令行](#71-cli-命令行)
  - [7.2 HTTP Server (OpenAI 兼容)](#72-http-server-openai-兼容)
  - [7.3 Rust SDK](#73-rust-sdk)
  - [7.4 Python SDK (PyO3)](#74-python-sdk-pyo3)
- [8. 添加新模型架构](#8-添加新模型架构)
- [9. 构建与测试](#9-构建与测试)
- [10. 关键文件索引](#10-关键文件索引)

---

## 1. 项目概览

mistral.rs 是一个用 Rust 编写的高性能 LLM 推理引擎，基于 Hugging Face 的 [Candle](https://github.com/huggingface/candle) 张量框架构建。它支持：

- **文本生成**：27+ 种模型架构（Llama, Mistral, Qwen3, Gemma, DeepSeek, Phi 等）
- **视觉模型**：18+ 种视觉语言模型（LLaVA, Qwen2-VL, Phi3-Vision 等）
- **图像生成**：Flux 等扩散模型
- **语音合成**：TTS 模型支持
- **嵌入模型**：文本向量化

硬件支持情况：

| 后端 | 状态 | Feature Flag |
|------|------|-------------|
| CPU | 完整支持 | 默认（无需额外 flag） |
| CUDA (NVIDIA GPU) | 完整支持 | `cuda`, `flash-attn`, `cudnn` |
| Metal (Apple GPU) | 完整支持 | `metal` |
| Apple Accelerate | CPU 加速 | `accelerate` |
| Intel MKL | CPU 加速 | `mkl` |
| NCCL (多卡) | 分布式推理 | `nccl` |

---

## 2. Workspace 结构与 Crate 依赖关系

```
mistral.rs/
├── mistralrs-core/          # 核心推理引擎：模型、Pipeline、Engine、调度器
├── mistralrs-quant/         # 量化实现：GGUF, GPTQ, HQQ, FP8, BnB, AFQ, MXFP4
├── mistralrs-paged-attn/    # PagedAttention：CUDA/Metal 分页注意力内核
├── mistralrs-vision/        # 视觉处理：图像预处理
├── mistralrs-audio/         # 音频处理：编解码、重采样、FFT
├── mistralrs-mcp/           # Model Context Protocol 客户端
├── mistralrs-macros/        # 过程宏
├── mistralrs/               # Rust SDK：高层 Builder API
├── mistralrs-cli/           # CLI 二进制：serve, run, bench, quantize 等
├── mistralrs-server-core/   # HTTP 服务器：OpenAI 兼容 API, Axum 路由
├── mistralrs-server/        # HTTP 服务器入口
├── mistralrs-pyo3/          # Python SDK：PyO3 绑定
├── mistralrs-bench/         # 基准测试（已废弃，使用 CLI bench）
└── mistralrs-web-chat/      # Web 聊天 UI
```

### Crate 依赖关系图

```
                        mistralrs-cli
                       /      |       \
              mistralrs   mistralrs-server-core
                  |            |
                  +------+-----+
                         |
                   mistralrs-core  ←─── 核心 Hub
                  /    |    |    \
    mistralrs-quant    |    |   mistralrs-paged-attn
         |             |    |         |
    candle-core    mistralrs-vision   |
    candle-nn      mistralrs-audio    |
                   mistralrs-mcp   candle-core
```

### 版本信息

- **版本**: 0.7.0
- **Rust 最低版本**: 1.88
- **Candle 版本**: 0.9.2

---

## 3. 核心模块详解

### 3.1 Pipeline 架构

Pipeline 是整个推理引擎的核心抽象，定义在 `mistralrs-core/src/pipeline/mod.rs`。

#### Pipeline Trait 定义

```rust
#[async_trait]
pub trait Pipeline: Send + Sync +
    PreProcessingMixin +      // 聊天模板、输入预处理
    IsqPipelineMixin +        // 即时量化 (ISQ) 支持
    CacheManagerMixin +       // KV Cache 管理
    MetadataMixin +           // 设备、分词器、元数据
    AnyMoePipelineMixin       // MoE 支持
{
    // 核心前向传播
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult>;

    // 单步推理（prompt 或 completion 阶段）
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
    ) -> Result<Duration>;

    // 因果生成采样
    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<()>;

    fn category(&self) -> ModelCategory;
}
```

#### Pipeline 实现种类

| Pipeline 类型 | 文件 | 说明 |
|--------------|------|------|
| `NormalPipeline` | `pipeline/normal.rs` | 标准文本模型 |
| `GGUFPipeline` | `pipeline/gguf.rs` | GGUF 量化模型 |
| `GGMLPipeline` | `pipeline/ggml.rs` | GGML 量化模型 |
| `VisionPipeline` | `pipeline/vision.rs` | 视觉语言模型 |
| `EmbeddingPipeline` | `pipeline/embedding.rs` | 嵌入模型 |
| `DiffusionPipeline` | `pipeline/diffusion.rs` | 扩散模型 |
| `SpeechPipeline` | `pipeline/speech.rs` | 语音模型 |
| `SpeculativePipeline` | `pipeline/speculative.rs` | 投机解码 |

#### 前向传播输出类型

```rust
pub enum ForwardInputsResult {
    CausalGeneration { logits: Tensor },           // 文本生成
    Embeddings { embeddings: Tensor },              // 文本嵌入
    RawLogits { logits: Tensor },                   // 原始 logits
    Image { images: Vec<DynamicImage> },            // 图像生成
    Speech { pcms, rates, channels },               // 语音合成
}
```

### 3.2 Engine 引擎与请求处理

Engine 定义在 `mistralrs-core/src/engine/mod.rs`，是异步请求处理的核心。

#### 核心结构

```rust
pub struct Engine {
    tx: Sender<Request>,                          // 请求发送端
    rx: Arc<Mutex<Receiver<Request>>>,            // 请求接收端
    pipeline: Arc<Mutex<dyn Pipeline>>,           // 模型 Pipeline
    scheduler: Arc<Mutex<dyn Scheduler>>,         // 调度器
    prefix_cacher: Arc<Mutex<PrefixCacheManagerV2>>, // 前缀缓存
    // ...
}
```

#### 多模型管理

`MistralRs` 结构体管理多个 Engine，支持运行时模型热加载/卸载：

```rust
pub struct MistralRs {
    engines: RwLock<HashMap<String, EngineHandle>>,    // 模型ID → Engine
    unloaded_models: RwLock<HashMap<String, UnloadedModel>>,
    model_aliases: RwLock<HashMap<String, String>>,
    default_engine_id: RwLock<String>,
    // ...
}
```

**锁获取顺序**（防止死锁）：`reloading_models` → `engines` → `unloaded_models` → `default_engine_id` → `model_aliases`

#### 请求处理完整流程

```
用户请求 (HTTP/SDK/CLI)
    │
    ├─> MistralRs::send_request(Request)
    │   ├─ 解析模型 ID（默认或指定）
    │   └─ 获取目标 Engine 的 Sender
    │
    ├─> Engine 通过 MPSC 通道接收 Request
    │   ├─ Request::Normal(NormalRequest)   → 正常推理
    │   ├─ Request::Tokenize               → 分词
    │   ├─ Request::Detokenize             → 反分词
    │   ├─ Request::ReIsq                  → 重新量化
    │   └─ Request::Terminate              → 终止
    │
    ├─> Engine::handle_request()
    │   ├─ 解析请求类型
    │   ├─ 如果是 Normal 请求:
    │   │   ├─ 创建 Sequence 对象
    │   │   ├─ 通过聊天模板处理消息
    │   │   ├─ 使用 tokenizer 分词
    │   │   └─ 加入 Scheduler 的等待队列
    │   └─ 如果是 Tokenize/Detokenize: 直接处理并返回
    │
    ├─> Scheduler::schedule() → SchedulerOutput
    │   ├─ 从等待队列选择序列
    │   ├─ 分为 Prompt 序列和 Completion 序列
    │   └─ 返回调度结果和元数据
    │
    ├─> Engine::run() 主循环
    │   ├─ 缓存预操作 (CacheInstruction::In/Reset/Out)
    │   │
    │   ├─ Prompt 阶段:
    │   │   ├─ 输入处理器创建张量
    │   │   ├─ Pipeline::forward_inputs() → 模型前向传播
    │   │   ├─ 提取最终 logits
    │   │   ├─ 采样下一个 token
    │   │   └─ 状态转换为 RunningCompletion
    │   │
    │   ├─ Completion 阶段 (逐 token 生成):
    │   │   ├─ 单 token 张量输入
    │   │   ├─ Pipeline::forward_inputs() → 模型前向传播
    │   │   ├─ 应用采样参数 (temperature, top-k, top-p)
    │   │   ├─ 采样下一个 token
    │   │   ├─ 检查停止条件 (EOS, max_tokens)
    │   │   └─ 追加 token 到序列
    │   │
    │   └─ 完成时:
    │       ├─ 计算使用统计
    │       ├─ 构建 ChatCompletionResponse
    │       └─ 通过 MPSC 通道发送给客户端
    │
    └─> 响应返回给用户
        ├─ 非流式: ChatCompletionResponse
        └─ 流式: ChatCompletionChunkResponse (SSE)
```

### 3.3 Scheduler 调度器

定义在 `mistralrs-core/src/scheduler/` 目录。

#### Scheduler Trait

```rust
pub trait Scheduler: Send + Sync {
    fn schedule(&mut self, logger: &IntervalLogger) -> SchedulerOutput<'_>;
    fn waiting_len(&self) -> usize;
    fn running_len(&self) -> usize;
    fn add_seq(&mut self, seq: Sequence);
    fn free_finished_sequence_groups(&mut self);
    fn block_tables(&self) -> Option<BlockTables>;
    fn block_size(&self) -> Option<usize>;
    fn block_engine(&self) -> Option<Arc<Mutex<BlockEngine>>>;
    fn set_prefix_caching_enabled(&mut self, enabled: bool);
}
```

#### 两种调度策略

| 策略 | 类型 | 说明 |
|------|------|------|
| **DefaultScheduler** | 固定批次 | 限制并发序列数量，FCFS 队列，按序列长度分桶 |
| **PagedAttentionScheduler** | 分页内存 | 类似 vLLM 的分页注意力调度，块引擎管理 GPU 内存 |

#### 序列分桶策略

序列按 `(cache_length, has_images_and_is_prompt, token_offset)` 分桶：
- `FixedBucketingManager`：优先选择最短的桶，允许短序列追赶长序列
- 优先级系统：对等待队列中阻塞的序列应用紧迫性乘数

#### 序列状态机

```
Waiting → RunningPrompt → RunningCompletion → Done
                ↑                                |
                └──── (被抢占时回到等待队列) ─────┘
```

### 3.4 模型加载流程

模型加载通过 `Loader` trait 驱动，定义在 `mistralrs-core/src/pipeline/loaders/`。

#### 自动检测机制

```rust
pub enum NormalLoaderType {
    Mistral,  Gemma,   Mixtral,    Llama,     Phi2,
    Phi3,     Phi3_5MoE, Qwen2,   Qwen3,     Qwen3Moe,
    Gemma2,   Starcoder2, DeepSeekV2, DeepSeekV3,
    GLM4,     GLM4MoeLite, GLM4Moe, SmolLm3,
    GraniteMoeHybrid, GptOss,
}

// 从 config.json 的 "architectures" 字段自动检测:
impl NormalLoaderType {
    pub fn from_causal_lm_name(name: &str) -> Result<Self> {
        match name {
            "MistralForCausalLM"  => Ok(Self::Mistral),
            "LlamaForCausalLM"    => Ok(Self::Llama),
            "Qwen3ForCausalLM"    => Ok(Self::Qwen3),
            // ... 20+ 架构变体
        }
    }
}
```

#### 加载流程

```
LoaderBuilder::build()
    │
    ├─> 创建具体的 Loader（NormalLoader, GGUFLoader, VisionLoader 等）
    │
    ├─> Loader::load_model_from_hf()
    │   ├─ 从 HuggingFace Hub 下载模型文件
    │   ├─ 解析 config.json 确定架构
    │   ├─ 创建 VarBuilder（权重加载器）
    │   │   └─ VarBuilder 使用 pp() 前缀系统匹配 PyTorch 命名
    │   │       例: vb.pp("model").pp("layers").pp("0").pp("self_attn")
    │   │       → 查找 "model.layers.0.self_attn.*" 的权重
    │   ├─ 创建 DeviceMapper（设备映射）
    │   ├─ 调用具体模型的 load() 方法
    │   └─ 返回 Arc<Mutex<dyn Pipeline>>
    │
    └─> Pipeline 注入到 Engine 中
```

#### VarBuilder 前缀系统

这是 Candle 框架的核心机制，模拟 PyTorch 的参数命名层级：

```rust
// Rust 中的模型构建
let vb = VarBuilder::from_safetensors(&filenames, dtype, &device);
let embed = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embed_tokens"));
// → 加载 "embed_tokens.weight"

let layer_vb = vb.pp("layers").pp("0");
let q_proj = linear(hidden, hidden, layer_vb.pp("self_attn").pp("q_proj"));
// → 加载 "layers.0.self_attn.q_proj.weight"
```

### 3.5 Attention 注意力机制

定义在 `mistralrs-core/src/attention/` 目录，采用分层调度策略。

#### 注意力实现选择优先级

```
1. Flash Attention（CUDA/CPU，非 F32-on-CUDA）
   ├─ 支持 F16, BF16, F32(仅 CPU)
   └─ 使用 candle_flash_attn 绑定

2. Metal Fused SDPA（Metal 设备）
   ├─ 有效 head_dim: [32, 64, 72, 80, 96, 128, 256]
   └─ 使用 candle_nn::ops::sdpa

3. cuBLASLt SDPA（CUDA + cuBLAS LT）
   ├─ 融合 bias 和 scale 操作
   └─ 使用分块注意力避免 OOM

4. Naive SDPA（兜底方案）
   ├─ 纯张量运算
   └─ 支持 2D 和 3D mask
```

#### 分块注意力（防 OOM）

```rust
// ATTENTION_CHUNK_SIZE = 1024
pub fn chunked_attention<F>(
    q: &Tensor, k: &Tensor, v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
// 将长序列分成 1024 token 的块分别计算
```

### 3.6 KV Cache 管理

定义在 `mistralrs-core/src/kv_cache/` 目录。

#### Cache 类型

| 类型 | 结构体 | 用途 |
|------|--------|------|
| **Normal** | `SingleCache` | 标准 KV 缓存，动态增长（每次 +512） |
| **Rotating** | `RotatingCache` | 滑动窗口注意力，环形缓冲区 |
| **Full** | `Cache` | 简单的 `Vec<Option<(Tensor, Tensor)>>` |
| **Hybrid** | `HybridCache` | Attention + Mamba 混合模型 |

#### Cache Manager 策略

```rust
pub enum EitherCache {
    Normal(Arc<Mutex<NormalCache>>),   // 批处理：合并/拆分序列缓存
    Full(Cache),                       // 简单模式
    Hybrid(Arc<Mutex<HybridCache>>),  // 混合 Attention + Mamba
}
```

- **NormalCacheManager**：将各序列的 KV Cache 合并为批张量（`clone_in_cache`），推理后拆分回各序列（`clone_out_cache`）
- **HybridCacheManager**：Attention 层使用批 KV Cache，Mamba 层使用基于池的状态管理（gather/scatter 索引）

### 3.7 PagedAttention 分页注意力

定义在 `mistralrs-paged-attn/` crate 和 `mistralrs-core/src/paged_attention/` 集成代码中。

#### 核心组件

```rust
// 块引擎：逻辑块 → 物理块映射
pub struct BlockEngine {
    num_gpu_blocks: usize,
    block_size: usize,                    // 每块 token 数 (8/16/32)
    pool: BlockPool,
    block_tables: HashMap<SeqID, BlockTable>,
    prefix_cacher: PrefixCacher,
}

// 引用计数的块：支持 Copy-on-Write
pub struct BlockRef(Arc<BlockInner>);

// 内存配置
pub enum MemoryGpuConfig {
    MbAmount(usize),       // 固定 MB 分配
    Utilization(f32),      // GPU 内存利用率百分比
    ContextSize(usize),    // 基于最大 token 数
}
```

#### GPU 块数计算

```
num_gpu_blocks = mem_gpu / dtype_size / block_size / num_layers / kv_cache_elements_per_token
```

#### Prefix Caching

相同前缀的请求可以共享物理块，减少内存开销：

```rust
pub struct PrefixCacher {
    enabled: bool,
    cache: HashMap<BlockHash, Vec<BlockRef>>,  // token序列hash → 共享块
}
```

### 3.8 量化系统 (mistralrs-quant)

定义在 `mistralrs-quant/` crate，是量化层的统一抽象。

#### 核心 Trait

```rust
pub trait QuantMethod: Send + Sync + Debug + QuantizedSerde {
    fn new(method: QuantMethodConfig) -> Result<Self>;
    fn dequantize_w(&self) -> Result<Tensor>;             // 反量化权重
    fn forward(&self, a: &Tensor) -> Result<Tensor>;      // 前向传播
    fn forward_autocast(&self, a: &Tensor) -> Result<Tensor>; // 自动类型转换
    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor>; // MoE 路由
    fn quantized_act_type(&self) -> Option<DType>;         // 激活类型
    fn dtype_and_device(&self) -> (DType, Device);         // 数据类型和设备
    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>>; // LoRA 增量
    fn apply_isq(self: Arc<Self>, ...) -> Result<Arc<dyn QuantMethod>>;    // ISQ 转换
}
```

#### 量化方法总览

| 方法 | 位宽 | 粒度 | CPU | CUDA | Metal | 目录 |
|------|------|------|-----|------|-------|------|
| **GGUF** | 2-8 | 块 | ✅ | ✅ | ✅ | `src/gguf/` |
| **GPTQ/AWQ** | 2-8 | 组 | ❌ | ✅ | ❌ | `src/gptq/` |
| **HQQ** | 1-8 | 通道/组 | ✅ | ✅ | ❌ | `src/hqq/` |
| **BitsAndBytes** | 4,8 | 块 | ✅ | ✅ | ✅ | `src/bitsandbytes/` |
| **AFQ** | 2-8 | 组 | ✅ | ✅ | ✅ | `src/afq/` |
| **BlockwiseFP8** | 8 | 块(32) | ✅ | ✅ | ❌ | `src/blockwise_fp8/` |
| **PerTensorFP8** | 8 | 张量 | ✅ | ✅ | ✅ | `src/pertensor_fp8/` |
| **VectorFP8** | 8 | 向量(128) | ✅ | ✅ | ✅ | `src/vector_fp8/` |
| **MXFP4** | 4(FP4) | 块(32) | ❌ | ✅ | ✅ | `src/mxfp4/` |
| **Unquantized** | 16/32 | 无 | ✅ | ✅ | ✅ | `src/unquantized/` |

#### ISQ (即时量化) 类型

```rust
pub enum IsqType {
    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,   // 标准 GGML 类型
    Q2K, Q3K, Q4K, Q5K, Q6K, Q8K,           // K-quant 类型
    HQQ8, HQQ4,                               // HQQ 类型
    F8E4M3,                                    // FP8 E4M3
    AFQ8, AFQ6, AFQ4, AFQ3, AFQ2,            // AFQ 类型
}
```

#### 量化层工厂函数

```rust
// 所有模型层通过这些工厂函数创建量化层
pub fn linear_no_bias(in_dim, out_dim, config, vb) -> Result<Arc<dyn QuantMethod>>
pub fn linear(in_dim, out_dim, config, vb) -> Result<Arc<dyn QuantMethod>>
pub fn linear_b(in_dim, out_dim, bias, config, vb) -> Result<Arc<dyn QuantMethod>>
```

创建流程：
1. 检查是否需要即时 ISQ，可能先移到 CPU 做量化
2. 根据 `QuantizedConfig` 选择量化后端
3. 如果没有量化配置，检查权重张量 → `UnquantLinear`
4. 如果权重缺失 → `DummyLayer`
5. 应用 ISQ 转换（如启用）

---

## 4. 硬件抽象层与设备管理

### 4.1 Candle 张量框架

mistral.rs 的硬件抽象完全依赖 Candle 框架。Candle 提供了统一的 `Tensor` 接口，底层通过 `Storage` 枚举分发到不同后端：

```rust
// candle-core 的核心抽象
pub enum Device {
    Cpu,
    Cuda(CudaDevice),    // 通过 cudarc 库
    Metal(MetalDevice),   // 通过 objc2-metal 库
}

pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
}
```

#### CustomOp 机制

Candle 的 `CustomOp1` trait 允许为不同设备实现自定义内核：

```rust
pub trait CustomOp1 {
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)>;
    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)>;
    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)>;
}
```

**这是添加新硬件后端的核心扩展点**——如果要支持 NPU，需要在 Candle 中添加对应的设备类型和 `CustomOp` 方法。

### 4.2 设备映射 (DeviceMapper)

定义在 `mistralrs-core/src/device_map.rs`，支持多设备层分配。

```rust
pub trait DeviceMapper: Debug + Send + Sync {
    fn set_device(&self, layer: usize, vb: ShardedVarBuilder, loading_isq: bool)
        -> ShardedVarBuilder;
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device>;
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor>;
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor>;
    fn get_unique_devices(&self) -> Vec<Device>;
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<Comm>>;  // NCCL
}
```

#### DeviceMapper 实现

| 映射器 | 用途 | 实现 |
|--------|------|------|
| `DummyDeviceMapper` | 单设备 | 所有层映射到同一设备 |
| `LayerDeviceMapper` | 多 GPU 层分片 | 每层独立指定设备 |
| `NcclDeviceMapper` | NCCL 张量并行 | 跨 GPU 通信 |
| `NcclPipelineParallelMapper` | 流水线并行 | 分层 NCCL 通信 |

#### 设备映射配置

```rust
pub enum DeviceMapSetting {
    Map(DeviceMapMetadata),           // 手动指定层-设备映射
    Auto(AutoDeviceMapParams),        // 自动设备映射
    DummyNccl { nm_device: Device },  // NCCL 分布式（虚拟）
    Nccl { nm_device: Device, comm: Arc<Comm> }, // NCCL 分布式（实际）
}

pub struct DeviceLayerMapMetadata {
    pub ordinal: usize,  // GPU 编号 (0-based)
    pub layers: usize,   // 此 GPU 上的层数
}
```

#### 多 GPU 层分配示例

```rust
// 24 层模型，2 个 GPU：GPU0 放 16 层，GPU1 放 8 层
let device_map = DeviceMapMetadata {
    device_layers: Some(vec![
        DeviceLayerMapMetadata { ordinal: 0, layers: 16 },
        DeviceLayerMapMetadata { ordinal: 1, layers: 8 },
    ]),
    host_layers: None,
};
```

### 4.3 Feature Flags 特性门控

#### 顶层 Feature 传播

```toml
# mistralrs-core/Cargo.toml 中的 feature 定义
[features]
cuda = [
    "candle-core/cuda", "candle-nn/cuda",
    "dep:bindgen_cuda",
    "mistralrs-quant/cuda",
    "mistralrs-paged-attn/cuda",
    "float8/cuda",
]
metal = [
    "candle-core/metal", "candle-nn/metal",
    "mistralrs-quant/metal",
    "mistralrs-paged-attn/metal",
    "dep:candle-metal-kernels", "dep:objc2-metal",
]
flash-attn = ["cuda", "dep:candle-flash-attn"]
cudnn = ["candle-core/cudnn"]
accelerate = ["candle-core/accelerate", "candle-nn/accelerate"]
mkl = ["candle-core/mkl", "candle-nn/mkl"]
nccl = ["cuda", "mistralrs-quant/nccl"]
```

#### 条件编译模式

项目中使用三种条件编译模式：

```rust
// 模式 1：Feature 门控整个模块
#[cfg(feature = "cuda")]
mod ffi;

// 模式 2：替代实现
#[cfg(feature = "cuda")]
pub use gptq_cuda::{gptq_linear, GptqLayer};
#[cfg(not(feature = "cuda"))]
pub use gptq_cpu::{gptq_linear, GptqLayer};

// 模式 3：运行时分支
#[cfg(feature = "cuda")]
if matches!(x.device(), Device::Cuda(_)) {
    // CUDA 优化路径
} else {
    // 通用路径
}
```

---

## 5. CPU 后端实现详解

### 5.1 CPU 上的张量运算

在不启用任何加速 feature 的默认构建中，所有计算在 CPU 上执行。Candle 的 CPU 后端使用 Rust 原生实现的张量运算。

#### CPU MatMul 路径

```rust
// mistralrs-quant/src/unquantized/mod.rs 中的 UnquantLinear
// 核心 MatMul 逻辑:

#[cfg(feature = "accelerate")]
{
    // Apple Accelerate：转换为 F32 → matmul → 转回
    let a = a.to_dtype(DType::F32)?;
    let w = self.weight.to_dtype(DType::F32)?;
    a.matmul(&w.t()?)?.to_dtype(original_dtype)?
}

#[cfg(not(feature = "accelerate"))]
{
    if a.device().is_cpu() {
        // CPU 默认：转换为 F16 → matmul → 转回（节省内存）
        let a = a.to_dtype(DType::F16)?;
        let w = self.weight.to_dtype(DType::F16)?;
        a.matmul(&w.t()?)?.to_dtype(original_dtype)?
    } else {
        // GPU 直接计算
        a.matmul(&self.weight.t()?)?
    }
}
```

**关键点**：CPU 上默认会将权重转换为 F16 进行 matmul 以减少内存带宽压力，然后转回原始类型。

#### cuBLASLt 对 CPU 的处理

```rust
// mistralrs-quant/src/cublaslt/ 目录
// cuBLASLt 仅在 CUDA 设备上启用
// CPU 路径不经过 cuBLASLt，直接走 Candle 原生 matmul
```

### 5.2 CPU 上的量化支持

#### GGUF（CPU 原生支持）

GGUF 是 CPU 上最完善的量化格式，由 Candle 的 `QTensor` 和 `QMatMul` 原生支持：

```rust
// mistralrs-quant/src/gguf/mod.rs
pub struct GgufMatMul {
    w: QMatMul,           // Candle 原生量化矩阵乘法
    b: Option<Tensor>,    // 可选偏置
}

impl QuantMethod for GgufMatMul {
    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        let x = self.w.forward(a)?;  // QMatMul 在 CPU 上直接执行量化乘法
        if let Some(b) = &self.b { x.broadcast_add(b) } else { Ok(x) }
    }
}
```

GGUF 的 CPU 路径直接在量化表示上计算，无需反量化为 F16/F32，这是其在 CPU 上高效的关键原因。

#### HQQ（CPU 部分支持）

```rust
// mistralrs-quant/src/hqq/hqq_op.rs - CPU 实现
// 使用宏生成不同数据类型的反量化代码
macro_rules! dequant_for_dtype {
    ($ty:ty, $dtype:expr) => { ... }
}
// CPU 路径：反量化 → 全精度 matmul
```

#### GPTQ/AWQ（CPU 不支持）

```rust
// mistralrs-quant/src/gptq/gptq_cpu.rs
pub fn gptq_linear(...) -> Result<Arc<dyn QuantMethod>> {
    candle_core::bail!("GPTQ is only supported on CUDA.")
}
// GPTQ 需要 CUDA 专用内核，CPU 上无法运行
```

#### BitsAndBytes（CPU 支持）

```rust
// mistralrs-quant/src/bitsandbytes/mod.rs
// 纯 Rust 实现，支持 NF4 和 FP4
// CPU 路径：通过查找表反量化
```

### 5.3 CPU 上的注意力计算

```rust
// mistralrs-core/src/attention/backends/cpu.rs
// CPU Flash Attention 实现
// 支持 F16, BF16, F32

// 如果 CPU 上不启用 flash-attn feature：
// → 使用 Naive SDPA（纯张量运算）
// 这是完全通用的实现，但比 Flash Attention 慢
```

### 5.4 CPU 性能优化选项

| 优化 | Feature Flag | 效果 |
|------|-------------|------|
| Apple Accelerate | `accelerate` | macOS 上使用 BLAS 加速 F32 矩阵运算 |
| Intel MKL | `mkl` | x86 上使用 MKL 加速矩阵运算 |
| Rayon 并行 | 默认启用 | 多核并行的数据处理 |
| GGUF 量化 | 默认支持 | 降低内存带宽需求 |

```bash
# 使用 Intel MKL 加速 CPU 推理
cargo build --release --features mkl

# 使用 Apple Accelerate
cargo build --release --features accelerate
```

---

## 6. NPU 扩展开发指南

本节详细说明如何为 mistral.rs 添加 NPU（神经网络处理器）支持。整体策略是参照 CUDA 和 Metal 的实现模式，在各个抽象层中添加 NPU 分支。

### 6.1 架构概览：从 CUDA/Metal 模式学习

当前的硬件抽象层次：

```
┌──────────────────────────────────────────────────────────────┐
│  模型代码 (llama.rs, mistral.rs 等)                          │
│  设备无关：使用 candle::Tensor                               │
│  调用 QuantMethod::forward() trait                           │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│  量化抽象层                                                   │
│  pub trait QuantMethod { fn forward(&self, a: &Tensor); }    │
│  UnquantLinear / GptqLayer / HqqLayer / ...                  │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│  Candle 张量框架 (candle-core)                                │
│  Storage { Cpu, Cuda, Metal }                                │
│  CustomOp1 { cpu_fwd, cuda_fwd, metal_fwd }                 │
└─────────┬────────────┬───────────────┬───────────────────────┘
          │            │               │
          ▼            ▼               ▼
       CPU 后端    CUDA 后端       Metal 后端
    (Rust 原生)   (cudarc FFI)   (objc2-metal)
```

**NPU 需要在每一层中添加对应的支持。**

### 6.2 第一步：扩展 Candle 张量框架

这是最关键的一步。需要 fork 或修改 `candle-core` 以添加 NPU 设备类型。

#### 6.2.1 添加 Device 变体

```rust
// 在 candle-core 中修改
pub enum Device {
    Cpu,
    Cuda(CudaDevice),
    Metal(MetalDevice),
    Npu(NpuDevice),        // 新增
}

pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
    Metal { gpu_id: usize },
    Npu { device_id: usize },  // 新增
}
```

#### 6.2.2 添加 Storage 变体

```rust
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
    Npu(NpuStorage),       // 新增
}
```

#### 6.2.3 实现 NpuDevice

```rust
pub struct NpuDevice {
    device_id: usize,
    // NPU 运行时句柄（取决于具体的 NPU SDK）
    // 例如华为昇腾: AscendCL 上下文
    // 或其他 NPU 的对应运行时
}

impl NpuDevice {
    pub fn new(device_id: usize) -> Result<Self> { ... }
    // 实现内存分配、数据传输、内核调度等基础方法
}
```

#### 6.2.4 实现 NpuStorage

```rust
pub struct NpuStorage {
    buffer: NpuBuffer,       // NPU 设备上的内存缓冲区
    dtype: DType,
    device: NpuDevice,
}

impl NpuStorage {
    // 实现基础张量操作：
    // - 内存分配/释放
    // - CPU ↔ NPU 数据传输
    // - 基础算术运算（加、减、乘、除）
    // - MatMul（矩阵乘法）
    // - Softmax
    // - Activation functions
    // - 张量重塑/转置/拼接
}
```

#### 6.2.5 扩展 CustomOp trait

```rust
pub trait CustomOp1 {
    fn cpu_fwd(&self, ...) -> Result<(CpuStorage, Shape)>;
    fn cuda_fwd(&self, ...) -> Result<(CudaStorage, Shape)>;
    fn metal_fwd(&self, ...) -> Result<(MetalStorage, Shape)>;
    fn npu_fwd(&self, ...) -> Result<(NpuStorage, Shape)>;  // 新增
}
```

### 6.3 第二步：实现 NPU 量化内核

在 `mistralrs-quant/` 中添加 NPU 支持。

#### 6.3.1 创建 NPU 内核目录

```
mistralrs-quant/
├── kernels/
│   ├── npu/                    # 新增 NPU 内核目录
│   │   ├── ops.npu             # 基础运算内核
│   │   ├── rotary.npu          # RoPE 内核
│   │   ├── hqq.npu             # HQQ 反量化内核
│   │   └── ...
│   └── ... (现有 CUDA/Metal 内核)
├── src/
│   ├── unquantized/
│   │   └── mod.rs              # 在 MatMul 路径中添加 NPU 分支
│   ├── gguf/
│   │   ├── npu.rs              # 新增 NPU 实现
│   │   └── ...
│   ├── hqq/
│   │   ├── npu_op.rs           # 新增 NPU 实现
│   │   └── ...
```

#### 6.3.2 修改 UnquantLinear

```rust
// mistralrs-quant/src/unquantized/mod.rs
impl QuantMethod for UnquantLinear {
    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "npu")]
        if matches!(a.device(), Device::Npu(_)) {
            // NPU 优化路径：使用 NPU 内核执行 matmul
            return npu_matmul(a, &self.weight, self.bias.as_ref());
        }

        // 现有 CPU/CUDA/Metal 路径...
    }
}
```

#### 6.3.3 修改条件编译

参照现有的 CUDA/CPU 切换模式：

```rust
// 在各量化模块中添加 NPU 分支
#[cfg(feature = "npu")]
pub use npu_impl::{npu_forward, NpuQuantLayer};

#[cfg(not(feature = "npu"))]
pub fn npu_forward(...) -> Result<Tensor> {
    candle_core::bail!("NPU support is not enabled.")
}
```

### 6.4 第三步：实现 NPU PagedAttention

在 `mistralrs-paged-attn/` 中添加 NPU 后端。

#### 6.4.1 创建目录结构

```
mistralrs-paged-attn/src/
├── cuda/           # 现有 CUDA 实现
├── metal/          # 现有 Metal 实现
├── npu/            # 新增 NPU 实现
│   ├── mod.rs
│   ├── ffi.rs                 # NPU 内核 FFI 绑定
│   ├── backend/
│   │   ├── mod.rs
│   │   ├── paged_attention.rs # PagedAttention CustomOp1 实现
│   │   ├── cache.rs           # 块复制/交换操作
│   │   └── scale_update.rs    # KV scale 更新
│   └── kernels/               # NPU 内核文件
│       ├── pagedattention.xxx  # NPU 专用内核格式
│       ├── copy_blocks.xxx
│       └── reshape_and_cache.xxx
```

#### 6.4.2 实现 CustomOp1

```rust
// mistralrs-paged-attn/src/npu/backend/paged_attention.rs
impl candle_core::CustomOp1 for PagedAttention {
    fn npu_fwd(
        &self,
        q: &NpuStorage,
        q_l: &Layout,
    ) -> Result<(NpuStorage, Shape)> {
        // 1. 从 NpuStorage 提取张量
        // 2. 调用 NPU 内核执行分页注意力
        // 3. 返回结果
    }
}
```

#### 6.4.3 修改 lib.rs 添加 NPU 导出

```rust
// mistralrs-paged-attn/src/lib.rs
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub use cuda::*;

#[cfg(feature = "metal")]
pub use metal::*;

#[cfg(feature = "npu")]      // 新增
pub use npu::*;
```

### 6.5 第四步：设备映射集成

修改 `mistralrs-core/src/device_map.rs`。

```rust
// 在 DeviceMapper 的层分配逻辑中添加 NPU 支持
for DeviceLayerMapMetadata { ordinal, layers } in device_layers {
    let dev = match device.location() {
        DeviceLocation::Cuda { gpu_id } => {
            Device::new_cuda(*ordinal)?
        }
        DeviceLocation::Metal { gpu_id } => {
            Device::new_metal(*ordinal)?
        }
        DeviceLocation::Npu { device_id } => {    // 新增
            Device::new_npu(*ordinal)?
        }
        DeviceLocation::Cpu => Device::Cpu,
    };
    combined.extend(vec![dev; *layers]);
}
```

### 6.6 第五步：构建系统集成

#### 6.6.1 修改 build.rs

```rust
// mistralrs-quant/build.rs - 添加 NPU 内核编译
#[cfg(feature = "npu")]
fn compile_npu_kernels() {
    // 使用 NPU SDK 的编译器编译内核
    // 例如华为昇腾: 使用 ATC 编译器
    // 生成静态库并链接
    println!("cargo:rustc-link-lib=static=mistralrs_npu_kernels");
}

fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();

    #[cfg(feature = "metal")]
    compile_metal_kernels();

    #[cfg(feature = "npu")]     // 新增
    compile_npu_kernels();
}
```

#### 6.6.2 修改 Cargo.toml

```toml
# mistralrs-quant/Cargo.toml
[features]
npu = ["candle-core/npu"]

[build-dependencies]
# NPU 编译工具链绑定（如需要）
```

### 6.7 第六步：Feature Flag 注册

#### 6.7.1 Workspace 级别

```toml
# 根目录 Cargo.toml
[workspace.features]
npu = []
```

#### 6.7.2 Core crate 级别

```toml
# mistralrs-core/Cargo.toml
[features]
npu = [
    "candle-core/npu",
    "candle-nn/npu",
    "mistralrs-quant/npu",
    "mistralrs-paged-attn/npu",
]
```

#### 6.7.3 CLI 级别

```toml
# mistralrs-cli/Cargo.toml
[features]
npu = ["mistralrs-core/npu", "mistralrs/npu"]
```

#### 6.7.4 构建命令

```bash
# NPU 构建
cargo build --release --features npu

# NPU + 量化
cargo build --release --features "npu"
```

### 6.8 渐进式实现策略

建议分阶段实现，每阶段都可独立验证：

#### 阶段 1：基础推理（最小可行产品）

**目标**：在 NPU 上运行一个简单模型

1. Fork candle-core，添加 `Device::Npu` 和 `NpuStorage`
2. 实现 NPU 上的基础张量操作（matmul, add, softmax, rope 等）
3. 实现 `UnquantLinear` 的 NPU 路径
4. 测试运行一个小型未量化模型（如 Phi-2）

**需要修改的文件**：
- candle-core（fork）
- `mistralrs-quant/src/unquantized/mod.rs`
- `mistralrs-core/src/device_map.rs`
- `mistralrs-core/Cargo.toml`

#### 阶段 2：量化支持

**目标**：在 NPU 上运行量化模型

1. 实现 GGUF 量化的 NPU 内核（优先级最高，CPU/GPU 通用）
2. 实现 ISQ（即时量化）的 NPU 支持
3. 可选：实现 HQQ、BnB 等其他量化格式

**需要修改的文件**：
- `mistralrs-quant/src/gguf/npu.rs`（新增）
- `mistralrs-quant/src/utils/isq.rs`
- `mistralrs-quant/build.rs`

#### 阶段 3：PagedAttention

**目标**：高效多请求批处理

1. 实现 NPU 版本的 PagedAttention 内核
2. 实现块复制/交换操作
3. 集成到调度器

**需要修改的文件**：
- `mistralrs-paged-attn/src/npu/`（新增目录）
- `mistralrs-paged-attn/build.rs`
- `mistralrs-paged-attn/Cargo.toml`

#### 阶段 4：优化与完善

**目标**：性能调优

1. 实现 Flash Attention NPU 变体
2. 实现融合内核（fused kernels）：RoPE + Attention 等
3. 实现 MoE 路由的 NPU 加速（topk_softmax 等）
4. 多 NPU 设备支持（类似 NCCL）

#### 降级策略

对于未实现 NPU 内核的操作，建议采用以下降级策略：

```rust
fn forward_npu(&self, a: &Tensor) -> Result<Tensor> {
    // 尝试 NPU 优化路径
    if let Ok(result) = npu_optimized_forward(a) {
        return Ok(result);
    }

    // 降级：数据拷贝到 CPU → CPU 计算 → 拷贝回 NPU
    let a_cpu = a.to_device(&Device::Cpu)?;
    let result = self.forward_cpu(&a_cpu)?;
    result.to_device(a.device())
}
```

**注意**：频繁的 NPU↔CPU 数据拷贝会严重影响性能。降级策略仅用于开发过渡期，最终应实现完整的 NPU 内核。

---

## 7. 对外接口层

### 7.1 CLI 命令行

入口文件：`mistralrs-cli/src/main.rs`

| 命令 | 说明 |
|------|------|
| `mistralrs serve` | 启动 HTTP 服务器 |
| `mistralrs run` | 交互式推理 |
| `mistralrs bench` | 性能基准测试 |
| `mistralrs quantize` | 生成 UQFF 量化文件 |
| `mistralrs doctor` | 系统诊断 |
| `mistralrs tune` | 推荐量化和设备映射 |
| `mistralrs login` | HuggingFace Hub 认证 |
| `mistralrs cache` | 缓存管理 |
| `mistralrs from-config` | 从 TOML 配置运行 |
| `mistralrs completions` | Shell 补全生成 |

### 7.2 HTTP Server (OpenAI 兼容)

入口文件：`mistralrs-server-core/src/`

#### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 聊天补全（流式/非流式） |
| `/v1/completions` | POST | 文本补全 |
| `/v1/embeddings` | POST | 文本嵌入 |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/models/unload` | POST | 卸载模型 |
| `/v1/models/reload` | POST | 重新加载模型 |
| `/v1/models/status` | POST | 模型状态查询 |
| `/v1/images/generations` | POST | 图像生成 |
| `/v1/audio/speech` | POST | 语音合成 |
| `/v1/responses` | POST | 创建响应 |
| `/re_isq` | POST | 重新应用 ISQ 量化 |
| `/health` | GET | 健康检查 |

#### Router 构建

```rust
let router = MistralRsServerRouterBuilder::new()
    .with_mistralrs(shared_state)
    .with_include_swagger_routes(true)  // 启用 /docs
    .with_base_path("/api")             // 路由前缀
    .with_allowed_origins(vec!["*".to_string()])
    .with_max_body_limit(50 * 1024 * 1024) // 50MB
    .build()
    .await?;
```

### 7.3 Rust SDK

入口文件：`mistralrs/src/lib.rs`

```rust
use mistralrs::*;

#[tokio::main]
async fn main() -> Result<()> {
    // 构建模型
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    // 发送请求
    let messages = TextMessages::new()
        .add_message(TextMessageRole::System, "You are helpful.")
        .add_message(TextMessageRole::User, "Hello!");

    let response = model.send_chat_request(messages).await?;

    // 流式输出
    let mut stream = model.stream_chat_request(messages).await?;
    while let Some(chunk) = stream.next().await { /* 处理 */ }

    Ok(())
}
```

#### 可用 Builder 类型

| Builder | 用途 |
|---------|------|
| `TextModelBuilder` | 标准文本模型 |
| `VisionModelBuilder` | 视觉语言模型 |
| `GgufModelBuilder` | GGUF 量化模型 |
| `LoraModelBuilder` | LoRA 适配器 |
| `XLoraModelBuilder` | X-LoRA 混合适配器 |
| `EmbeddingModelBuilder` | 嵌入模型 |
| `DiffusionModelBuilder` | 图像生成 |
| `SpeechModelBuilder` | 语音合成 |
| `AnyMoeModelBuilder` | MoE 模型 |
| `MultiModelBuilder` | 多模型同时加载 |

### 7.4 Python SDK (PyO3)

入口文件：`mistralrs-pyo3/src/lib.rs`

```python
from mistralrs import Runner, ChatCompletionRequest, Architecture

# 创建推理器
runner = Runner(
    which=Architecture.Plain,
    model_id="microsoft/Phi-3.5-mini-instruct",
    isq=IsqType.Q8_0,
)

# 发送请求
response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ],
        max_tokens=256,
    )
)
```

---

## 8. 添加新模型架构

### 步骤概览

1. **实现模型**：在 `mistralrs-core/src/models/` 中添加新模型文件
2. **注册架构**：在 `NormalLoaderType` 枚举中添加变体
3. **实现 Loader**：创建模型专用的 `NormalModelLoader` 实现
4. **更新检测**：在 `from_causal_lm_name()` 中添加架构名映射
5. **添加 Pipeline 支持**：在 `pipeline/normal.rs` 或对应 pipeline 中注册

### 模型实现要点

```rust
// mistralrs-core/src/models/my_model.rs

pub struct MyModel {
    embed_tokens: Embedding,
    layers: Vec<MyDecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    cache: EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl MyModel {
    pub fn new(
        config: &MyConfig,
        vb: ShardedVarBuilder,
        normal_loading_metadata: &NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // 使用 vb.pp() 前缀系统匹配 PyTorch 权重名
        let embed = embedding(
            config.vocab_size, config.hidden_size,
            vb.pp("model.embed_tokens"), &config.quantized,
        )?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer_vb = vb.pp(format!("model.layers.{i}"));
            // 使用 DeviceMapper 将层放到正确的设备
            let layer_vb = normal_loading_metadata.mapper
                .set_device(i, layer_vb, false);
            layers.push(MyDecoderLayer::new(config, layer_vb)?);
        }

        // ...
    }
}

// 实现 NormalModel trait
impl NormalModel for MyModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        // 实现前向传播
    }
}
```

### VarBuilder 前缀验证

在实现新模型前，建议查看模型的 `model.safetensors.index.json` 文件，确认权重命名：

```json
{
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
    ...
  }
}
```

确保 Rust 代码中的 `vb.pp()` 路径与 JSON 中的键完全匹配。

---

## 9. 构建与测试

### 构建命令

```bash
# 纯 CPU 构建
cargo build --release

# CUDA 完整构建
cargo build --release --features "cuda flash-attn cudnn"

# Metal 构建 (macOS)
cargo build --release --features metal

# CPU + Intel MKL
cargo build --release --features mkl

# CPU + Apple Accelerate
cargo build --release --features accelerate

# 安装 CLI
cargo install --path mistralrs-cli --features <features>
```

### 测试命令

```bash
# 核心测试
cargo test -p mistralrs-core -p mistralrs-quant -p mistralrs-vision

# 编译检查（推荐在修改代码后立即运行）
cargo check

# 格式化
make fmt

# 格式检查
cargo fmt --all -- --check

# Clippy 静态分析
cargo clippy --workspace --tests --examples -- -D warnings
```

### 编译检查优先

根据项目规范，**修改代码后应始终运行 `cargo check`** 确保编译通过，然后再进行更复杂的测试。

---

## 10. 关键文件索引

### 核心架构

| 文件 | 说明 |
|------|------|
| `mistralrs-core/src/lib.rs` | 公共 API、MistralRs 结构体定义 |
| `mistralrs-core/src/engine/mod.rs` | 引擎主循环、请求处理 |
| `mistralrs-core/src/pipeline/mod.rs` | Pipeline trait 定义 |
| `mistralrs-core/src/pipeline/normal.rs` | NormalPipeline 实现 |
| `mistralrs-core/src/pipeline/loaders/mod.rs` | Loader trait、模型路径管理 |
| `mistralrs-core/src/pipeline/loaders/normal_loaders.rs` | 文本模型加载器 |
| `mistralrs-core/src/pipeline/loaders/vision_loaders.rs` | 视觉模型加载器 |
| `mistralrs-core/src/device_map.rs` | 设备映射、多 GPU 支持 |
| `mistralrs-core/src/model_loader.rs` | LoaderBuilder |

### 模型实现

| 文件 | 说明 |
|------|------|
| `mistralrs-core/src/models/` | 所有文本模型实现（27+） |
| `mistralrs-core/src/vision_models/` | 视觉语言模型（18+） |
| `mistralrs-core/src/diffusion_models/` | 扩散模型 |
| `mistralrs-core/src/speech_models/` | 语音模型 |
| `mistralrs-core/src/embedding_models/` | 嵌入模型 |
| `mistralrs-core/src/xlora_models/` | X-LoRA 模型 |

### 推理子系统

| 文件 | 说明 |
|------|------|
| `mistralrs-core/src/attention/` | 注意力机制（Flash/SDPA/Naive） |
| `mistralrs-core/src/kv_cache/` | KV Cache 管理 |
| `mistralrs-core/src/paged_attention/` | 分页注意力集成 |
| `mistralrs-core/src/scheduler/` | 调度器 |
| `mistralrs-core/src/sampler.rs` | Token 采样 |
| `mistralrs-core/src/sequence.rs` | 序列状态管理 |
| `mistralrs-core/src/layers.rs` | 通用层实现（RmsNorm, RoPE 等） |
| `mistralrs-core/src/ops.rs` | 自定义操作（topk, 排序等） |

### 量化系统

| 文件 | 说明 |
|------|------|
| `mistralrs-quant/src/lib.rs` | QuantMethod trait、工厂函数 |
| `mistralrs-quant/src/gguf/` | GGUF 量化（CPU/CUDA/Metal） |
| `mistralrs-quant/src/gptq/` | GPTQ/AWQ（仅 CUDA） |
| `mistralrs-quant/src/hqq/` | HQQ 量化（CPU/CUDA） |
| `mistralrs-quant/src/bitsandbytes/` | BitsAndBytes（全平台） |
| `mistralrs-quant/src/afq/` | AFQ 量化 |
| `mistralrs-quant/src/blockwise_fp8/` | 块级 FP8 |
| `mistralrs-quant/src/mxfp4/` | MXFP4 量化 |
| `mistralrs-quant/src/unquantized/` | 未量化（回退） |
| `mistralrs-quant/src/distributed/` | 分布式张量并行 |
| `mistralrs-quant/build.rs` | CUDA/Metal 内核编译 |

### 硬件内核

| 文件 | 说明 |
|------|------|
| `mistralrs-paged-attn/src/cuda/` | CUDA PagedAttention 内核 |
| `mistralrs-paged-attn/src/metal/` | Metal PagedAttention 内核 |
| `mistralrs-quant/kernels/` | 量化 CUDA 内核（41 个 .cu 文件） |
| `mistralrs-core/src/cuda/` | Core CUDA 内核（排序、MoE GEMM） |

### 构建系统

| 文件 | 说明 |
|------|------|
| `Cargo.toml` | Workspace 定义、依赖、Feature Flags |
| `mistralrs-core/build.rs` | Core CUDA 内核编译 |
| `mistralrs-quant/build.rs` | 量化内核编译（CUDA + Metal） |
| `mistralrs-paged-attn/build.rs` | PagedAttention 内核编译 |
| `Makefile` | 格式化（cargo fmt + ruff + clang-format） |

### 对外接口

| 文件 | 说明 |
|------|------|
| `mistralrs-cli/src/main.rs` | CLI 入口 |
| `mistralrs-server-core/src/` | HTTP 服务器路由 |
| `mistralrs/src/lib.rs` | Rust SDK |
| `mistralrs-pyo3/src/lib.rs` | Python SDK |

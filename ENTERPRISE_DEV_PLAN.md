# mistral.rs 企业级私有化推理引擎 — 二次开发方案

> 本文档基于对 mistral.rs 源码的深度分析，结合企业实际部署架构，提出将其改造为企业级私有化推理引擎的完整方案。

---

## 目录

- [一、企业整体架构](#一企业整体架构)
- [二、各层职责划分](#二各层职责划分)
- [三、推理引擎改造方案（基于 mistral.rs）](#三推理引擎改造方案基于-mistralrs)
  - [3.1 保留完整推理能力](#31-保留完整推理能力)
  - [3.2 必须补充的引擎侧功能](#32-必须补充的引擎侧功能)
  - [3.3 推理性能优化](#33-推理性能优化)
  - [3.4 引擎稳定性加固](#34-引擎稳定性加固)
- [四、模型转换服务独立方案](#四模型转换服务独立方案)
- [五、统一推理网关设计要点](#五统一推理网关设计要点)
- [六、推理引擎与网关的接口协议](#六推理引擎与网关的接口协议)
- [七、多引擎统一适配层](#七多引擎统一适配层)
- [八、分阶段实施计划](#八分阶段实施计划)
- [九、风险与对策](#九风险与对策)

---

## 一、企业整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         企业智能体服务                           │
│            (Agent / RAG / Workflow / 应用层)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │  OpenAI 兼容 API (HTTP/gRPC)
┌──────────────────────────▼──────────────────────────────────────┐
│                   统一推理网关 (Rust 开发)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐  │
│  │ 认证鉴权  │ │ 限流背压  │ │ 路由分发  │ │ 审计/可观测/计量  │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘  │
└────────┬─────────────────────┬──────────────────┬───────────────┘
         │                     │                  │
         ▼                     ▼                  ▼
  ┌──────────────┐   ┌──────────────┐    ┌──────────────┐
  │ mistral.rs   │   │    vLLM      │    │   SGLang     │
  │ 推理引擎      │   │  (Python)    │    │  (Python)    │
  │ (自开发)      │   │  (三方)      │    │  (三方)      │
  │              │   │              │    │              │
  │ 文本/视觉/   │   │              │    │              │
  │ 语音/图像/   │   │              │    │              │
  │ 嵌入/MoE    │   │              │    │              │
  └──────┬───────┘   └──────────────┘    └──────────────┘
         │
  ┌──────▼───────┐
  │ 模型管理服务  │
  │  ┌─────────┐ │
  │  │模型转换  │ │   ← 从推理引擎中独立出来
  │  │量化调优  │ │
  │  │版本管理  │ │
  │  └─────────┘ │
  └──────────────┘
```

### 核心设计思路

1. **网关承担所有公共能力**：认证、限流、背压、审计、计量等在网关层统一处理
2. **推理引擎专注推理**：保持完整模态能力（文本/视觉/语音/图像/嵌入/MoE），专注于高效推理
3. **模型转换独立部署**：量化、格式转换、模型调优工具从推理引擎中抽离，供模型管理服务调用
4. **多引擎统一接口**：网关后端可对接 mistral.rs / vLLM / SGLang 等，通过统一适配层屏蔽差异

---

## 二、各层职责划分

| 能力 | 网关层 | 推理引擎 | 模型管理服务 |
|------|--------|---------|-------------|
| 用户认证/鉴权 | ✅ | ❌ | ❌ |
| 请求限流/背压 | ✅ | ❌ | ❌ |
| 审计日志 | ✅ | ❌ | ❌ |
| 使用计量/计费 | ✅ | ❌ | ❌ |
| 路由分发/负载均衡 | ✅ | ❌ | ❌ |
| 模型推理（前向传播） | ❌ | ✅ | ❌ |
| KV Cache / PagedAttention | ❌ | ✅ | ❌ |
| 调度器 / Continuous Batching | ❌ | ✅ | ❌ |
| 量化推理（GGUF/GPTQ/HQQ/FP8） | ❌ | ✅ | ❌ |
| 多模型加载/卸载 | ❌ | ✅ | 发起指令 |
| 健康检查 / 就绪探针 | 消费 | ✅ 提供 | ❌ |
| Metrics 暴露 | 消费聚合 | ✅ 提供 | ❌ |
| OOM 防护 / 资源保护 | ❌ | ✅ | ❌ |
| 模型量化转换 | ❌ | ❌ | ✅ |
| 模型格式校验/版本管理 | ❌ | ❌ | ✅ |
| 硬件适配调优 | ❌ | ❌ | ✅ |

---

## 三、推理引擎改造方案（基于 mistral.rs）

### 3.1 保留完整推理能力

**不做功能裁剪**，保留 mistral.rs 的全部推理能力：

| Pipeline | 保留状态 | 说明 |
|----------|---------|------|
| NormalPipeline | ✅ 保留 | 27+ 标准文本模型（Llama, Qwen3, DeepSeek, Phi 等） |
| GGUFPipeline | ✅ 保留 | GGUF 量化模型推理 |
| GGMLPipeline | ✅ 保留 | GGML 兼容 |
| VisionPipeline | ✅ 保留 | 18+ 视觉语言模型（LLaVA, Qwen2-VL 等） |
| EmbeddingPipeline | ✅ 保留 | 嵌入向量模型 |
| DiffusionPipeline | ✅ 保留 | Flux 图像生成 |
| SpeechPipeline | ✅ 保留 | TTS 语音合成 |
| SpeculativePipeline | ✅ 保留 | 投机解码（低延迟场景有价值） |
| AnyMoE | ✅ 保留 | 动态 MoE |
| X-LoRA | ✅ 保留 | LoRA 适配器热切换 |

**理由**：企业场景下不同业务方可能需要不同模态。保持完整能力让单一引擎可以服务多种业务需求，降低运维复杂度。

### 3.2 必须补充的引擎侧功能

虽然网关承担了大部分公共能力，但推理引擎自身仍需补充以下功能才能达到生产级要求：

#### 3.2.1 Prometheus Metrics 暴露

**当前状态**：只有 `IntervalLogger` 打印到控制台，**无 Prometheus 端点**。

网关需要从引擎拉取实时指标做负载均衡决策。

```rust
// 新增 mistralrs-server-core/src/metrics.rs
use prometheus::{IntGauge, Histogram, HistogramVec, Registry, TextEncoder};

pub struct EngineMetrics {
    pub registry: Registry,

    // ---- 网关决策必需指标 ----
    pub scheduler_waiting_seqs: IntGauge,      // 等待队列深度
    pub scheduler_running_seqs: IntGauge,      // 正在运行的序列数
    pub gpu_memory_used_bytes: IntGauge,       // GPU 已用内存
    pub gpu_memory_total_bytes: IntGauge,      // GPU 总内存
    pub paged_attn_blocks_free: IntGauge,      // 空闲 KV Cache 块数
    pub paged_attn_blocks_total: IntGauge,     // 总 KV Cache 块数

    // ---- 推理性能指标 ----
    pub request_duration_seconds: HistogramVec, // 请求处理耗时（按 model 标签）
    pub time_to_first_token: Histogram,         // TTFT（首 Token 延迟）
    pub time_per_output_token: Histogram,       // TPOT（每 Token 延迟）
    pub tokens_per_second: Histogram,           // 吞吐量
    pub prompt_tokens_total: IntCounter,        // 总 prompt token 数
    pub completion_tokens_total: IntCounter,    // 总生成 token 数

    // ---- 引擎状态指标 ----
    pub models_loaded: IntGauge,                // 已加载模型数
    pub inflight_requests: IntGauge,            // 在途请求数
}
```

**暴露方式**：在现有 HTTP 服务上新增 `/metrics` 端点

```rust
// mistralrs-server-core/src/routes.rs 中新增
async fn prometheus_metrics(State(state): State<Arc<SharedState>>) -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = state.metrics.registry.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        buffer,
    )
}
```

**指标采集点嵌入**（修改 Engine 主循环）：

```rust
// mistralrs-core/src/engine/mod.rs 的 run() 循环中
// 每次调度后更新指标
let scheduler_output = self.scheduler.lock().schedule(&logger);
metrics.scheduler_waiting_seqs.set(self.scheduler.lock().waiting_len() as i64);
metrics.scheduler_running_seqs.set(self.scheduler.lock().running_len() as i64);

// 每次推理完成后记录延迟
let start = Instant::now();
pipeline.step(&mut seqs, is_prompt, ...).await?;
let duration = start.elapsed();
metrics.time_per_output_token.observe(duration.as_secs_f64());
```

#### 3.2.2 增强健康检查（Liveness + Readiness）

**当前状态**：`/health` 只返回固定 "OK"，不反映真实状态。

```rust
// 增强版健康检查
// GET /health/live — Kubernetes liveness 探针
async fn liveness() -> impl IntoResponse {
    // 进程存活即返回 200
    StatusCode::OK
}

// GET /health/ready — Kubernetes readiness 探针
async fn readiness(State(state): State<Arc<SharedState>>) -> impl IntoResponse {
    let details = serde_json::json!({
        "status": if is_ready { "ready" } else { "not_ready" },
        "models_loaded": loaded_count,
        "gpu_memory_ratio": gpu_used as f64 / gpu_total as f64,
        "queue_depth": scheduler_waiting,
        "kv_cache_free_blocks": free_blocks,
        "kv_cache_total_blocks": total_blocks,
    });

    if is_ready {
        (StatusCode::OK, Json(details))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(details))
    }
}
```

网关通过 readiness 探针判断该实例是否可以接收请求。当 GPU 内存水位过高或队列过深时返回 503，网关自动摘除实例。

#### 3.2.3 OOM 防护与自我保护

**当前状态**：**无 OOM 防护**。GPU 内存耗尽时直接 panic。

```rust
// mistralrs-core/src/engine/memory_guard.rs（新增）
pub struct MemoryGuard {
    gpu_high_watermark: f32,       // 0.90 — 告警
    gpu_critical_watermark: f32,   // 0.95 — 拒绝新请求
    check_interval: Duration,
}

impl MemoryGuard {
    /// 嵌入 Engine 主循环
    pub fn should_accept_new_request(&self) -> bool {
        let usage = MemoryUsage::new(None);
        let ratio = usage.gpu_used as f32 / usage.gpu_total as f32;

        if ratio > self.gpu_critical_watermark {
            tracing::error!(gpu_ratio = ratio, "GPU memory critical, rejecting request");
            false
        } else {
            if ratio > self.gpu_high_watermark {
                tracing::warn!(gpu_ratio = ratio, "GPU memory high watermark");
                // 主动触发空闲 KV Cache 清理
                self.evict_idle_caches();
            }
            true
        }
    }
}
```

**在 Engine 的 handle_request 中集成**：

```rust
// mistralrs-core/src/engine/mod.rs
fn handle_request(&mut self, request: Request) {
    if !self.memory_guard.should_accept_new_request() {
        // 通过 response channel 返回 503
        let _ = request.response_sender.send(
            Response::InternalError("GPU memory exhausted, please retry".into())
        );
        return;
    }
    // 正常处理...
}
```

#### 3.2.4 优雅启停

**当前状态**：没有 drain 机制，直接终止会丢失进行中的请求。

```rust
// mistralrs-core/src/engine/lifecycle.rs（新增）
pub struct EngineLifecycle {
    state: Arc<AtomicU8>,        // 0=Starting, 1=Ready, 2=Draining, 3=Stopped
    inflight: Arc<AtomicUsize>,  // 在途请求计数
    drain_timeout: Duration,
}

impl EngineLifecycle {
    pub async fn graceful_shutdown(&self) {
        // 1. 标记 Draining — readiness 探针返回 503
        //    网关不再发送新请求
        self.state.store(2, Ordering::SeqCst);

        // 2. 等待在途请求完成（带超时）
        let deadline = Instant::now() + self.drain_timeout;
        while self.inflight.load(Ordering::Relaxed) > 0 {
            if Instant::now() > deadline {
                tracing::warn!("Drain timeout, {} requests still inflight",
                    self.inflight.load(Ordering::Relaxed));
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // 3. 停止
        self.state.store(3, Ordering::SeqCst);
    }
}
```

**在 server main 中注册 shutdown hook**：

```rust
// mistralrs-server/src/main.rs
let lifecycle = engine_lifecycle.clone();
tokio::spawn(async move {
    tokio::signal::ctrl_c().await.unwrap();
    lifecycle.graceful_shutdown().await;
    std::process::exit(0);
});
```

#### 3.2.5 结构化日志

**当前状态**：使用 `tracing` 但输出是非结构化的文本格式。

```rust
// 替换为 JSON 格式输出，便于 ELK/Loki 采集
fn init_structured_logging() {
    let json_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true);

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(json_layer)
        .init();
}

// 日志输出示例：
// {"timestamp":"2026-02-06T10:30:00Z","level":"INFO","target":"engine",
//  "message":"Request completed","request_id":"req-123","model":"llama-70b",
//  "prompt_tokens":150,"completion_tokens":89,"latency_ms":1200}
```

#### 3.2.6 请求超时与取消

**当前状态**：请求一旦进入调度器，**无法超时取消**。

```rust
// 在 Engine 中增加超时检测
// mistralrs-core/src/sequence.rs 中增加字段
pub struct Sequence {
    // ... 现有字段
    pub deadline: Option<Instant>,  // 新增：请求截止时间
}

// Engine 主循环中检查
for seq in running_seqs.iter_mut() {
    if let Some(deadline) = seq.deadline {
        if Instant::now() > deadline {
            seq.set_state(SequenceState::Done(FinishReason::Timeout));
            // 通过 response channel 返回超时错误
        }
    }
}
```

### 3.3 推理性能优化

这些是推理引擎自身应该做的优化，与网关无关。

#### 3.3.1 默认开启 PagedAttention

```rust
// 当前 PagedAttention 是可选的，企业部署应默认开启
// 修改 mistralrs-server/src/main.rs 的默认值
pub struct ServerArgs {
    #[arg(long, default_value_t = true)]   // 改为 true
    pub paged_attention: bool,
    // ...
}
```

#### 3.3.2 Prefix Caching 调优

企业场景下大量请求共享 system prompt，Prefix Caching 效果显著：

```toml
# 推荐配置
[paged_attention]
prefix_caching = true
prefix_cache_n = 64              # 缓存更多前缀（默认 16）
```

**预估效果**：相同 system prompt 的请求，prefill 阶段计算量减少 50-80%。

#### 3.3.3 量化策略推荐

| 场景 | GPU | 模型 | 推荐量化 | 备注 |
|------|-----|------|---------|------|
| 高精度 | A100 80G | 7B-13B | FP16 / BF16 | 无精度损失 |
| 均衡 | A100 80G | 70B | Q8_0 / AFQ8 | 精度损失 <1% |
| 高吞吐 | A100 40G | 70B | Q4_K / GPTQ-4bit | 精度损失 2-3% |
| 低成本 | RTX 4090 | 70B | Q4_K + CPU offload | 部分层在 CPU |
| 多卡并行 | 4×A100 | 70B+ | FP16 + NCCL | 张量并行 |

#### 3.3.4 KV Cache 压缩

```rust
// 使用 FP8 KV Cache，内存减半
PagedAttentionMetaBuilder::default()
    .with_cache_type(PagedCacheType::F8E4M3)  // FP8
    .with_block_size(16)
    .build()
```

FP8 KV Cache 在 Ampere+ GPU 上可将 KV 缓存内存减少约 50%，从而在同等 GPU 显存下服务更多并发请求。

#### 3.3.5 投机解码（低延迟场景）

对于延迟敏感场景（如实时聊天），投机解码可显著降低延迟：

```rust
// 使用小模型做 draft，大模型做 verify
SpeculativeLoader::new(
    target_loader,   // 70B 大模型
    draft_loader,    // 7B 小模型
    gamma: 5,        // 每次投机 5 个 token
)
```

### 3.4 引擎稳定性加固

#### 3.4.1 Lock Poisoning 防护

**当前问题**：代码中大量使用 `.unwrap()` 获取锁，lock panic 会导致进程崩溃。

```rust
// 当前代码（危险）：
let mut tasks = self.tasks.write().unwrap();

// 改为安全版本：
let mut tasks = self.tasks.write()
    .unwrap_or_else(|poisoned| {
        tracing::error!("Lock poisoned, recovering...");
        poisoned.into_inner()  // 恢复被污染的锁
    });
```

#### 3.4.2 Channel 容量调优

**当前问题**：SSE 流式 channel buffer 固定为 10000，可能不够或浪费。

```rust
// 改为可配置
pub struct EngineConfig {
    pub channel_buffer_size: usize,  // 默认 10000，可调
    pub sse_keepalive_secs: u64,     // 默认 10，可调
}
```

#### 3.4.3 请求 ID 透传

**当前问题**：请求无法追踪。网关分配的 request_id 需要透传到引擎内部。

```rust
// 在 Request 中增加 request_id
pub struct NormalRequest {
    pub request_id: Option<String>,  // 新增：网关透传的请求 ID
    // ... 现有字段
}

// 在 Engine 日志和 metrics 中关联 request_id
tracing::info!(
    request_id = %req.request_id.as_deref().unwrap_or("unknown"),
    model = %model_id,
    "Processing request"
);
```

---

## 四、模型转换服务独立方案

### 4.1 需要抽离的功能

分析 mistral.rs 源码后，以下功能属于"模型准备"而非"模型推理"，应独立为模型转换服务：

| 功能 | 当前位置 | 抽离方式 |
|------|---------|---------|
| **模型量化 (ISQ)** | `mistralrs-core/src/pipeline/isq.rs` | 核心算法抽取 |
| **UQFF 序列化** | ISQ pipeline + Loader | 独立命令 |
| **CLI quantize 命令** | `mistralrs-cli/src/commands/quantize.rs` | 独立二进制 |
| **硬件调优 (tune)** | `mistralrs-cli/src/commands/tune.rs` | 独立 API |
| **系统诊断 (doctor)** | `mistralrs-cli/src/commands/doctor.rs` | 独立 API |
| **缓存管理 (cache)** | `mistralrs-cli/src/commands/cache.rs` | 独立命令 |
| **模型下载** | `Loader::load_model_from_hf()` 的下载部分 | 独立命令 |

### 4.2 模型转换服务架构

```
┌─────────────────────────────────────────────┐
│            模型管理服务 (上层)                │
│  模型注册 / 版本管理 / 部署调度               │
└─────────────┬───────────────────────────────┘
              │ API 调用
┌─────────────▼───────────────────────────────┐
│       模型转换服务 (mistralrs-converter)      │
│                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ 量化转换  │ │ 格式转换  │ │ 硬件调优推荐 │ │
│  │          │ │          │ │              │ │
│  │ HF→UQFF │ │ HF→GGUF  │ │ 自动选量化   │ │
│  │ ISQ量化  │ │ GGUF→UQFF│ │ 设备映射推荐 │ │
│  │ Q4/Q8/FP8│ │          │ │ 内存估算     │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
│                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ 模型下载  │ │ 缓存管理  │ │ 系统诊断    │ │
│  │          │ │          │ │              │ │
│  │ HF Hub   │ │ 清理/列表 │ │ GPU检测     │ │
│  │ 本地路径  │ │ 磁盘空间  │ │ 驱动检查    │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
└──────────────────────────────────────────────┘
```

### 4.3 Crate 结构

```
mistralrs-converter/
├── Cargo.toml
├── src/
│   ├── main.rs              # CLI 入口（独立二进制）
│   ├── lib.rs               # 库接口（供模型管理服务调用）
│   ├── quantize.rs          # 量化转换
│   ├── tune.rs              # 硬件调优
│   ├── doctor.rs            # 系统诊断
│   ├── cache.rs             # 缓存管理
│   ├── download.rs          # 模型下载
│   └── format_convert.rs    # 格式转换
└── proto/
    └── converter.proto       # 如果需要 gRPC 接口
```

### 4.4 抽离方案

#### 4.4.1 抽取 quantize 逻辑

当前 `mistralrs quantize` 命令的核心流程：

```
加载模型 (Loader) → 应用 ISQ 量化 → 序列化为 UQFF
```

**抽取方式**：将 `MistralRsForServerBuilder` 的 `write_uqff` 路径封装为独立函数：

```rust
// mistralrs-converter/src/quantize.rs
pub struct QuantizeConfig {
    pub model_source: ModelSource,         // HF repo 或本地路径
    pub model_type: ModelType,             // Auto / Text / Vision / Embedding
    pub isq_type: IsqType,                // Q4_K, Q8_0, HQQ4 等
    pub output_path: PathBuf,             // UQFF 输出路径
    pub imatrix_path: Option<PathBuf>,    // 可选的 imatrix 校准文件
    pub calibration_file: Option<PathBuf>,// 可选的校准数据
}

pub async fn quantize_model(config: QuantizeConfig) -> Result<QuantizeReport> {
    // 复用 mistralrs-core 的 Loader + ISQ 逻辑
    // 但不启动推理服务，只做 load → quantize → serialize
    let loader = build_loader(&config)?;
    let pipeline = loader.load_model_from_hf(
        // ... 参数
        Some(config.isq_type),
    ).await?;

    // 写入 UQFF
    // 返回量化报告（大小、精度、速度估算等）
    Ok(QuantizeReport { ... })
}
```

#### 4.4.2 抽取 tune 逻辑

```rust
// mistralrs-converter/src/tune.rs
pub async fn auto_tune(request: AutoTuneRequest) -> Result<AutoTuneResult> {
    // 复用 mistralrs_core::auto_tune()
    // 返回推荐的量化方案和设备映射
    mistralrs_core::auto_tune(request).await
}
```

#### 4.4.3 抽取 doctor 逻辑

```rust
// mistralrs-converter/src/doctor.rs
pub fn run_diagnostics() -> DoctorReport {
    // 复用 mistralrs_core::run_doctor()
    mistralrs_core::run_doctor()
}
```

### 4.5 模型转换服务 API

```rust
// CLI 模式
mistralrs-converter quantize \
    --model meta-llama/Llama-3.2-70B \
    --isq Q4_K \
    --output /models/llama-70b-q4k.uqff

mistralrs-converter tune \
    --model meta-llama/Llama-3.2-70B \
    --profile balanced

mistralrs-converter doctor

mistralrs-converter download \
    --model meta-llama/Llama-3.2-70B \
    --output /models/llama-70b/
```

```rust
// 库模式（供模型管理服务调用）
use mistralrs_converter::{quantize_model, auto_tune, QuantizeConfig};

let report = quantize_model(QuantizeConfig {
    model_source: ModelSource::HuggingFace("meta-llama/Llama-3.2-70B".into()),
    isq_type: IsqType::Q4K,
    output_path: "/models/llama-70b-q4k.uqff".into(),
    ..Default::default()
}).await?;

println!("Output size: {} GB", report.output_size_bytes / 1_000_000_000);
```

### 4.6 依赖关系

```
mistralrs-converter
├── mistralrs-core (模型加载和 ISQ 量化核心逻辑)
├── mistralrs-quant (量化类型定义和序列化)
├── hf-hub (HuggingFace 模型下载)
├── candle-core (张量运算)
└── clap (CLI 解析)
```

**不依赖**：`mistralrs-server-core`（HTTP 服务）、`mistralrs-paged-attn`（推理优化）

---

## 五、统一推理网关设计要点

网关是独立于推理引擎的 Rust 服务。这里列出关键设计要点，确保与 mistral.rs 引擎配合良好。

### 5.1 网关核心能力

```rust
// 网关核心结构（独立项目，不在 mistral.rs 仓库内）
pub struct InferenceGateway {
    backends: Vec<BackendConfig>,           // 多后端引擎
    router: Router,                          // 路由策略
    rate_limiter: RateLimiter,              // 限流器
    auth: AuthProvider,                      // 认证
    audit_logger: AuditLogger,              // 审计
    metrics: MetricsCollector,              // 计量
}
```

### 5.2 网关与引擎通信协议

**推荐方案**：直接使用 **HTTP + OpenAI 兼容协议**

**理由**：
1. mistral.rs 已实现完整的 OpenAI 兼容 HTTP API
2. vLLM、SGLang 同样使用 OpenAI 兼容 HTTP API
3. 统一 HTTP 协议使网关可以无差别地对接所有引擎
4. HTTP/2 性能足够，gRPC 的额外复杂度不值得

```
网关 ──HTTP/2──→ mistral.rs  (OpenAI API)
网关 ──HTTP/2──→ vLLM        (OpenAI API)
网关 ──HTTP/2──→ SGLang      (OpenAI API)
```

**如果确实需要 gRPC**（例如内部通信有强需求），在 mistral.rs 上增加 tonic gRPC 层，但建议**先评估 HTTP/2 是否足够**。

### 5.3 网关路由策略

```rust
// 基于引擎状态的智能路由
pub struct WeightedRouter {
    backends: Vec<Backend>,
}

impl WeightedRouter {
    pub fn select_backend(&self, model: &str) -> &Backend {
        // 1. 过滤出支持该模型的后端
        let candidates = self.backends.iter()
            .filter(|b| b.supports_model(model) && b.is_ready());

        // 2. 按权重选择（基于引擎的 /health/ready 返回的指标）
        //    权重 = 1 / (queue_depth + running_seqs * alpha)
        candidates.min_by_key(|b| {
            b.last_health.queue_depth + b.last_health.running_seqs * 2
        })
    }
}
```

网关定期（如每秒）调用各引擎的 `/health/ready` 端点获取指标，用于路由决策。

### 5.4 网关需要的引擎 API

| 端点 | 用途 | 当前状态 |
|------|------|---------|
| `POST /v1/chat/completions` | 推理 | ✅ 已有 |
| `POST /v1/completions` | 文本补全 | ✅ 已有 |
| `POST /v1/embeddings` | 嵌入 | ✅ 已有 |
| `POST /v1/images/generations` | 图像生成 | ✅ 已有 |
| `POST /v1/audio/speech` | 语音合成 | ✅ 已有 |
| `GET /v1/models` | 模型列表 | ✅ 已有 |
| `GET /health/live` | 存活探针 | ❌ **需新增** |
| `GET /health/ready` | 就绪探针（含指标） | ❌ **需新增** |
| `GET /metrics` | Prometheus 指标 | ❌ **需新增** |
| `POST /v1/models/load` | 加载模型 | ✅ 已有 (reload) |
| `POST /v1/models/unload` | 卸载模型 | ✅ 已有 |
| `POST /v1/models/status` | 模型状态 | ✅ 已有 |

**关键结论**：只需在引擎侧新增 3 个端点（liveness、readiness、metrics），其余都可直接使用。

---

## 六、推理引擎与网关的接口协议

### 6.1 请求头透传

网关需要透传以下请求头到引擎：

```
X-Request-ID: <uuid>            # 请求追踪 ID
X-Priority: <high|normal|low>   # 请求优先级（可选）
X-Timeout-Ms: <milliseconds>    # 请求超时时间
```

**引擎侧修改**：

```rust
// mistralrs-server-core/src/chat_completion.rs
async fn chat_completion(
    headers: HeaderMap,
    State(state): State<Arc<SharedState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // 提取网关透传的头
    let request_id = headers.get("X-Request-ID")
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    let timeout = headers.get("X-Timeout-Ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_millis);

    // 传入 Engine
    let core_request = NormalRequest {
        request_id,
        deadline: timeout.map(|t| Instant::now() + t),
        // ...
    };
}
```

### 6.2 响应增强

引擎响应中增加以下字段供网关使用：

```json
{
  "id": "chatcmpl-abc123",
  "model": "llama-70b",
  "choices": [...],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 89,
    "total_tokens": 239
  },
  // 新增：引擎侧指标（嵌入 response headers 或 body）
  "x_engine_metrics": {
    "queue_wait_ms": 12,
    "inference_ms": 1188,
    "ttft_ms": 45,
    "tps": 74.2
  }
}
```

或通过 Response Header 返回：

```
X-Queue-Wait-Ms: 12
X-Inference-Ms: 1188
X-TTFT-Ms: 45
X-Tokens-Per-Second: 74.2
```

---

## 七、多引擎统一适配层

### 7.1 适配层设计

网关内部的统一适配层，屏蔽 mistral.rs / vLLM / SGLang 的差异：

```rust
// 网关代码（非 mistral.rs 仓库）
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// 推理
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse>;

    /// 流式推理
    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<impl Stream<Item = ChatCompletionChunk>>;

    /// 嵌入
    async fn embedding(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse>;

    /// 健康检查
    async fn health_check(&self) -> BackendHealth;

    /// 支持的模型列表
    async fn list_models(&self) -> Vec<ModelInfo>;

    /// 引擎类型
    fn engine_type(&self) -> EngineType;
}

pub enum EngineType {
    MistralRs,
    VLlm,
    SGLang,
    Custom(String),
}
```

### 7.2 各引擎差异对照

| 能力 | mistral.rs | vLLM | SGLang |
|------|-----------|------|--------|
| API 协议 | OpenAI 兼容 | OpenAI 兼容 | OpenAI 兼容 |
| 流式 | SSE | SSE | SSE |
| 文本生成 | ✅ | ✅ | ✅ |
| 视觉模型 | ✅ | ✅ | ✅ |
| 图像生成 | ✅ (Flux) | ❌ | ❌ |
| 语音合成 | ✅ | ❌ | ❌ |
| 嵌入 | ✅ | ✅ | ✅ |
| GGUF 量化 | ✅ | ❌ | ❌ |
| GPTQ/AWQ | ✅ | ✅ | ✅ |
| PagedAttention | ✅ | ✅ | ✅ |
| LoRA 热切换 | ✅ | ✅ | ✅ |
| 投机解码 | ✅ | ✅ | ✅ |
| 语言 | Rust | Python | Python |
| GPU 内存效率 | 高 | 高 | 高 |
| CPU 推理 | ✅ (GGUF) | ❌ | ❌ |
| Prometheus | ❌ **需加** | ✅ 内置 | ✅ 内置 |
| 健康检查 | 基础 **需增强** | ✅ | ✅ |

**mistral.rs 的独特优势**：
- Rust 实现，内存安全，无 GIL 限制
- 原生 CPU 推理（GGUF）
- 图像生成 + 语音合成支持
- 极低的部署依赖（单二进制）

---

## 八、分阶段实施计划

### 阶段一：推理引擎生产加固（3-4 周）

**目标**：让 mistral.rs 达到生产部署标准

| 任务 | 工作量 | 说明 |
|------|--------|------|
| 新增 `/metrics` Prometheus 端点 | 5 天 | Engine 内嵌采集点 + HTTP 暴露 |
| 新增 `/health/live` + `/health/ready` | 2 天 | 含 GPU/队列/KV Cache 详细状态 |
| OOM 防护 (MemoryGuard) | 3 天 | GPU 水位监控 + 拒绝新请求 |
| 优雅启停 (drain) | 2 天 | shutdown hook + 在途请求等待 |
| 结构化 JSON 日志 | 1 天 | tracing-subscriber JSON layer |
| 请求超时与取消 | 2 天 | 序列 deadline + Engine 循环检查 |
| 请求 ID 透传 | 1 天 | Header 提取 → Engine → 日志/metrics |
| Lock poisoning 加固 | 1 天 | 全局 unwrap() → unwrap_or_else() |
| 集成测试 | 3 天 | 端到端推理 + 并发 + 超时 |

**交付物**：可部署的推理引擎 Docker 镜像

### 阶段二：模型转换服务独立（2-3 周）

**目标**：将模型准备工具从推理引擎中独立出来

| 任务 | 工作量 | 说明 |
|------|--------|------|
| 创建 mistralrs-converter crate | 2 天 | 项目骨架 + Cargo.toml |
| 抽取 quantize 逻辑 | 3 天 | 封装为 lib + CLI |
| 抽取 tune 逻辑 | 2 天 | 封装为 lib + CLI |
| 抽取 doctor 逻辑 | 1 天 | 封装为 lib + CLI |
| 抽取 download/cache 逻辑 | 2 天 | 封装为 lib + CLI |
| 定义转换服务 API | 2 天 | 库接口 + 可选 HTTP/gRPC |
| 测试：量化 + 加载验证 | 3 天 | 端到端验证 |

**交付物**：独立的 `mistralrs-converter` 二进制 + 库

### 阶段三：推理网关集成（3-4 周）

**目标**：网关与 mistral.rs 引擎联调通过

| 任务 | 工作量 | 说明 |
|------|--------|------|
| 网关适配层（mistral.rs 后端） | 3 天 | 实现 InferenceBackend trait |
| 网关路由策略（基于 /health/ready） | 3 天 | 队列感知的加权路由 |
| 网关 Prometheus 聚合 | 2 天 | 从各引擎拉取 + 聚合 |
| 网关流式透传 | 3 天 | SSE 转发 |
| 压力测试 | 5 天 | 全链路（网关→引擎）并发压测 |
| 故障注入测试 | 3 天 | 引擎重启/OOM/超时 场景 |

**交付物**：完整的 网关→引擎 联调报告

### 阶段四：多引擎适配 + 高级优化（持续）

| 任务 | 工作量 | 说明 |
|------|--------|------|
| 网关适配 vLLM 后端 | 2 天 | 复用 OpenAI 兼容 API |
| 网关适配 SGLang 后端 | 2 天 | 复用 OpenAI 兼容 API |
| 多 GPU 张量并行部署 (NCCL) | 1 周 | 70B+ 模型 |
| NPU 后端适配 | 8-12 周 | 参考 DEV_GUIDE.md 第 6 章 |
| 投机解码集成调优 | 1 周 | TTFT 优化 |
| Prefix Caching 策略优化 | 1 周 | 企业 system prompt 缓存 |

---

## 九、风险与对策

### 9.1 上游更新同步

**风险**：修改 mistralrs-core 后难以合并上游新模型支持。

**对策**：
- 引擎侧改动尽量以**新增文件**方式实现（`metrics.rs`、`memory_guard.rs`、`lifecycle.rs`），不修改现有核心逻辑
- 对现有文件的修改限制在**埋点**级别（如 metrics 采集点），不改变业务逻辑
- 定期从上游 cherry-pick 新模型 + bugfix

### 9.2 推理引擎单点故障

**风险**：单个推理引擎实例故障导致请求失败。

**对策**：
- 网关层做多实例健康检查和自动摘除
- 推理引擎实现优雅启停，Kubernetes 滚动更新时不丢请求
- readiness 探针在模型加载完成前返回 503

### 9.3 GPU 内存竞争

**风险**：多模型加载时 GPU 内存不足。

**对策**：
- MemoryGuard 监控 GPU 水位
- 模型管理服务在下发加载指令前，先通过 `/health/ready` 检查目标引擎的显存余量
- 引擎拒绝加载时返回明确错误码，模型管理服务可调度到其他实例

### 9.4 流式请求中断

**风险**：SSE 流式推理过程中网络断开，引擎仍在生成 token 浪费算力。

**对策**：
- 检测 SSE 连接断开后标记序列为取消状态
- 在 Engine 主循环中检查 response channel 是否已关闭
```rust
// 检查 response sender 是否还有接收者
if request.response_tx.is_closed() {
    seq.set_state(SequenceState::Done(FinishReason::Cancelled));
}
```

### 9.5 冷启动延迟

**风险**：大模型加载需要数分钟，期间引擎无法服务。

**对策**：
- 模型预热：加载完成后执行几次 warmup 推理
- Kubernetes 中设置合理的 `initialDelaySeconds`
- 使用 UQFF 预量化模型文件，减少加载时的量化时间
- readiness 探针在预热完成后才返回 200

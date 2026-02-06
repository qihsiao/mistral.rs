# mistral.rs 企业级私有化推理引擎 — 二次开发方案

> 本文档基于对 mistral.rs 源码的深度分析，提出将其改造为企业级私有化部署推理引擎的完整方案，包括功能裁剪、gRPC 网关集成、缺失功能补充、性能优化建议，以及分阶段实施计划。

---

## 目录

- [一、总体架构设计](#一总体架构设计)
- [二、功能裁剪方案](#二功能裁剪方案)
- [三、必须补充的企业级功能](#三必须补充的企业级功能)
  - [3.1 gRPC 服务层](#31-grpc-服务层)
  - [3.2 认证与鉴权](#32-认证与鉴权)
  - [3.3 可观测性体系](#33-可观测性体系)
  - [3.4 流量控制与限流](#34-流量控制与限流)
  - [3.5 优雅启停与健康检查](#35-优雅启停与健康检查)
  - [3.6 OOM 防护与内存管理](#36-oom-防护与内存管理)
  - [3.7 请求队列与背压](#37-请求队列与背压)
  - [3.8 多模型生命周期管理](#38-多模型生命周期管理)
  - [3.9 审计日志](#39-审计日志)
  - [3.10 配置热更新](#310-配置热更新)
- [四、性能优化建议](#四性能优化建议)
  - [4.1 推理性能优化](#41-推理性能优化)
  - [4.2 吞吐与延迟优化](#42-吞吐与延迟优化)
  - [4.3 内存优化](#43-内存优化)
  - [4.4 多实例部署优化](#44-多实例部署优化)
- [五、分阶段实施计划](#五分阶段实施计划)
- [六、目标架构蓝图](#六目标架构蓝图)
- [七、风险与对策](#七风险与对策)

---

## 一、总体架构设计

### 目标架构

```
                     企业内部网络
                         │
           ┌─────────────▼──────────────┐
           │      企业推理网关           │
           │  (路由/鉴权/限流/审计)      │
           │  已有系统，不在本方案范围    │
           └─────────────┬──────────────┘
                         │ gRPC (TLS)
           ┌─────────────▼──────────────┐
           │    gRPC Service Layer      │
           │  (tonic-based gRPC 服务)   │
           │  ┌────────────────────┐    │
           │  │ Proto 定义:        │    │
           │  │ - ChatCompletion   │    │
           │  │ - Completion       │    │
           │  │ - Embedding        │    │
           │  │ - ModelManagement  │    │
           │  │ - HealthCheck      │    │
           │  └────────────────────┘    │
           └─────────────┬──────────────┘
                         │ Rust 函数调用
           ┌─────────────▼──────────────┐
           │  mistralrs-core (裁剪后)   │
           │  ┌──────────────────────┐  │
           │  │ Engine + Scheduler   │  │
           │  │ Pipeline (文本推理)  │  │
           │  │ KV Cache / PagedAttn │  │
           │  │ Quantization         │  │
           │  │ DeviceMapper         │  │
           │  └──────────────────────┘  │
           └─────────────┬──────────────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
           CPU        CUDA      Metal/NPU
```

### 设计原则

1. **最小化原则**：只保留文本推理核心能力，去除非必需模态
2. **gRPC 优先**：企业网关通过 gRPC 连接，保留 HTTP 作为调试接口
3. **可观测性内建**：metrics/tracing/logging 从第一天就集成
4. **防御性设计**：OOM 防护、背压、优雅降级内建于系统
5. **运维友好**：配置热更新、健康检查、模型热加载

---

## 二、功能裁剪方案

### 裁剪总览

| 模块 | 当前状态 | 裁剪决策 | 耦合度 | 工作量 |
|------|---------|---------|--------|--------|
| **mistralrs-web-chat/** | 已废弃 | **删除** | 1/10 | 0.5 天 |
| **mistralrs-bench/** | 已废弃 | **删除** | 1/10 | 0.5 天 |
| **mistralrs-mcp/** | MCP 工具协议客户端 | **删除** | 2/10 | 1 天 |
| **Diffusion 图像生成** | 编译时包含 | **删除** | 4/10 | 3 天 |
| **Speech 语音合成** | 编译时包含 | **删除** | 4/10 | 3 天 |
| **mistralrs-audio/** | 音频处理 | **删除** | 4/10 | 1 天 |
| **mistralrs-vision/** | 视觉处理 | **可选保留** | 4/10 | 3 天 |
| **Vision 模型** | 18+ 视觉语言模型 | **可选保留** | 4/10 | 3 天 |
| **GGML Pipeline** | 旧格式支持 | **删除** | 5/10 | 2 天 |
| **Speculative 推测解码** | 可选 Pipeline | **删除** | 3/10 | 1 天 |
| **AnyMoE** | 动态 MoE | **删除** | 5/10 | 3 天 |
| **CLI run 命令** | 交互式模式 | **删除** | 2/10 | 1 天 |
| **mistralrs-pyo3/** | Python SDK | **剥离为可选** | 9/10 | 不删，Feature 门控 |
| **X-LoRA** | LoRA 适配器 | **保留** | 9/10 | 耦合太深 |
| **HTTP Server** | OpenAI 兼容 API | **保留（调试用）** | — | — |
| **Embedding Pipeline** | 嵌入模型 | **保留** | — | — |

### 裁剪后保留的核心

```
mistral.rs-enterprise/
├── mistralrs-core/            # 核心推理（仅文本+嵌入）
│   ├── src/models/            # 27+ 文本模型（全部保留）
│   ├── src/pipeline/
│   │   ├── normal.rs          # 标准文本 Pipeline ✅
│   │   ├── gguf.rs            # GGUF Pipeline ✅
│   │   ├── embedding.rs       # 嵌入 Pipeline ✅
│   │   ├── vision.rs          # 可选保留 🔧
│   │   ├── ggml.rs            # ❌ 删除
│   │   ├── diffusion.rs       # ❌ 删除
│   │   ├── speech.rs          # ❌ 删除
│   │   ├── speculative.rs     # ❌ 删除
│   │   └── amoe.rs            # ❌ 删除
│   ├── src/engine/            # 引擎（完整保留）
│   ├── src/scheduler/         # 调度器（完整保留）
│   ├── src/attention/         # 注意力（完整保留）
│   ├── src/kv_cache/          # 缓存（完整保留）
│   ├── src/paged_attention/   # 分页注意力（完整保留）
│   └── src/sampler.rs         # 采样器（完整保留）
├── mistralrs-quant/           # 量化（完整保留）
├── mistralrs-paged-attn/      # PagedAttention 内核（完整保留）
├── mistralrs-grpc/            # 🆕 gRPC 服务层（新增）
├── mistralrs-enterprise/      # 🆕 企业功能（新增）
├── mistralrs-server-core/     # HTTP API（保留，调试用）
├── mistralrs/                 # Rust SDK（保留）
└── mistralrs-cli/             # CLI（精简：仅 serve + bench + quantize）
```

### 裁剪具体步骤

#### 第一步：删除已废弃模块（0.5 天）

```toml
# 从根 Cargo.toml 的 workspace members 中删除：
# - "mistralrs-web-chat"
# - "mistralrs-bench"
```

#### 第二步：删除 MCP 客户端（1 天）

```rust
// mistralrs-core/src/lib.rs 中删除：
// pub use mistralrs_mcp::{McpClient, ...};

// mistralrs-core/Cargo.toml 中删除：
// mistralrs-mcp = { path = "../mistralrs-mcp" }

// 从 EngineConfig 和 AddModelConfig 中移除 mcp_config 字段
```

#### 第三步：删除非文本模态（5-7 天）

关键修改点：

```rust
// 1. mistralrs-core/src/lib.rs 中删除相关导出
// 删除: DiffusionLoader*, SpeechLoader*, SpeculativeLoader*
// 删除: AnyMoeLoader*, AnyMoeConfig, AnyMoeExpertType

// 2. mistralrs-core/src/pipeline/mod.rs 中删除模块声明
// 删除: mod diffusion; mod speech; mod speculative; mod amoe;

// 3. ModelCategory 枚举精简
pub enum ModelCategory {
    Text,       // 保留
    Embedding,  // 保留
    Vision,     // 可选保留
    // Diffusion  — 删除
    // Speech     — 删除
}

// 4. 删除整个目录
// rm -rf mistralrs-core/src/diffusion_models/
// rm -rf mistralrs-core/src/speech_models/
// rm -rf mistralrs-core/src/amoe/
// rm -rf mistralrs-audio/

// 5. 从 auto.rs 中删除对应的自动检测逻辑
```

#### 第四步：删除 GGML Pipeline（2 天）

```rust
// 删除 mistralrs-core/src/pipeline/ggml.rs
// 从 lib.rs 删除 GGMLLoader* 导出
// 从 CLI args 删除 GGML 相关参数
```

### 预计裁剪效果

| 指标 | 裁剪前 | 裁剪后 | 减少 |
|------|--------|--------|------|
| Crate 数量 | 13 | 8 | 38% |
| 编译时间（估算） | 100% | ~70% | 30% |
| 二进制大小（估算） | 100% | ~75% | 25% |
| 外部依赖 | ~300+ | ~250+ | ~15% |
| 攻击面 | 全模态 | 仅文本 | 显著缩小 |

---

## 三、必须补充的企业级功能

### 3.1 gRPC 服务层

**当前状态**：项目只有 HTTP REST API（OpenAI 兼容），**没有任何 gRPC 支持**。

**方案**：新建 `mistralrs-grpc` crate，使用 [tonic](https://github.com/hyperium/tonic) 框架。

#### Proto 定义

```protobuf
syntax = "proto3";
package mistralrs.v1;

// ========== 推理服务 ==========
service InferenceService {
  // 同步推理
  rpc ChatCompletion(ChatCompletionRequest)
      returns (ChatCompletionResponse);

  // 流式推理（服务端流）
  rpc ChatCompletionStream(ChatCompletionRequest)
      returns (stream ChatCompletionChunk);

  // 文本补全
  rpc Completion(CompletionRequest)
      returns (CompletionResponse);

  // 嵌入
  rpc Embedding(EmbeddingRequest)
      returns (EmbeddingResponse);
}

// ========== 管理服务 ==========
service ManagementService {
  // 模型管理
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
  rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
  rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);
  rpc ModelStatus(ModelStatusRequest) returns (ModelStatusResponse);

  // 健康检查（兼容 gRPC Health Checking Protocol）
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
  rpc Watch(HealthCheckRequest) returns (stream HealthCheckResponse);
}

// ========== 消息定义 ==========
message ChatCompletionRequest {
  string model = 1;
  repeated ChatMessage messages = 2;
  float temperature = 3;
  int32 max_tokens = 4;
  float top_p = 5;
  float frequency_penalty = 6;
  float presence_penalty = 7;
  repeated string stop = 8;
  bool stream = 9;
  string request_id = 10;       // 企业级：请求追踪 ID
  int32 priority = 11;          // 企业级：请求优先级
  map<string, string> metadata = 12;  // 企业级：业务元数据
}

message ChatMessage {
  string role = 1;
  string content = 2;
}

message ChatCompletionResponse {
  string id = 1;
  string model = 2;
  repeated Choice choices = 3;
  Usage usage = 4;
  int64 created = 5;
  int64 latency_ms = 6;        // 企业级：处理延迟
  string request_id = 7;
}

message Usage {
  int32 prompt_tokens = 1;
  int32 completion_tokens = 2;
  int32 total_tokens = 3;
}

message ChatCompletionChunk {
  string id = 1;
  string model = 2;
  repeated ChunkChoice choices = 3;
  string request_id = 4;
}

message HealthCheckResponse {
  enum ServingStatus {
    UNKNOWN = 0;
    SERVING = 1;
    NOT_SERVING = 2;
    MODEL_LOADING = 3;
  }
  ServingStatus status = 1;
  map<string, string> details = 2;  // GPU 内存、队列深度等
}
```

#### 实现要点

```rust
// mistralrs-grpc/src/lib.rs
use tonic::{transport::Server, Request, Response, Status};
use mistralrs_core::MistralRs;

pub struct InferenceServiceImpl {
    mistralrs: Arc<MistralRs>,
    metrics: Arc<MetricsCollector>,
}

#[tonic::async_trait]
impl InferenceService for InferenceServiceImpl {
    async fn chat_completion(
        &self,
        request: Request<ChatCompletionRequest>,
    ) -> Result<Response<ChatCompletionResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();

        // 1. 构建 mistralrs_core::Request
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let core_request = self.build_core_request(&req, tx)?;

        // 2. 发送到引擎
        self.mistralrs.send_request(core_request)
            .map_err(|e| Status::internal(e.to_string()))?;

        // 3. 等待响应
        let response = rx.recv().await
            .ok_or_else(|| Status::internal("Engine closed"))?;

        // 4. 记录 metrics
        self.metrics.record_latency(start.elapsed());

        Ok(Response::new(self.to_proto_response(response, &req)?))
    }

    // 流式推理：使用 tonic 的 ServerStreaming
    type ChatCompletionStreamStream = ReceiverStream<Result<ChatCompletionChunk, Status>>;

    async fn chat_completion_stream(
        &self,
        request: Request<ChatCompletionRequest>,
    ) -> Result<Response<Self::ChatCompletionStreamStream>, Status> {
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        // 将 mistralrs 的流式 channel 桥接到 gRPC stream
        // ...
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
```

#### gRPC 服务配置

```toml
# config.toml
[grpc]
listen_addr = "0.0.0.0:50051"
tls_cert = "/etc/certs/server.pem"
tls_key = "/etc/certs/server.key"
max_concurrent_requests = 1000
keepalive_interval_secs = 30
max_message_size_mb = 64
```

### 3.2 认证与鉴权

**当前状态**：HTTP 服务器**没有任何认证机制**。

**方案**：实现 gRPC 拦截器 + HTTP 中间件双层认证。

```rust
// 方式 1：gRPC 拦截器（推荐用于网关连接）
pub struct AuthInterceptor {
    valid_tokens: Arc<RwLock<HashSet<String>>>,
}

impl tonic::service::Interceptor for AuthInterceptor {
    fn call(&mut self, req: Request<()>) -> Result<Request<()>, Status> {
        let token = req.metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| Status::unauthenticated("Missing authorization"))?;

        if !self.valid_tokens.read().unwrap().contains(token) {
            return Err(Status::unauthenticated("Invalid token"));
        }
        Ok(req)
    }
}

// 方式 2：mTLS（推荐用于生产环境网关连接）
// tonic 原生支持 mTLS，网关和推理引擎互相验证证书
let tls = ServerTlsConfig::new()
    .identity(server_identity)
    .client_ca_root(ca_cert);  // 要求客户端证书
```

**企业推荐**：网关与推理引擎之间使用 **mTLS**（双向 TLS），无需额外 token 验证。网关侧负责终端用户认证。

### 3.3 可观测性体系

**当前状态**：
- 有 `tracing` 日志（但无结构化输出）
- 有 `IntervalLogger`（仅控制台打印吞吐量）
- 无 Prometheus metrics
- 无分布式追踪

#### 3.3.1 Prometheus Metrics

```rust
// mistralrs-enterprise/src/metrics.rs
use prometheus::{
    IntCounter, IntGauge, Histogram, HistogramVec,
    Registry, Encoder, TextEncoder,
};

pub struct MetricsCollector {
    pub registry: Registry,

    // 请求指标
    pub requests_total: IntCounter,
    pub requests_active: IntGauge,
    pub request_duration: HistogramVec,  // 按 model, method 标签

    // 推理指标
    pub tokens_generated_total: IntCounter,
    pub prompt_tokens_total: IntCounter,
    pub tokens_per_second: Histogram,
    pub time_to_first_token: Histogram,   // TTFT
    pub time_per_output_token: Histogram, // TPOT

    // 引擎指标
    pub scheduler_waiting_seqs: IntGauge,
    pub scheduler_running_seqs: IntGauge,
    pub kv_cache_usage_ratio: Gauge,      // KV Cache 利用率
    pub paged_attn_blocks_used: IntGauge,
    pub paged_attn_blocks_free: IntGauge,

    // 系统指标
    pub gpu_memory_used_bytes: IntGauge,
    pub gpu_memory_total_bytes: IntGauge,
    pub cpu_memory_used_bytes: IntGauge,

    // 模型指标
    pub model_loaded: IntGauge,           // 已加载模型数
    pub model_load_duration: Histogram,   // 模型加载耗时
}
```

**暴露端点**：
```
GET /metrics                # Prometheus 抓取端点
GET /v1/system/metrics      # JSON 格式（调试用）
```

#### 3.3.2 分布式追踪

```rust
// 集成 OpenTelemetry
use opentelemetry::trace::{Tracer, SpanKind};
use tracing_opentelemetry::OpenTelemetryLayer;

// 在 Engine 的请求处理中埋入 Span
async fn handle_request(&self, request: Request) {
    let span = tracing::info_span!(
        "inference_request",
        request_id = %request.id,
        model = %request.model,
        otel.kind = "server",
    );
    let _enter = span.enter();

    // 子 Span：分别追踪各阶段
    let _tokenize_span = tracing::info_span!("tokenize").entered();
    // ...
    let _forward_span = tracing::info_span!("forward_pass").entered();
    // ...
    let _sample_span = tracing::info_span!("sampling").entered();
}
```

#### 3.3.3 结构化日志

```rust
// 替换现有 tracing 初始化
use tracing_subscriber::{fmt, EnvFilter, Layer};

fn init_logging() {
    let json_layer = fmt::layer()
        .json()                           // JSON 格式输出
        .with_target(true)
        .with_thread_ids(true)
        .with_span_list(true);

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(json_layer)
        .with(opentelemetry_layer)        // OTel 追踪
        .init();
}
```

### 3.4 流量控制与限流

**当前状态**：**无任何限流机制**。

```rust
// mistralrs-enterprise/src/rate_limiter.rs
use governor::{Quota, RateLimiter};

pub struct RequestLimiter {
    // 全局限流
    global_limiter: RateLimiter<NotKeyed, InMemoryState, DefaultClock>,

    // 每模型限流
    per_model_limiters: HashMap<String, RateLimiter<...>>,

    // 请求队列深度限制
    max_queue_depth: usize,

    // 并发请求限制
    max_concurrent: Arc<Semaphore>,
}

impl RequestLimiter {
    pub fn check(&self, model: &str) -> Result<(), RateLimitError> {
        // 1. 检查全局限流
        self.global_limiter.check()
            .map_err(|_| RateLimitError::GlobalLimitExceeded)?;

        // 2. 检查模型级限流
        if let Some(limiter) = self.per_model_limiters.get(model) {
            limiter.check()
                .map_err(|_| RateLimitError::ModelLimitExceeded)?;
        }

        Ok(())
    }

    pub async fn acquire_slot(&self) -> Result<OwnedSemaphorePermit, RateLimitError> {
        // 3. 获取并发槽位
        self.max_concurrent.clone().acquire_owned().await
            .map_err(|_| RateLimitError::ConcurrencyLimitExceeded)
    }
}
```

配置：
```toml
[rate_limit]
global_rps = 1000                    # 全局每秒请求数
max_concurrent_requests = 256        # 最大并发请求
max_queue_depth = 1024               # 最大排队深度
per_model_rps = 500                  # 每模型每秒请求数
```

### 3.5 优雅启停与健康检查

**当前状态**：
- 有 `/health` 端点（返回 "OK"，无详细状态）
- 有全局终止标志，但没有优雅 drain
- 无就绪探针（readiness probe）

```rust
// mistralrs-enterprise/src/lifecycle.rs

pub struct LifecycleManager {
    state: Arc<AtomicU8>,  // 0=Starting, 1=Ready, 2=Draining, 3=Stopped
    inflight: Arc<AtomicUsize>,
    drain_timeout: Duration,
}

impl LifecycleManager {
    /// Kubernetes liveness probe
    pub fn is_alive(&self) -> bool {
        self.state.load(Ordering::Relaxed) != 3 // 非 Stopped
    }

    /// Kubernetes readiness probe
    pub fn is_ready(&self) -> bool {
        self.state.load(Ordering::Relaxed) == 1 // Ready
    }

    /// 优雅关闭
    pub async fn graceful_shutdown(&self) {
        // 1. 标记为 Draining，不接受新请求
        self.state.store(2, Ordering::SeqCst);
        tracing::info!("Entering drain mode, waiting for inflight requests...");

        // 2. 等待现有请求完成（带超时）
        let deadline = Instant::now() + self.drain_timeout;
        while self.inflight.load(Ordering::Relaxed) > 0 {
            if Instant::now() > deadline {
                tracing::warn!(
                    "Drain timeout reached, {} requests still inflight",
                    self.inflight.load(Ordering::Relaxed)
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // 3. 标记为 Stopped
        self.state.store(3, Ordering::SeqCst);
        tracing::info!("Shutdown complete");
    }
}
```

gRPC 健康检查端点扩展（兼容 [gRPC Health Checking Protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md)）：

```rust
impl Health for HealthServiceImpl {
    async fn check(&self, _req: Request<HealthCheckRequest>)
        -> Result<Response<HealthCheckResponse>, Status>
    {
        let mut details = HashMap::new();

        // GPU 内存
        let mem = MemoryUsage::new(None);
        details.insert("gpu_memory_used_mb".into(), format!("{}", mem.gpu_used_mb()));
        details.insert("gpu_memory_free_mb".into(), format!("{}", mem.gpu_free_mb()));

        // 队列深度
        details.insert("queue_depth".into(),
            format!("{}", self.scheduler_waiting()));

        // KV Cache
        if let Some(engine) = self.block_engine() {
            details.insert("kv_cache_blocks_free".into(),
                format!("{}", engine.free_blocks()));
        }

        let status = if self.lifecycle.is_ready() {
            ServingStatus::Serving
        } else {
            ServingStatus::NotServing
        };

        Ok(Response::new(HealthCheckResponse { status: status as i32, details }))
    }
}
```

### 3.6 OOM 防护与内存管理

**当前状态**：
- 有 `MemoryUsage` 工具类查询 GPU/CPU 内存
- 有 PagedAttention 块分配，但**无 OOM 防护**
- 无内存水位告警

```rust
// mistralrs-enterprise/src/memory_guard.rs

pub struct MemoryGuard {
    gpu_high_watermark: f32,     // e.g. 0.90（90% 触发告警）
    gpu_critical_watermark: f32, // e.g. 0.95（95% 拒绝新请求）
    check_interval: Duration,
}

impl MemoryGuard {
    pub async fn run(&self, lifecycle: Arc<LifecycleManager>, metrics: Arc<MetricsCollector>) {
        loop {
            let usage = MemoryUsage::new(None);
            let ratio = usage.gpu_used as f32 / usage.gpu_total as f32;

            metrics.gpu_memory_used_bytes.set(usage.gpu_used as i64);

            if ratio > self.gpu_critical_watermark {
                tracing::error!(
                    gpu_usage_ratio = ratio,
                    "GPU memory critical! Rejecting new requests"
                );
                // 标记为不接受新请求
                lifecycle.set_overloaded(true);
            } else if ratio > self.gpu_high_watermark {
                tracing::warn!(
                    gpu_usage_ratio = ratio,
                    "GPU memory high watermark reached"
                );
                // 触发预防性 KV Cache 清理
                self.evict_idle_caches().await;
            } else {
                lifecycle.set_overloaded(false);
            }

            tokio::time::sleep(self.check_interval).await;
        }
    }

    async fn evict_idle_caches(&self) {
        // 清理空闲序列的 KV Cache 块
        // 优先释放最久未活跃的序列
    }
}
```

配置：
```toml
[memory]
gpu_high_watermark = 0.90
gpu_critical_watermark = 0.95
check_interval_ms = 1000
max_kv_cache_blocks = 0        # 0 = 自动计算
```

### 3.7 请求队列与背压

**当前状态**：
- 使用 MPSC 无界通道，**无背压机制**
- 如果推理速度跟不上请求速度，队列会无限增长

```rust
// 改造方案：有界通道 + 背压

pub struct RequestQueue {
    sender: tokio::sync::mpsc::Sender<Request>,     // 有界
    receiver: Arc<Mutex<tokio::sync::mpsc::Receiver<Request>>>,
    max_depth: usize,
    timeout: Duration,
}

impl RequestQueue {
    pub fn new(max_depth: usize, timeout: Duration) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(max_depth);
        Self { sender: tx, receiver: Arc::new(Mutex::new(rx)), max_depth, timeout }
    }

    pub async fn enqueue(&self, req: Request) -> Result<(), QueueError> {
        // 带超时的入队，防止无限等待
        match tokio::time::timeout(self.timeout, self.sender.send(req)).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err(QueueError::ChannelClosed),
            Err(_) => Err(QueueError::Timeout),  // 队列满，返回 429
        }
    }
}
```

**在 gRPC 层面的背压响应**：
```rust
// 队列满时返回 RESOURCE_EXHAUSTED
if queue.is_full() {
    return Err(Status::resource_exhausted(
        "Server overloaded, please retry later"
    ));
}
```

### 3.8 多模型生命周期管理

**当前状态**：已有基础的 load/unload/reload，但缺少企业级管理能力。

需要补充：

```rust
// mistralrs-enterprise/src/model_manager.rs

pub struct ModelManager {
    mistralrs: Arc<MistralRs>,
    model_configs: RwLock<HashMap<String, ModelConfig>>,
    warmup_prompts: HashMap<String, Vec<String>>,
}

impl ModelManager {
    /// 模型预热：加载后执行预热推理，避免首次请求延迟高
    pub async fn warmup_model(&self, model_id: &str) -> Result<()> {
        tracing::info!(model = model_id, "Warming up model...");
        let prompts = self.warmup_prompts.get(model_id)
            .unwrap_or(&default_warmup_prompts());

        for prompt in prompts {
            let _ = self.run_inference(model_id, prompt).await;
        }
        tracing::info!(model = model_id, "Model warmup complete");
        Ok(())
    }

    /// 模型滚动更新：先加载新版本，验证后切换，再卸载旧版本
    pub async fn rolling_update(
        &self,
        model_id: &str,
        new_loader_config: LoaderConfig,
    ) -> Result<()> {
        let staging_id = format!("{}_staging", model_id);

        // 1. 加载新模型到临时 ID
        self.load_model(&staging_id, new_loader_config).await?;

        // 2. 预热
        self.warmup_model(&staging_id).await?;

        // 3. 验证
        self.validate_model(&staging_id).await?;

        // 4. 原子切换别名
        self.mistralrs.register_model_alias(model_id, &staging_id);

        // 5. 卸载旧模型
        self.mistralrs.unload_model(model_id).await?;

        Ok(())
    }
}
```

配置：
```toml
[[models]]
id = "llama-70b"
source = "/models/llama-70b-chat-hf"
quantization = "Q4_K"
auto_load = true
warmup_prompts = ["Hello", "Summarize this:"]

[[models]]
id = "qwen-7b"
source = "Qwen/Qwen3-7B"
quantization = "Q8_0"
auto_load = false               # 按需加载
max_idle_minutes = 30           # 空闲 30 分钟后自动卸载
```

### 3.9 审计日志

**当前状态**：**无审计日志**。

```rust
// mistralrs-enterprise/src/audit.rs

#[derive(Serialize)]
pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub request_id: String,
    pub client_id: String,       // 从 mTLS 证书或 token 中提取
    pub model: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub latency_ms: u64,
    pub status: &'static str,    // "success" | "error" | "timeout" | "rejected"
    pub metadata: HashMap<String, String>,
}

pub enum AuditEventType {
    InferenceRequest,
    ModelLoad,
    ModelUnload,
    ConfigChange,
    AuthFailure,
}

pub trait AuditSink: Send + Sync {
    fn emit(&self, event: AuditEvent);
}

// 实现：文件、Kafka、数据库等
pub struct FileAuditSink { writer: BufWriter<File> }
pub struct StdoutAuditSink;   // 输出到 stdout，由日志收集器采集
```

### 3.10 配置热更新

**当前状态**：启动时读取配置，运行期间**无法变更**。

```rust
// 基于文件系统 watch 的配置热更新
use notify::{Watcher, RecursiveMode, Event};

pub struct ConfigWatcher {
    config_path: PathBuf,
    tx: watch::Sender<EnterpriseConfig>,
}

impl ConfigWatcher {
    pub async fn start(config_path: PathBuf) -> watch::Receiver<EnterpriseConfig> {
        let (tx, rx) = watch::channel(load_config(&config_path));

        tokio::spawn(async move {
            let (file_tx, mut file_rx) = tokio::sync::mpsc::channel(1);
            let mut watcher = notify::recommended_watcher(move |res: Result<Event, _>| {
                if let Ok(event) = res {
                    if event.kind.is_modify() {
                        let _ = file_tx.blocking_send(());
                    }
                }
            }).unwrap();
            watcher.watch(&config_path, RecursiveMode::NonRecursive).unwrap();

            while file_rx.recv().await.is_some() {
                match load_config(&config_path) {
                    new_config => {
                        tracing::info!("Configuration reloaded");
                        let _ = tx.send(new_config);
                    }
                }
            }
        });

        rx
    }
}
```

**可热更新项**：限流参数、模型列表、内存水位、日志级别

**不可热更新项**（需要重启）：gRPC 端口、TLS 证书、GPU 设备映射

---

## 四、性能优化建议

### 4.1 推理性能优化

#### 4.1.1 PagedAttention 必须开启

当前 PagedAttention 是可选的。在企业部署中应**默认开启**：

```rust
// 将 PagedAttention 设为默认调度策略
// 修改 mistralrs-core/src/scheduler/mod.rs
impl Default for SchedulerConfig {
    fn default() -> Self {
        // 原来：DefaultScheduler
        // 改为：PagedAttention（如果硬件支持）
        #[cfg(any(feature = "cuda", feature = "metal"))]
        { SchedulerConfig::PagedAttentionMeta { ... } }

        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        { SchedulerConfig::DefaultScheduler { ... } }
    }
}
```

#### 4.1.2 启用 Prefix Caching

对于企业场景（大量相同 system prompt 的请求），Prefix Caching 效果显著：

```toml
[inference]
prefix_caching = true          # 默认开启
prefix_cache_max_blocks = 1024  # 最大缓存块数
```

效果预估：对于相同 system prompt 的请求，prefill 阶段可节省 50-80% 的计算。

#### 4.1.3 量化策略选择

企业场景推荐量化配置：

| 场景 | 模型大小 | 推荐量化 | 精度损失 | 速度提升 |
|------|---------|---------|---------|---------|
| 高精度要求 | 7B-13B | FP16 / BF16 | 无 | 基准 |
| 均衡 | 7B-70B | Q8_0 / AFQ8 | <1% | 1.5-2x |
| 高吞吐 | 70B+ | Q4_K / AFQ4 | 2-3% | 3-4x |
| 极限压缩 | 70B+ | Q3_K / AFQ3 | 5%+ | 4-5x |
| GPU 部署优选 | 任意 | GPTQ-4bit | 1-2% | 2-3x |

#### 4.1.4 Flash Attention

CUDA 部署必须启用 Flash Attention：

```bash
cargo build --release --features "cuda flash-attn cudnn"
```

### 4.2 吞吐与延迟优化

#### 4.2.1 Continuous Batching 调优

```toml
[scheduler]
max_num_seqs = 256              # 最大并发序列数（根据 GPU 显存调整）
block_size = 16                 # 块大小（16 是多数模型最优）
max_tokens_in_batch = 4096      # 单批次最大 token 数
```

#### 4.2.2 首 Token 延迟优化 (TTFT)

对延迟敏感的场景（如聊天），需要关注 TTFT（Time To First Token）：

1. **减少 prefill 计算量**：启用 Prefix Caching
2. **chunked prefill**：将长 prompt 分块处理，不阻塞短请求
3. **优先级调度**：短请求优先，减少排队延迟

```rust
// 建议在 Scheduler 中增加优先级队列
pub struct PriorityScheduler {
    high_priority: VecDeque<Sequence>,    // 延迟敏感请求
    normal_priority: VecDeque<Sequence>,  // 普通请求
    low_priority: VecDeque<Sequence>,     // 批量处理请求
}
```

#### 4.2.3 Token 生成吞吐优化 (TPS)

1. **增大 batch size**：在 GPU 显存允许的范围内尽量增大并发序列数
2. **FP8 KV Cache**：减少 KV Cache 内存占用，从而容纳更多并发序列
3. **量化推理**：使用 Q4/Q8 量化减少每次 forward 的计算量

### 4.3 内存优化

#### 4.3.1 KV Cache 压缩

```toml
[paged_attention]
cache_type = "f8e4m3"           # FP8 KV Cache，内存减半
block_size = 16
```

FP8 KV Cache 可将 KV 缓存内存减少约 50%，从而在同等 GPU 显存下服务更多并发请求。需要 Ampere+ (SM 8.0+) GPU。

#### 4.3.2 多 GPU 设备映射

```toml
# 70B 模型分布在 2 张 A100 80G 上
[device_map]
layers = [
    { ordinal = 0, layers = 40 },   # GPU 0: 前 40 层
    { ordinal = 1, layers = 40 },   # GPU 1: 后 40 层
]
```

#### 4.3.3 CPU Offload

对于 GPU 显存不够的情况，可以将部分层放到 CPU：

```toml
[device_map]
layers = [
    { ordinal = 0, layers = 60 },   # GPU 0: 60 层
]
host_layers = 20                     # CPU: 20 层（较慢但节省显存）
```

### 4.4 多实例部署优化

#### 4.4.1 负载均衡策略

企业网关层面的多实例部署建议：

```
                  企业推理网关
                 /     |     \
         ┌──────┐ ┌──────┐ ┌──────┐
         │ 实例1 │ │ 实例2 │ │ 实例3 │
         │ GPU0 │ │ GPU1 │ │ GPU2 │
         └──────┘ └──────┘ └──────┘
```

**推荐负载均衡算法**：基于队列深度的加权调度（不是简单的 Round-Robin）：

```
选择目标实例 = argmin(instance.queue_depth + instance.running_seqs * weight)
```

通过 gRPC 健康检查接口暴露队列深度，由网关做决策。

#### 4.4.2 NCCL 张量并行

对于超大模型（70B+），使用多 GPU 张量并行：

```bash
cargo build --release --features "cuda flash-attn nccl"
```

```toml
[distributed]
tensor_parallel_size = 4        # 4 GPU 张量并行
```

---

## 五、分阶段实施计划

### 阶段一：最小可行产品（MVP）— 4-6 周

**目标**：能跑起来的企业级文本推理服务

| 任务 | 工作量 | 优先级 |
|------|--------|--------|
| 功能裁剪（删除非文本模态） | 2 周 | P0 |
| gRPC 服务层（tonic + proto） | 2 周 | P0 |
| 基础健康检查（liveness + readiness） | 2 天 | P0 |
| mTLS 认证 | 3 天 | P0 |
| 有界请求队列 + 背压 | 2 天 | P0 |
| 基础 Prometheus metrics | 3 天 | P1 |
| `cargo check` + 集成测试 | 3 天 | P0 |

**交付物**：
- 裁剪后的推理引擎
- gRPC 接口（ChatCompletion、Embedding、HealthCheck）
- mTLS 加密通信
- 基础指标暴露
- Docker 镜像

### 阶段二：生产就绪 — 3-4 周

**目标**：可以上线的稳定系统

| 任务 | 工作量 | 优先级 |
|------|--------|--------|
| OOM 防护与内存水位管理 | 1 周 | P0 |
| 优雅启停（drain + shutdown） | 3 天 | P0 |
| 完整 Prometheus metrics（TTFT/TPS/Queue） | 1 周 | P0 |
| 审计日志 | 3 天 | P1 |
| 限流（全局 + 每模型） | 3 天 | P1 |
| 模型预热 | 2 天 | P1 |
| 结构化 JSON 日志 | 2 天 | P1 |
| 压力测试 + 性能基准 | 1 周 | P0 |

**交付物**：
- 生产级 OOM 防护
- 完整可观测性
- 压测报告
- Kubernetes 部署 manifests（Deployment + Service + HPA）

### 阶段三：企业增强 — 4-6 周

**目标**：差异化的企业能力

| 任务 | 工作量 | 优先级 |
|------|--------|--------|
| 多模型动态管理（热加载/卸载） | 2 周 | P1 |
| 模型滚动更新 | 1 周 | P1 |
| 配置热更新 | 1 周 | P2 |
| 请求优先级调度 | 1 周 | P2 |
| 分布式追踪（OpenTelemetry） | 1 周 | P2 |
| 多 GPU 张量并行部署支持 | 1 周 | P2 |
| 生产运维手册 | 3 天 | P1 |

**交付物**：
- 多模型管理能力
- 运维手册
- Grafana 监控面板模板

### 阶段四：高级优化 — 持续迭代

| 任务 | 工作量 | 优先级 |
|------|--------|--------|
| NPU 后端适配（参考 DEV_GUIDE.md 第 6 章） | 8-12 周 | P2 |
| 自定义量化内核优化 | 持续 | P3 |
| Prefix Caching 策略优化 | 2 周 | P2 |
| 投机解码（如需低延迟） | 2 周 | P3 |

---

## 六、目标架构蓝图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 企业推理网关 (已有系统)                                       │   │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │   │
│  │ │ 路由分发  │ │ 用户鉴权  │ │ 限流熔断  │ │ 负载均衡(队列感知)│ │   │
│  │ └──────────┘ └──────────┘ └──────────┘ └──────────────────┘ │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                         │ gRPC (mTLS)                               │
│  ┌──────────────────────▼───────────────────────────────────────┐   │
│  │ mistral.rs 推理引擎 (StatefulSet, 多副本)                     │   │
│  │                                                               │   │
│  │  ┌─── gRPC Server (tonic) ───────────────────────────────┐   │   │
│  │  │  InferenceService    ManagementService    HealthService│   │   │
│  │  └─────────────┬─────────────────────────────────────────┘   │   │
│  │                │                                              │   │
│  │  ┌─── Enterprise Layer ──────────────────────────────────┐   │   │
│  │  │ RateLimiter │ MemoryGuard │ AuditLog │ Metrics │ ...  │   │   │
│  │  └─────────────┬─────────────────────────────────────────┘   │   │
│  │                │                                              │   │
│  │  ┌─── mistralrs-core (裁剪后) ───────────────────────────┐   │   │
│  │  │ Engine ─→ Scheduler ─→ Pipeline ─→ Model Forward      │   │   │
│  │  │   │          │            │           │                │   │   │
│  │  │   │    PagedAttention   KV Cache   QuantMethod         │   │   │
│  │  │   │                                   │                │   │   │
│  │  │   │                            ┌──────┴────────┐       │   │   │
│  │  │   │                          GGUF  GPTQ  HQQ  FP8     │   │   │
│  │  └───┼────────────────────────────────────────────────────┘   │   │
│  │      │                                                        │   │
│  │  ┌───▼─── Hardware ──────────────────────────────────────┐    │   │
│  │  │  candle-core (CPU / CUDA / Metal / NPU)               │    │   │
│  │  └───────────────────────────────────────────────────────┘    │   │
│  │                                                               │   │
│  │  Sidecar: Prometheus exporter ──→ /metrics                    │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌───────────────────────────────────────────────────┐              │
│  │ 可观测性                                           │              │
│  │ Prometheus ←── metrics │ Jaeger ←── traces         │              │
│  │ Grafana ←── dashboards │ ELK ←── JSON logs         │              │
│  └───────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 七、风险与对策

### 7.1 上游更新同步风险

**风险**：裁剪后难以合并上游 mistral.rs 的新模型和修复。

**对策**：
- 保持 fork，不修改 `mistralrs-core/src/models/` 目录结构
- 裁剪通过 Feature Gate 而非删除代码实现（优先推荐）
- 定期从上游 cherry-pick 新模型支持
- 企业新增代码放在独立 crate（`mistralrs-grpc/`、`mistralrs-enterprise/`），不修改核心代码

```toml
# 推荐的 Feature Gate 方式（而非物理删除）
[features]
default = ["text"]
text = []
vision = ["dep:mistralrs-vision"]
diffusion = []
speech = ["dep:mistralrs-audio"]
enterprise = ["dep:mistralrs-grpc", "dep:mistralrs-enterprise"]
```

### 7.2 Candle 框架绑定风险

**风险**：Candle 框架版本锁定（当前 0.9.2），更新可能破坏 API。

**对策**：
- 在 `Cargo.toml` 中固定 candle 版本
- 建立 candle API 兼容层（thin wrapper）
- 如需 NPU 支持，fork candle 并维护企业分支

### 7.3 GPU 显存不足

**风险**：大模型 + 高并发可能导致 OOM。

**对策**：
- 实施 3.6 节的 MemoryGuard
- 启用 FP8 KV Cache 减少 50% 缓存内存
- 使用 Q4/Q8 量化减少模型内存
- 多 GPU 分片或 CPU offload

### 7.4 长连接稳定性

**风险**：gRPC 长连接在网络抖动时断开。

**对策**：
- 启用 gRPC keepalive
- 客户端实现指数退避重连
- 使用 gRPC 内置的重试策略（retry policy）

```toml
[grpc]
keepalive_interval_secs = 30
keepalive_timeout_secs = 10
max_connection_age_secs = 300   # 定期重建连接
```

### 7.5 安全合规

**风险**：企业部署需要满足安全合规要求。

**对策**：
- mTLS 加密所有通信
- 审计日志记录所有推理请求（不记录敏感内容）
- 模型文件加密存储
- 容器以非 root 用户运行
- 网络策略限制出入站流量

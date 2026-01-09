# AGENT.md — FedGuard（单机多进程联邦学习隐私保护系统）

> 本项目为“联邦学习隐私保护系统”课程设计实现说明，覆盖实现要求：联邦学习框架、差分隐私、安全聚合、异构数据处理、通信优化、恶意客户端检测、模型压缩、性能监控（网页形式）。

---

## 1. 单机多进程演示目标

### 1.1 目标
在**一台电脑**上通过**多进程**模拟“1个服务器 + N个客户端”的联邦学习训练流程，支持：
- 多客户端协同训练（FedAvg / FedProx）
- 差分隐私（DP-SGD）与隐私预算（ε/δ）统计
- 安全聚合（pairwise mask 加性掩码）
- 非IID数据切分（Dirichlet）
- 通信优化（每轮采样K客户端、deadline/超时策略）
- 模型压缩（Top-K 稀疏 + 量化 + 误差反馈）
- 恶意客户端检测与鲁棒聚合（trimmed mean/median）
- **网页监控 Dashboard**（浏览器访问，实时曲线+告警+通信/隐私指标）

### 1.2 单机约束（必须满足）
- 不使用多机分布式、无需 Docker
- 通过 **Python 多进程**（subprocess/multiprocessing）启动：
  - `server`：1 个进程
  - `client`：N 个进程（每个进程代表一个客户端）
  - `dashboard`：不单独起进程，**由 server 托管网页**（推荐）

---

## 2. 进程拓扑与端口约定（验收展示一眼明白）

### 2.1 拓扑
- **Server 进程**
  - 负责：轮次调度、模型下发、聚合、检测、指标持久化、网页Dashboard服务
- **Client 进程 x N**
  - 负责：本地数据加载（非IID分片）、本地训练、DP、压缩、安全掩码、上传更新

### 2.2 通信方式（单机本地回环）
- Client ↔ Server：HTTP（FastAPI）或 gRPC（可选）
- 本项目推荐：**FastAPI + JSON/bytes**（实现快、验收稳定）
- 默认端口建议：
  - `SERVER_API_PORT=8000`
  - `DASHBOARD_URL=http://127.0.0.1:8000/dashboard`
  - WebSocket：`ws://127.0.0.1:8000/ws/metrics`

---

## 3. Agent（功能模块）划分与职责边界

> 以“Agent = 可验收的功能模块”划分，确保每个要求项都有明确对应模块。:contentReference[oaicite:1]{index=1}

### 3.1 Server 侧 Agents

#### A. CoordinatorAgent（联邦调度 / 轮次控制）
- **职责**
  - 控制 Round 循环：采样客户端 → 下发模型/配置 → 收集更新 → 触发聚合 → 写入指标
  - 支持 `clients_per_round=K` 抽样（通信优化的一部分）
  - 支持 `deadline_ms`：超时客户端不纳入本轮聚合
- **输入**：全局模型 θ_t、实验配置、在线客户端列表
- **输出**：θ_{t+1}、round_summary（耗时、参与列表、失败列表）
- **验收点**：单机启动 N 客户端后可稳定跑 ≥10 轮

#### B. ClientManagerAgent（客户端注册与状态）
- **职责**
  - 客户端注册、心跳、在线状态维护
  - 黑名单管理（与恶意检测联动）
- **输入**：join/heartbeat 请求
- **输出**：client_id、在线状态、参与资格
- **验收点**：Dashboard/接口可看到在线客户端与本轮参与客户端

#### C. AggregationAgent（聚合 + 鲁棒聚合）
- **职责**
  - FedAvg（按样本数加权）
  - 鲁棒聚合：`trimmed_mean` / `median`
- **输入**：客户端更新（解压后、解掩码后）、样本数 n_i
- **输出**：聚合更新 ΣΔθ 或 θ_{t+1}
- **验收点**：恶意客户端注入时，鲁棒聚合比普通FedAvg更稳定

#### D. SecureAggregationAgent（安全聚合：Server端）
- **职责**
  - 接收 `masked_update_i` 并求和，得到聚合更新（掩码抵消）
  - 不应暴露任何单客户端更新内容
- **输入**：masked updates
- **输出**：ΣΔθ（聚合结果）
- **验收点**：Server 端日志/API 只能得到聚合和，无法还原单体更新

#### E. MaliciousDetectionAgent（恶意/异常检测）
- **职责**
  - 检测异常客户端并触发策略：告警、剔除、降权、黑名单
- **检测模式（与安全聚合兼容）**
  - **模式1（安全聚合开启时推荐）**：基于客户端上报统计特征检测  
    - 例：更新范数（客户端自报）、本地loss/acc、训练耗时、历史一致性
  - **模式2（安全聚合关闭时增强）**：基于更新向量检测  
    - 例：cosine相似度离群、范数离群、层级统计离群
- **输入**：client_metrics /（可选）client_update
- **输出**：anomaly_score、alerts、excluded_clients、blacklist_updates
- **验收点**：sign-flip / scaling 攻击能触发告警与剔除

#### F. MetricsAgent（指标归档与对外API）
- **职责**
  - 持久化每轮指标：收敛、隐私、通信、鲁棒/告警
  - 提供 REST 查询与 WebSocket 实时推送
- **输入**：round_metrics payload
- **输出**：sqlite/jsonl 数据 + `/api/v1/metrics/*` + `/ws/metrics`
- **验收点**：网页可实时显示曲线与告警列表

#### G. WebDashboardAgent（网页监控系统：Server托管）
- **职责**
  - 提供浏览器页面 `/dashboard`
  - 图表实时刷新（优先WebSocket，备选轮询）
- **输入**：MetricsAgent 输出的指标流
- **输出**：网页可视化（收敛/隐私/通信/安全）
- **验收点**：打开浏览器即能看到训练实时状态与关键指标

---

### 3.2 Client 侧 Agents（每个Client进程拥有一套）

#### H. DataAgent（非IID数据切分与加载）
- **职责**
  - 通过 Dirichlet(α) 切分数据，模拟非IID
  - 输出 label 分布统计给Dashboard展示
- **输入**：dataset、client_id、α、seed
- **输出**：本地数据 D_i + label_histogram
- **验收点**：不同客户端 label 分布明显不同（α越小越明显）

#### I. LocalTrainerAgent（本地训练：FedAvg/FedProx）
- **职责**
  - 在本地数据上训练若干 epochs/steps
  - 可切换 FedProx（缓解非IID漂移）
- **输入**：θ_t、D_i、训练超参、(可选) μ
- **输出**：Δθ_i、local_loss/acc、n_i
- **验收点**：非IID条件下 FedProx 收敛更稳定（对比曲线）

#### J. DifferentialPrivacyAgent（差分隐私：DP-SGD）
- **职责**
  - 梯度裁剪 + 高斯噪声
  - 计算隐私预算（ε/δ）并随轮次上报
- **输入**：grad/Δθ、clip_norm C、noise_multiplier σ、δ
- **输出**：Δθ_i^DP、epsilon_i
- **验收点**：DP开启时网页显示 ε 曲线；关闭DP时不消耗预算

#### K. CompressionAgent（压缩：Top-K + 量化 + 误差反馈）
- **职责**
  - Top-K 稀疏化（发送重要参数）
  - 8-bit/16-bit 量化（带scale）
  - 误差反馈（EF）：保存残差下一轮补偿
- **输入**：Δθ、topk_ratio、quant_bits、residual_state
- **输出**：compressed_update + new_residual_state + payload_size
- **验收点**：网页通信量明显下降，精度接近未压缩版本

#### L. SecureMaskingAgent（安全聚合：Client端掩码）
- **职责**
  - 生成 pairwise masks，对更新加掩码
- **输入**：compressed_update、本轮参与集合 S、(简化) seeds
- **输出**：masked_update
- **验收点**：客户端可本地打印单体Δθ；Server无法获取单体Δθ

#### M. CommAgent（与Server通信）
- **职责**
  - join/heartbeat
  - 拉取本轮配置与模型
  - 上传更新与本地统计（含隐私、通信、训练时间）
- **输入**：Server API 地址、client_id、payload
- **输出**：HTTP 请求/响应
- **验收点**：多进程并发下通信稳定，不阻塞轮次

---

## 4. 单机多进程“Round”端到端流程

1. **Server(CoordinatorAgent)** 选择参与客户端集合 S（K个）并生成 round_config
2. Server 下发：`θ_t + round_config`（DP参数、压缩参数、是否安全聚合、deadline）
3. **Client进程**：
   - DataAgent 准备本地数据
   - LocalTrainerAgent 训练得到 Δθ_i
   - DifferentialPrivacyAgent 处理得到 Δθ_i^DP 与 epsilon_i
   - CompressionAgent 压缩得到 compressed_update_i（记录payload字节）
   - SecureMaskingAgent 掩码得到 masked_update_i（若启用安全聚合）
   - CommAgent 上传 masked_update_i + local_metrics
4. **Server**：
   - SecureAggregationAgent 求和得到聚合更新
   - MaliciousDetectionAgent 检测异常（安全聚合开则基于统计，关则可用向量）
   - AggregationAgent（含鲁棒聚合）更新 θ_{t+1}
   - MetricsAgent 写入指标并推送 WebSocket
5. **浏览器 Dashboard** 实时刷新：收敛曲线、ε、通信量、告警

---

## 5. 性能监控系统（网页形式）✅

### 5.1 页面入口与交互
- 访问：`http://127.0.0.1:8000/dashboard`
- 实时数据：优先 WebSocket `/ws/metrics`，备选轮询 `/api/v1/metrics/latest`

### 5.2 页面布局（验收展示建议）
- **顶部指标卡**
  - 当前 round、K/N、round耗时、DP开关、ε_max、上传字节、压缩率、剔除客户端数
- **收敛性 Tab**
  - Global Loss 折线（随round）
  - Global Accuracy 折线（随round）
- **隐私 Tab**
  - ε_max / ε_avg 折线
  - DP参数（C、σ、δ）展示
- **通信 Tab**
  - 每轮 upload_bytes_total / download_bytes_total
  - 压缩率（raw_bytes / compressed_bytes）
- **安全 Tab**
  - 告警表格（time、client_id、reason、action）
  - anomaly_score TopN
  - excluded_clients 列表

### 5.3 指标口径（每轮一条）
建议 `round_metrics` 字段：
- `round_id`
- `global_loss`, `global_accuracy`
- `participants`, `dropped_clients`
- `privacy`: `{epsilon_max, epsilon_avg, delta, noise_multiplier, clip_norm}`
- `comm`: `{upload_bytes_total, download_bytes_total, compressed_ratio}`
- `robust`: `{agg_method, excluded_clients, anomaly_scores}`
- `timing`: `{round_wall_time_ms, train_time_ms_avg, comm_time_ms_avg}`

---

## 6. 目录结构（与Agent一一对应）

```text
FedGuard/
  server/
    app.py                  # FastAPI入口：API + /dashboard + WS
    orchestrator.py         # CoordinatorAgent
    client_manager.py       # ClientManagerAgent
    aggregation/
      fedavg.py             # AggregationAgent
      robust.py             # trimmed mean / median
      secure_agg.py         # SecureAggregationAgent
    security/
      malicious_detect.py   # MaliciousDetectionAgent
    metrics/
      store.py              # MetricsAgent（sqlite/jsonl）
      schema.py
    web/
      dashboard.html        # WebDashboardAgent页面
      dashboard.js
  client/
    main.py                 # Client入口（每个进程一个client）
    data/partition.py       # DataAgent（Dirichlet non-IID）
    train/local_trainer.py  # LocalTrainerAgent + FedProx
    privacy/dp.py           # DifferentialPrivacyAgent
    compression/
      topk.py               # CompressionAgent
      quant.py
      error_feedback.py
    secure/mask.py          # SecureMaskingAgent
    comm/api_client.py      # CommAgent
  experiments/
    configs/default.yaml
    scripts/
      run_local_demo.py     # 单机多进程启动器（spawn server+N clients）
  tests/
    test_secure_agg.py
    test_dp.py
    test_compression.py
    test_robust.py
  README.md

---
## 7. 附加
所有操作只能在此工作区内进行
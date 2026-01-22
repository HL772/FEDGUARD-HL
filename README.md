# FedGuard — 单机多进程联邦学习隐私保护系统

本项目基于 `AGENT.md` 约定实现，目标是在单机环境用多进程模拟 “1 Server + N Client” 的联邦学习训练流程，并具备隐私保护、安全聚合、非 IID 数据、通信优化、模型压缩、恶意检测与可视化监控能力。

## 完成情况

已完成的核心能力：
- 联邦学习主流程（轮次调度、模型下发、FedAvg/FedProx/FedPer/FedPer‑dual 聚合）
- 差分隐私（客户端裁剪 + 高斯噪声 + ε 上报）
- RDP 隐私会计 + 自适应噪声/裁剪（adaptive DP）
- 安全聚合（Pairwise Masking 掩码抵消，Server 只见聚合和）
- 异构数据处理（MNIST + Dirichlet 非 IID 切分 + 标签直方图）
- 通信优化（每轮采样 K 客户端 + deadline 超时策略 + 分批运行）
- 模型压缩（Top‑K + 量化 + 误差反馈）
- 恶意检测与鲁棒聚合（loss/norm 统计 + cosine 相似度，trimmed/median/krum）
- 采样策略（score/随机，固定 pre 选择模式）
- 攻击模拟与检测统计（scale/label‑flip，P/R/F1）
- 性能监控（REST + WebSocket Dashboard）
- 客户端管理（在线列表、手动拉黑/解除、超时展示）
- Web 启动配置（Dashboard 直接配置参数并启动训练）

## 目录结构

```text
FedGuard/
  AGENT.md                  # 项目规范与验收要点
  README.md
  server/
    app.py                  # FastAPI 入口：API + /dashboard + WS
    orchestrator.py         # CoordinatorModule（轮次调度、聚合触发、指标）
    client_manager.py       # ClientManagerModule（注册/心跳/拉黑/在线）
    launcher.py             # Web 启动器（Dashboard 启动客户端进程）
    aggregation/
      fedavg.py             # AggregationModule + FedAvg/FedProx
      robust.py             # trimmed_mean / median
      secure_agg.py         # SecureAggregationModule
    privacy/
      accountant.py         # RDP Accountant（ε 统计）
    security/
      malicious_detect.py   # 恶意检测
    metrics/
      store.py              # MetricsModule（JSONL + WS 推送）
      schema.py             # metrics schema 归一化
      metrics.jsonl         # 训练指标记录（启动会清空）
    web/
      dashboard.html        # Dashboard 页面
      dashboard.js          # Dashboard 逻辑
      dashboard.css         # Dashboard 样式
  client/
    main.py                 # Client 进程入口
    data/partition.py       # DataModule（Dirichlet 非 IID）
    train/local_trainer.py  # LocalTrainerModule（FedAvg/FedProx）
    privacy/dp.py           # DifferentialPrivacyModule
    compression/
      topk.py               # Top‑K + 量化
      quant.py
      error_feedback.py     # 误差反馈（EF）
    secure/mask.py          # SecureMaskingModule
    comm/api_client.py      # CommModule（HTTP 客户端）
  experiments/
    configs/
      default.yaml          # 默认全量配置
      best.yaml             # 当前最佳演示配置
      baseline.yaml         # 基线对照
      innovation.yaml       # 创新对照
      fedper.yaml           # FedPer 方案示例
    scripts/
      run_local_demo.py     # 单机多进程启动器（server + N clients）
      run_ablation.py       # 对照实验脚本
    results/                # 对照实验输出
    runtime/                # 运行期临时文件
  tests/
    test_secure_agg.py
    test_dp.py
    test_compression.py
    test_robust.py
  data/                     # 数据目录（运行时下载）
```

## 快速启动

1) 命令行启动演示（示例：6 客户端、每轮 6 个、每批 2 个、10 轮）
```bash
ONLINE_TTL_SEC=600 MAX_ROUNDS=10 NUM_CLIENTS=6 CLIENTS_PER_ROUND=6 CLIENT_BATCH_SIZE=2 \
python experiments/scripts/run_local_demo.py
```

2) Web 启动（推荐）  
```bash
python server/app.py
```
打开 Dashboard：`http://127.0.0.1:8000/dashboard` → “启动配置” → 启动。

3) 打开 Dashboard（命令行启动场景）  
`http://127.0.0.1:8000/dashboard`

4) 指定配置文件运行（示例：best.yaml）
```bash
CONFIG_PATH=experiments/configs/best.yaml \
ONLINE_TTL_SEC=600 MAX_ROUNDS=10 NUM_CLIENTS=6 CLIENTS_PER_ROUND=6 CLIENT_BATCH_SIZE=2 \
python experiments/scripts/run_local_demo.py
```

## 对照实验模板（baseline vs innovation）

已提供配置模板与脚本：
- Baseline：`experiments/configs/baseline.yaml`（static DP）
- Innovation：`experiments/configs/innovation.yaml`（adaptive_rdp）
- 脚本：`experiments/scripts/run_ablation.py`

示例：
```bash
ONLINE_TTL_SEC=60 MAX_ROUNDS=10 NUM_CLIENTS=6 CLIENTS_PER_ROUND=6 CLIENT_BATCH_SIZE=2 \
python experiments/scripts/run_ablation.py
```

输出：
`experiments/results/baseline_metrics.jsonl`  
`experiments/results/innovation_metrics.jsonl`

## Dashboard 详细介绍与使用说明

### 入口与数据源
- 页面入口：`http://127.0.0.1:8000/dashboard`
- 实时更新：`/ws/metrics`（WebSocket，失败自动切换为轮询 `/api/v1/metrics/latest`）
- 历史加载：`/api/v1/metrics/all`

### 顶部概览卡片
- 轮次、K/N、轮次耗时
- DP 模式、聚合算法、采样模式
- ε 最大值、上传总量、压缩比
- 拉黑数、超时数（全局）

### 客户端指标页
- 损失曲线 / 准确率曲线（客户端训练统计）
- 隐私预算 ε（最大/平均/会计累计）
- 噪声 / 裁剪率（DP 噪声乘子 + 裁剪率）
- 通信负载（上传 KB + 压缩比）
- 客户端更新表（支持轮次切换；含状态、ε、标签分布）

### 客户端管理页
- 在线客户端列表（状态颜色：在线=绿、超时=黄、离线=灰、已拉黑=红）
- 客户端管理表：可手动拉黑/解除
- 告警日志（包含轮次、时间、原因、动作）

### 服务端聚合页
- 服务端评估损失 / 评估准确率
- 聚合更新范数
- 个体准确率公平性（均值/最小/标准差/Jain）
- 检测摘要（P/R/F1、阈值、攻击模拟）
- 相似度排名（cosine z‑score）

### 指标口径补充
- `clip_rate`：参与统计的客户端中，`pre_dp_norm > clip_norm` 的比例。
- `compressed_ratio`：`raw_bytes / compressed_bytes`，越大表示压缩越明显。
- `server.update_norm`：本轮聚合后的更新向量 L2 范数。
- 相似度排名：客户端与参考更新方向的 cosine 分数做中位数/MAD 标准化后得到 z‑score。
- 通信负载（上传总量）：`upload_bytes_total / 1024`。
- ε 曲线：`epsilon_max` / `epsilon_avg` 来自客户端上报；`epsilon_accountant` 来自服务器 RDP 会计。

### 操作提示
- “客户端更新”支持上下轮切换
- “客户端管理”支持一键拉黑/解除（立即影响参与资格）
- “帮助”按钮可查看指标解释

### 常用 API（手动拉黑/解除）
```bash
# 查看在线客户端
curl -s http://127.0.0.1:8000/api/v1/clients

# 拉黑指定客户端
curl -s -X POST http://127.0.0.1:8000/api/v1/clients/blacklist \
  -H "Content-Type: application/json" \
  -d '{"client_id":"<client-id>","reason":"manual"}'

# 解除拉黑
curl -s -X POST http://127.0.0.1:8000/api/v1/clients/unblacklist \
  -H "Content-Type: application/json" \
  -d '{"client_id":"<client-id>"}'
```

## 配置与关键环境变量

`experiments/configs/default.yaml`（示例节选）
```yaml
dp:
  enabled: true
  mode: adaptive_rdp
  accountant: rdp
  clip_norm: 2.0
  noise_multiplier: 0.3
  delta: 1e-5
  target_epsilon: 8.0
  schedule:
    type: cosine_decay
    sigma_end_ratio: 0.5
  adaptive_clip:
    enabled: true
    percentile: 0.9
    ema: 0.2

compression:
  enabled: true
  topk_ratio: 0.5
  quant_bits: 16
  error_feedback: true

security:
  secure_aggregation: true
  malicious_detection: true
  robust_aggregation: trimmed_mean
  trim_ratio: 0.1

train:
  algo: fedavg
  fedprox_mu: 0.01
  deadline_ms: 15000

model:
  split:
    backbone_keys:
      - backbone.*
    head_keys:
      - head.*
    private_head_keys:
      - private_head.*

sampling:
  enabled: true
  strategy: score
  epsilon: 0.1
  score_ema: 0.2
  timeout_penalty: 0.5
  anomaly_penalty: 1.0
  fairness_window: 3
  softmax_temp: 1.0
```

常用环境变量：
- `MAX_ROUNDS`：总轮次
- `NUM_CLIENTS`：客户端总数
- `CLIENTS_PER_ROUND`：每轮参与的客户端数（K）
- `CLIENT_BATCH_SIZE`：分批并发运行的客户端数量
- `ONLINE_TTL_SEC`：在线心跳窗口（秒，默认 600）
- `CONFIG_PATH`：指定配置文件路径

## 说明与注意

- 指标落盘为 `server/metrics/metrics.jsonl`，服务启动时会自动清空。
- WebSocket 连接失败时 Dashboard 会自动切换为轮询模式。
- Client Metrics 曲线是“本地训练统计”；Server Aggregation 曲线是“统一测试集评估”，两者趋势可能不同。
- 使用 FedPer 时，将 `train.algo` 设置为 `fedper`，仅聚合 `backbone_keys`，`head_keys` 保持客户端私有。
- 使用 FedPer 双头版本时，将 `train.algo` 设置为 `fedper_dual`，聚合 `backbone_keys` + `head_keys`，`private_head_keys` 保持客户端私有。
- 超时模拟支持 `train.timeout_simulation.client_ranks`（0-based，对应 client-1 为 0）。

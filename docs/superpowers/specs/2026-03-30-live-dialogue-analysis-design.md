# Live Dialogue Analysis - Design Spec

## Overview

将 LiveTranslate 从「实时翻译」改造为「直播连线对话实时识别 + AI 分析指导」系统。

**核心场景**：直播连线（带货、商务谈判、情感连线、采访、娱乐连麦、客服售后等），为主播提供实时 AI 分析和建议话术。

## Architecture

### Overall Pipeline

```
┌─────────────┐     ┌─────────────┐
│ 系统音频(对方) │     │ 麦克风(我方)  │
│ WASAPI Loop  │     │ PyAudio     │
└──────┬───────┘     └──────┬──────┘
       ▼                    ▼
   VAD₁ + ASR队列₁      VAD₂ + ASR队列₂     ← 双通道独立管道
   ASR工作线程₁          ASR工作线程₂
   "对方: ..."          "我方: ..."
       │                    │
       └────────┬───────────┘
                ▼
        ┌───────────────┐
        │   对话缓冲区    │  带时间戳 + 说话人标签
        └───────┬───────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌─────────────┐   ┌──────────────┐
│  分析调度器   │   │  摘要压缩器   │
│ debounce合并  │   │  异步独立运行  │
│ latest-wins  │   │  每N句压缩一次 │
│ 手动触发     │   │  替换旧摘要    │
│ 流式输出     │   │              │
└──────┬──────┘   └──────┬───────┘
       ▼                 ▼
┌─────────────┐   ┌──────────────┐
│ AI分析面板    │   │ 摘要存储      │
│ (固定区域)    │   │ (供分析器引用)  │
└─────────────┘   └──────────────┘
```

## Component Design

### 1. Audio Dual Channel

改造 `audio_capture.py`，从混合单路输出改为双通道独立输出。

- **系统音频通道**：WASAPI loopback 捕获对方声音（保持现有逻辑）
- **麦克风通道**：独立 PyAudio 流捕获我方声音（从 mix-in 改为独立通道）
- 两路各自输出 32ms 音频 chunk，独立送入各自的 VAD

**接口变化**：
- 现有 `read()` 返回单路混合音频
- 改为两个独立方法：`read_system()` 返回系统音频，`read_mic()` 返回麦克风音频
- 各自被独立的音频线程调用，互不阻塞

### 2. Dual VAD + ASR (Async)

每路音频独立的 VAD → ASR 管道，ASR 异步化。

**VAD 改动**：
- 两个独立 `VADProcessor` 实例（VAD₁ 处理系统音频，VAD₂ 处理麦克风）
- 新增硬性时间上限（3 秒强制切分），不等停顿，切分点选 VAD confidence 最低处
- 复用现有 backtrack split 逻辑

**ASR 异步化（关键改动）**：
- 现有：管道线程同步调 ASR，阻塞音频读取
- 改为：VAD 产出的段落推入队列，独立 ASR 工作线程消费
- 两路各自有独立的 ASR 队列 + 工作线程
- VAD 永远不被 ASR 阻塞，音频捕获零丢失

```
VAD₁ → ASR队列₁ → ASR工作线程₁ → 标记 "对方" → 对话缓冲区
VAD₂ → ASR队列₂ → ASR工作线程₂ → 标记 "我方" → 对话缓冲区
```

**ASR 实例共享策略**：
- 两个 ASR 工作线程可以共享同一个 ASR 模型实例（加锁互斥）
- 或各自加载独立实例（消耗更多显存，但零等待）
- 默认共享，设置中可选独立

**段落积压处理**：
- ASR 队列积压 > 3 段时：合并相邻段落为一个长段一起识别
- 极端情况：标记最旧段落为「跳过」，UI 显示 `[...语速过快，部分跳过...]`

### 3. Dialogue Buffer

新增组件，作为双通道 ASR 结果的汇合点。

```python
@dataclass
class Utterance:
    speaker: str        # "对方" | "我方"
    text: str
    timestamp: float    # time.time()

class DialogueBuffer:
    utterances: list[Utterance]       # 完整对话历史
    pending_analysis: list[Utterance]  # 待分析队列（未被分析器消费的）
    summary: str                       # 当前滚动摘要（由压缩器维护）
    summary_cursor: int                # 摘要覆盖到的 utterance index
```

- 新 utterance 到达时：追加到 `utterances`，追加到 `pending_analysis`，通知 UI 显示原文，通知分析调度器
- 线程安全：所有操作加锁

### 4. Analysis Scheduler

分析调度器，管理 AI 分析的触发时机和请求生命周期。

**自动触发逻辑**：
```
新句子到达 → 推入 pending_analysis
  │
  有正在进行的 API 请求？
  ├─ 否 → 启动 debounce 计时器（800ms）
  │        计时器到期 或 累积 ≥ 3 句 → 发起分析请求
  └─ 是 → 句子留在队列
           当前请求完成后 → 立即合并队列中所有句子发起新请求
```

**手动触发**：
- 用户点击「分析」按钮
- 取消 debounce 计时器
- 如有正在进行的请求 → 标记为 stale（完成后丢弃结果）
- 立即把 pending_analysis 所有句子 + 摘要 → 发起新请求

**API 调用 messages 结构**：
```
System: {场景预设提示词}

User:
  ## 对话摘要
  {滚动摘要}

  ## 最新对话
  [对方 12:03:01] 这个价格能再便宜一点吗
  [我方 12:03:03] 我考虑一下
  [对方 12:03:05] 最低八折，今天最后一天

  请基于以上对话给出分析和建议。
```

- 流式输出直接覆盖面板内容
- 新请求到来时旧的流式停止，新的接管

### 5. Summary Compressor

独立于分析器的异步组件，负责控制 context 大小。

**触发条件**：`len(utterances) - summary_cursor > 15`（未被摘要覆盖的句子超过 15 句）

**调用流程**：
1. 取 `当前摘要 + utterances[summary_cursor:]`
2. 发送给 AI：固定的摘要压缩提示词（不受场景预设影响）
3. 返回新摘要 → 替换 `summary`，更新 `summary_cursor`
4. 压缩过程中分析器继续用旧摘要，不阻塞

**摘要压缩提示词**（固定）：
```
将以下对话摘要和新增对话合并，生成简洁的结构化摘要。
保留：关键事实、双方立场、已达成共识、待解决问题、情绪变化。
删除：重复信息、无实质内容的寒暄。
输出纯文本，不超过 500 字。
```

### 6. Analysis Presets (Scene Prompts)

场景预设系统，两层结构：结构化模板 + 高级模式。

**内置预设**（`ANALYSIS_PRESETS`）：

| 预设 | 关注重点 | 输出包含 |
|------|---------|---------|
| 带货直播 | 报价、优惠条件、关键承诺 | 价格对比、砍价建议、风险提醒 |
| 商务谈判 | 对方诉求、分歧点、让步信号 | 局势判断、建议话术、底线分析 |
| 情感连线 | 情绪变化、关键诉求、矛盾点 | 情绪分析、共情话术、风险提醒 |
| 采访访谈 | 关键信息、未回答问题 | 信息提取、追问建议、话题延伸 |
| 娱乐连麦 | 话题走向、互动节奏 | 话题建议、互动策略、气氛调节 |
| 客服售后 | 对方问题、情绪状态 | 问题归类、应对策略、升级判断 |

**结构化模板编辑器**（普通用户）：

```python
@dataclass
class AnalysisPreset:
    name: str                      # 场景名称
    role: str                      # AI 角色（如 "带货分析师"）
    focus_tags: list[str]          # 关注重点标签（多选）
    output_tags: list[str]         # 输出包含标签（多选）
    extra_instructions: str        # 额外指令（自由文本）
    is_advanced: bool = False      # 是否使用高级模式
    advanced_prompt: str = ""      # 高级模式：完整提示词
```

可选的 focus_tags：`情绪变化, 关键诉求, 矛盾点, 报价, 承诺, 让步信号, 关键信息, 未回答问题, 互动节奏`

可选的 output_tags：`局势判断, 建议话术, 风险提醒, 价格对比, 情绪分析, 问题归类, 话题建议, 信息提取`

**模板拼装逻辑**：系统根据字段自动生成完整提示词，用户不需要写 prompt。高级用户切换到高级模式可直接编辑。

**存储**：内置预设在代码中，用户自定义预设存入 `user_settings.json` 的 `analysis_presets` 字段。

### 7. UI Changes

#### 7.1 Main Overlay Layout

```
┌──────────────────────────────────────┐
│ 拖动条 │ 暂停 │ 清空 │ ⚙设置 │ 退出    │  Row 1: 保留现有
│ 场景:[带货直播 ▼] │ 模型:[GPT-4 ▼]    │  Row 2: 场景下拉 替代 目标语言下拉
├──────────────────────────────────────┤
│ [对方 12:03:01] 这个价格最低八折       │
│ [我方 12:03:03] 我再考虑下            │  上半区：对话原文滚动
│ [对方 12:03:05] 今天最后一天活动       │  带说话人标签(颜色区分) + 时间戳
│ ...                                   │
├──────────────────────────────────────┤
│ 📊 AI 分析                    [分析▶]  │  分隔栏 + 手动触发按钮
│                                       │
│ ## 关键信息                            │
│ - 对方底价：八折                       │  下半区：AI 分析面板
│ - 限时优惠，制造紧迫感                 │  固定区域，markdown 渲染
│ ## 建议话术                            │  流式刷新覆盖
│ - "其他平台有七五折，能匹配吗？"        │
└──────────────────────────────────────┘
```

#### 7.2 Component Mapping

| 现有组件 | 改造 |
|---------|------|
| `ChatMessage`（原文+译文） | 改为只显示带说话人标签的原文，颜色区分对方/我方 |
| 目标语言下拉 | 替换为场景预设下拉 |
| 翻译计数 / token 显示 | 改为分析计数 + 摘要 token 用量 |
| `subtitle_window` | 可选用作独立 AI 分析面板（OBS 采集） |
| 新增：AI 分析面板 | overlay 下半区，固定高度，markdown 渲染，流式覆盖 |
| 新增：分析按钮 | 手动触发分析 |

#### 7.3 Control Panel Tab Changes

| Tab | 改造 |
|-----|------|
| VAD/ASR | 保留，增加双通道设备选择（系统音频设备 + 麦克风设备分开选） |
| Translation → AI 分析 | 模型选择保留，提示词区域换成场景预设管理 |
| Style | 保留，增加说话人标签颜色设置 |
| Subtitle → 分析面板 | 改为 AI 分析面板的字体、大小、位置设置 |
| 新增：场景预设 | 结构化模板编辑器 + 高级模式切换 |
| Benchmark | 保留（可改为分析模型 benchmark） |
| Cache | 保留 |
| Changelog | 保留 |

## Threading Model

```
主线程:       Qt 事件循环（所有 UI）

音频线程₁:    系统音频捕获 → VAD₁ → ASR队列₁ (持续运行，不阻塞)
音频线程₂:    麦克风捕获 → VAD₂ → ASR队列₂ (持续运行，不阻塞)

ASR线程₁:     消费 ASR队列₁ → 识别 → 标记"对方" → 对话缓冲区
ASR线程₂:     消费 ASR队列₂ → 识别 → 标记"我方" → 对话缓冲区

分析线程:     debounce → 合并待分析句子 → API 流式调用 → 更新面板
压缩线程:     监控缓冲区 → 超阈值时压缩摘要 → 更新摘要存储
```

共 6 个工作线程 + 主线程。跨线程通信统一用 Qt signals。

## File Changes Summary

| 文件 | 操作 | 说明 |
|------|------|------|
| `audio_capture.py` | 改造 | 双通道独立输出 |
| `vad_processor.py` | 小改 | 增加硬性时间上限强制切分 |
| `main.py` | 大改 | 管道从单线程串行改为多线程异步，新增调度逻辑 |
| `translator.py` | 改造为 `analyzer.py` | 翻译逻辑 → 分析逻辑，保留流式输出和模型管理 |
| `subtitle_overlay.py` | 改造 | 布局改为上下分区，新增分析面板，场景下拉 |
| `control_panel.py` | 改造 | Tab 调整，新增场景预设编辑器 |
| 新增 `dialogue_buffer.py` | 新建 | 对话缓冲区 + 分析调度器 |
| 新增 `analysis_presets.py` | 新建 | 场景预设定义 + 结构化模板拼装 |
| 新增 `summary_compressor.py` | 新建 | 摘要压缩器 |
| `config.yaml` | 更新 | 新增分析相关默认配置 |
| `i18n/*.yaml` | 更新 | 新增 UI 文案 |

## Reusable Components (No Change)

- `model_manager.py` — 模型管理完全复用
- `asr_engine.py` / `asr_sensevoice.py` / `asr_funasr_nano.py` / `asr_qwen3.py` — ASR 引擎完全复用
- `dialogs.py` — 设置向导、模型下载对话框复用
- `log_window.py` — 日志窗口复用
- `benchmark.py` — 可复用（改为分析 benchmark）
- `subtitle_window.py` — 可选复用为独立分析面板
- `subtitle_settings.py` — 可复用（调整设置项）

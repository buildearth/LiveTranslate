# 多窗口并行 AI 分析 设计文档

## 目标

支持同时运行多个场景预设的 AI 分析，每个预设在独立的弹出窗口中显示结果，分析历史滚动追加不丢失。

## 架构

- 每个弹出窗口对应一个独立的 `AnalysisScheduler` + 一个 `AnalysisWindow`
- 所有 analyzer 共享同一个 `DialogueBuffer`（数据源相同，各自独立消费）
- Overlay 内现有分析面板保留，右键场景下拉框可弹出独立窗口

### 数据流

```
DialogueBuffer (共享)
  ├── AnalysisScheduler (overlay 内置) → overlay._analysis_text
  ├── AnalysisScheduler (弹出窗口1) → AnalysisWindow 1
  └── AnalysisScheduler (弹出窗口2) → AnalysisWindow 2
```

每个 AnalysisScheduler 通过 `buffer.on_utterance()` 注册监听，独立触发分析。

## 组件

### AnalysisWindow（新文件 `analysis_window.py`）

独立的浮动窗口：
- `Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool`，半透明背景
- 标题栏：预设名称标签 + 关闭按钮，中键拖动
- 内容区：只读 QTextEdit，滚动追加模式
- 每条分析结果以时间戳分隔线追加到底部，自动滚到最新
- Streaming 过程中最后一条实时更新，完成后固定
- 窗口关闭时发出 `closed` 信号，触发资源清理

结果格式：
```
──── 11:23:45 ────
分析内容第一条...

──── 11:24:12 ────
分析内容第二条（streaming 实时更新中）
```

### 交互流程

1. 用户在 overlay 的场景下拉框（`_scene_combo`）上右键
2. 弹出 QMenu，显示"在新窗口中打开"
3. 创建新的 `AnalysisScheduler`（绑定同一个 `DialogueBuffer` + 选中的预设）
4. 创建新的 `AnalysisWindow`，连接 analyzer 的 streaming 回调
5. 设置 analyzer 的 client/model（与当前活跃模型相同）
6. analyzer.start()，窗口显示

### LiveTranslateApp 管理

- `self._analysis_windows: list[tuple[AnalysisScheduler, AnalysisWindow]]`
- 创建窗口：`_open_analysis_window(preset_name)` 方法
- 模型切换时：遍历所有弹出窗口的 analyzer，同步更新 client/model
- stop() 时：统一 stop 所有 analyzer，关闭所有窗口
- 窗口关闭回调：stop 对应 analyzer，从列表中移除

### DialogueBuffer 兼容性

当前 `on_utterance()` 是 append-only 的 listener 列表，多个 analyzer 注册多个 callback 即可，无需修改。

每个 AnalysisScheduler 实例独立维护自己的 `_pending_count`、`_last_analysis_text`、`_debounce_timer` 等状态，互不干扰。

## 不做什么

- 不修改现有 overlay 内置分析面板的行为
- 不做窗口位置持久化（弹出窗口是临时的）
- 不做跨窗口的分析结果合并
- 不限制弹出窗口数量（用户自行控制）

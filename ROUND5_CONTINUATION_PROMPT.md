# TongSIM Agent 后续优化交接与 Round 5 指令词

## 仓库与基线

- 当前成果仓库：https://github.com/jackzhu119/baseline-agent
- 当前主分支：`main`
- Round 4 保留分支：`codex-optimization-round-4`
- 原始上游仓库：https://github.com/hoosh11161/baseline-agent
- Round 4 起点：`2e391e52b9a6d84ea0c639ec678d6e0bd3d4d4d4`
- Round 4 功能完成提交：`9c90974fbcc0cac37d93450a588fcf18ea621aad`
- Round 4 报告提交：`f4f456cd49408a3ff532d2a338beb3cfdc81a740`
- `submit_answer` 生命周期修复提交：`8527700`（以仓库完整提交记录为准）

接手者必须从 `jackzhu119/baseline-agent` 的最新 `main` 继续，不要重新克隆旧上游后从零实现。开始时运行 `git rev-parse HEAD`，以实际最新 SHA 为准。

## Round 4 相对上游新增的能力

以下对比以 `hoosh11161/baseline-agent@2e391e5` 为基线。

### 1. NPC 信息增益规划

- 结构化跟踪 required/known/missing facts、人物与物品位置、所有权、依赖、已问及未解决问题。
- 根据 information gain、任务相关性、重复/已知惩罚和成本选择问题。
- 支持 owner-first、多跳线索、NPC redirect、语义重复拦截和无关问题拒绝。

### 2. 物理动作后置条件

- 新增 `ActionExpectation` 和 `SUCCESS / FAILURE / UNKNOWN`。
- 抓取验证手持状态；放置验证松手以及公开位置或官方完成证据。
- 导航、倒水、切割、清洗、拖地、坐下只在公开字段支持时判定，缺证据保持 `UNKNOWN`。

### 3. 模型恢复与成本控制

- 固定恢复顺序：正常调用 → 短上下文重试 → 保守确定性 fallback。
- timeout、rate limit、malformed JSON 不再立即结束 episode。
- 增加模型失败、token、prompt 字符数、重试、重规划和后置条件统计。
- 根据真实 step limit 划分 EARLY/MID/LATE/CRITICAL。

### 4. Counting 边界表达式

- 支持有界 `OR(AND(...)) AND NOT` 表达式。
- 覆盖“红色或蓝色”“除了红色”“不是蓝色”“分别多少”和成对分组条件。
- 最终仍由 Python Registry 确定性计算；不确定语义保留结构化模型解析入口。

### 5. Raven 可验证规则层

- 从公开题面提取数量、填充、位置、方向、大小和对称性。
- 验证 alternation、progression、union、intersection、difference、XOR 等规则。
- 规律最多 12 个，并记录 coverage、consistency、complexity penalty。
- 只重排已有本地 ResNet18+MLP 候选，不增加模型调用；没有规则证据时保持原排名。

### 6. Jigsaw 全局分配

- 构建 piece × cell 得分并用小规模搜索求全局 assignment。
- 使用公开位置、格心容差与旋转证据。
- 多个同分解时保留候选集合，禁止虚构精确放置。

### 7. Tidyroom 置信度与 FinishGuard

- placement candidate 带 source、confidence、evidence、selection margin 和 `direct_safe`。
- AABB 推算表面不直接视为高置信放置；并列表面不会盲选。
- FinishGuard 要求官方完成证据或所有初始官方物体/拼图块均验证完成。
- “执行过一个动作”或“模型说完成了”不再足以 `finish_task`。

### 8. 测试与证据

- 上游基线：76 tests passed。
- Round 4：109 tests passed、compileall PASS、公开轨迹 replay PASS。
- 后续生命周期修复：111 tests passed；`submit_answer` 提交后立即退出本地 subject 循环并调用 `evaluate_subject`，不再依赖服务端 15 秒 fallback。
- 新增 preflight、官方 episode benchmark、failure analyzer、model matrix 和换机指南。
- 真实 TongSIM 未运行：50051/50060 不可达、无模型凭据、未提供 official release，所以真实分数提升为 `NOT VERIFIED`。

详细证据位于：

- `OPTIMIZATION_ROUND_4_REPORT.md`
- `benchmark/round4_before.json`
- `benchmark/round4_after.json`
- `benchmark/round4_preflight.json`
- `benchmark/round4_replay.json`
- `docs/ROUND4_WINDOWS_HANDOFF_GUIDE.md`
- `15_SECOND_TIMEOUT_ANALYSIS.md`

## 当前最需要下一轮解决的三件事

1. 在真实 TongSIM 和官方任务服务中建立 matched A/B，得到五类任务的 verified episode。
2. 用官方 Raven 图片测量 CNN 与规则层准确率，检查裁剪、旋转和噪声域迁移。
3. 从真实 Tidyroom 失败回合提取目标表面公开字段，完善多候选目标语义匹配和置信度校准。

## 可直接复制给下一位开发者的指令词

```text
你现在接手并继续优化一个已经完成 Round 4 的 TongSIM / 通用智能体任务挑战赛 Agent。

当前成果仓库：
https://github.com/jackzhu119/baseline-agent

原始上游仓库：
https://github.com/hoosh11161/baseline-agent

这是 Round 5，不是从零开发。必须从当前成果仓库最新 main 开始，不得退回旧 baseline，不得重复 Round 4 已完成的功能。

一、建立工作分支

git clone https://github.com/jackzhu119/baseline-agent.git baseline-agent-round5
cd baseline-agent-round5
git remote rename origin user-origin
git remote add upstream https://github.com/hoosh11161/baseline-agent.git
git fetch user-origin
git fetch upstream
git checkout -b codex-optimization-round-5 user-origin/main
git push -u user-origin codex-optimization-round-5

记录 git rev-parse HEAD、git remote -v 和 git log --oneline -15。

二、必须完整阅读

README.md
OPTIMIZATION_ROUND_4_REPORT.md
ROUND5_CONTINUATION_PROMPT.md
docs/ROUND4_WINDOWS_HANDOFF_GUIDE.md
benchmark/round4_before.json
benchmark/round4_after.json
benchmark/round4_preflight.json
benchmark/round4_replay.json
15_SECOND_TIMEOUT_ANALYSIS.md
arenaagent/competition/
arenaagent/vlm_agent/vlm_agent.py
arenaagent/vlm_agent/raven_skill.py
tests/
scripts/

不要重写已经存在的 NPC information-gain planner、ActionExpectation、Counting boolean filters、Raven rule verifier、Jigsaw global assignment、Tidyroom confidence、bounded model recovery、FinishGuard 或 `submit_answer → evaluate_subject` 生命周期修复。只有真实失败证据证明有缺陷时才修改。

三、先复现当前状态

固定使用 Python 3.12：

uv python install 3.12.14
uv sync --python 3.12.14 --extra dev
uv run --python 3.12.14 python scripts/generate_pb2.py
uv run --python 3.12.14 python -m pytest -q
uv run --python 3.12.14 python -m compileall -q arenaagent scripts tests
uv run --python 3.12.14 python scripts/replay_episode.py examples/public_trace_example.json

当前预期是 111 tests passed、compileall PASS、offline replay PASS。若不同，先定位原因，不要继续叠功能。

四、最高优先级是真实评测闭环

自动检测官方 release、TongSIM、赛题系统、127.0.0.1:50051、127.0.0.1:50060、模型 credentials 和官方 verified episodes。

运行：
uv run --python 3.12.14 python scripts/preflight.py --release-dir "实际 release 路径"

如果真实环境可用，冻结当前 main 为 BEFORE，并在相同任务、seed、模型、参数和次数下运行 Counting、Raven、Tidyroom、NPC、Jigsaw。记录 success、score、steps、invalid actions、model/VLM calls、tokens、retry、replan、finish reason、first critical failure 和 failure category。

如果真实环境不可用，明确写 REAL_TONGSIM_NOT_AVAILABLE。不得虚构分数、成功率或排行榜提升；只允许使用 unit tests、mock episodes、public trace 和 synthetic regressions。

五、Round 5 优先顺序

1. 根据官方 verified episode 的 first-critical-failure 频率，优先修最高频根因。
2. Raven：建立官方图片回归集，测量原 CNN、规则层及组合排序的 Top-1/Top-K；只根据测量结果校准融合权重。
3. Tidyroom：分析真实错误 surface、placement height、目标语义和已正确放置物体重复移动；仅使用公开字段。
4. Jigsaw：用真实多 piece/multi-cell 回合校验 assignment、坐标轴、格心容差和旋转证据。
5. NPC 与 Counting：仅修真实回合暴露的解析缺陷，不扩建脆弱的大型自然语言规则库。
6. 模型比较必须使用相同 verified 输入，统计 structured-output validity、latency、tokens 和最终任务结果。

六、安全与证据边界

- 不得读取隐藏 ground truth、硬编码 evaluation seed/答案或利用比赛漏洞。
- 不得修改官方 protobuf/gRPC 协议。
- 不得提交 API key、token、.env、日志缓存或私人凭据。
- UNKNOWN 不得当作 SUCCESS，模型文本不得作为唯一完成证据。
- 不得删除旧测试，不得 force push main。

七、开发节奏

每个稳定模块必须增加针对真实缺陷的 regression test，运行相关测试与完整 pytest，小提交并立即 push 到 codex-optimization-round-5，再用同一批 matched cases 比较修改前后。

八、最终交付

创建 OPTIMIZATION_ROUND_5_REPORT.md，包含 Starting/Ending SHA、真实环境、实现与文件、tests before/after、replay、官方 matched A/B、VERIFIED/NOT VERIFIED、失败实验、first-failure frequency、Top 3 risks 和下一轮建议。

最后确认无 secret/cache/logs，push Round 5 分支；只有可以 fast-forward 时才合并用户 main 并重跑关键测试。失败时停止，绝不 force push。

最终回复列出仓库、分支、起止 SHA、测试数、replay、真实 TongSIM、真实模型 benchmark、分支推送、main 合并、工作树和未同步文件状态。
```

## 接手者停止条件

如果没有真实服务、模型凭据或官方回合，不能把更多离线规则包装成“排行榜提升”。此时应交付可复现回归、明确的 `NOT VERIFIED`，并列出需要在比赛电脑执行的最小命令和数据需求。

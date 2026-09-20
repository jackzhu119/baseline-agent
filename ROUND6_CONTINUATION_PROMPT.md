# 下一位开发者继续优化指令（Round 6）

请直接复制下面内容给下一位开发者：

```text
你现在接手一个已经多轮优化的 TongSIM / 通用智能体任务挑战赛 Agent。

成果仓库：https://github.com/jackzhu119/baseline-agent
继续工作的分支：fix/tidyroom-deadline-hardening
原始上游：https://github.com/hoosh11161/baseline-agent

不要从原始 baseline 重新设计。先执行：
1. git fetch --all --prune
2. git checkout fix/tidyroom-deadline-hardening
3. git pull --ff-only
4. git rev-parse HEAD
5. 阅读 TIDYROOM_DEADLINE_HARDENING.md、TIDYROOM_REAL_RUN_OPTIMIZATION.md、
   15_SECOND_TIMEOUT_ANALYSIS.md 和最新 git log。
6. 运行 uv sync --extra dev && uv run pytest -q。

当前自动回归为 143 passed，但真实 TongSIM 成绩尚未在本机验证。运行时必须先确认日志中的
competition-runtime-v4+<sha> 与当前提交一致，防止拿旧包测试。

本轮已经解决：
- 未经视觉审查的 Unknown 禁止抓取；
- look_at_object 后的单物体聚焦分类，避免批量 ID—像素错配；
- Unknown 收纳目标的 target_surface 分类和持久化；
- 每个 Unknown 最多两次聚焦审查，模糊对象有界降级；
- put 后主动 look_at_location，避免 PLACED 无限等待；
- 中文/英文/结构化不可抓取错误和跨帧持久化；
- 每局记录真实源码 SHA。

下一步只依据真实失败日志优化，禁止凭空改 prompt。优先顺序：
1. 在官方 release 复现 TidyRoom，保存 episode JSON、failure_report、控制台日志和录像。
2. 对每个失败建立可重复回归：感知 ID、聚焦审查输出、目标分配、抓取手状态、放置证据、结束生命周期。
3. 若图像中存在可公开读取的 2D bbox/segmentation 字段，再实现带框截图；没有字段时不得伪造 bbox。
4. 比较至少 5 个固定 seed/预设的成功率、步数、VLM 调用数、错误率和官方分数。
5. 不修改服务端、protobuf、官方 timeout，不读取隐藏 evaluator 状态，不把计划或启发式分数当成真实成绩。
6. 修改后必须通过全量 pytest、compileall、Ruff E/F、git diff --check，并把报告和可复制指令一同推送。

验收目标：真实 TidyRoom 不再出现 Unknown 直接抓取、同一对象无限 look、无目标盲放、put 后永久等待、
已确认不可抓取对象复活或 finish_task 提前结束。若任何目标未通过，请明确标记 NOT VERIFIED，不得声称完美。
```

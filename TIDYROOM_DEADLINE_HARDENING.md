# TidyRoom 截止日前最终加固报告

## 当前代码

- 成果仓库：<https://github.com/jackzhu119/baseline-agent>
- 最终加固分支：`fix/tidyroom-deadline-hardening`
- 本轮起点：`ab8217e`（已包含前两轮 TidyRoom 修复）
- 原始上游：<https://github.com/hoosh11161/baseline-agent>

运行时会输出 `Agent build: competition-runtime-v4+<git-sha>`。提交日志和每局指标也会保存同一构建标识，
因此真机录像、日志和代码版本可以一一对应。部署环境若没有 `.git`，请在启动前设置
`AGENT_BUILD_SHA` 为实际提交 SHA。

## 为什么前两次真实运行仍出现大量问题

1. 真实场景中的物体语义经常是 `Unknown`，而旧离线用例主要使用 `shoe`、`cup`、`shoe rack` 等干净标签。
2. 第一轮规则会过滤 Unknown，造成漏捡；第二轮虽加入批量 VLM，却把多个数字 ID 和一张图片一起交给模型，
   没有可靠的 ID—像素对应关系，模型可能把附近物体的外观套到错误 ID 上。
3. 旧审查只识别杂物，Unknown 的鞋架、垃圾桶、沙发、桌面或食品收纳区仍无法成为目标。
4. `put_down_sth` 后若感知没有立即返回被放物体的新坐标，状态会长期停在 `PLACED`，随后掉回自由规划或循环。
5. 不可抓取错误只匹配少量英文短语，中文消息或结构化错误码会被当作普通失败，污染重试状态。
6. 运行日志只有固定字符串 `competition-runtime-v3`，无法证明真机当时究竟运行了哪次提交。

## 本轮修复

### 1. 单物体聚焦视觉审查

- Unknown 物体不能再直接进入抓取 API。
- 先执行确定性的 `look_at_object`，下一帧只审查刚刚居中的一个物体。
- VLM 只负责返回严格分类 JSON，不负责规划动作。
- 聚焦步骤必须与当前帧步骤一致，避免使用已过期的相机绑定。
- 每个物体最多审查两次；仍然模糊时标记为 `DEFERRED_AMBIGUOUS`，保留审计证据但不无限循环。

### 2. 同时识别杂物和收纳目标

视觉分类角色扩展为：

- `clutter`：鞋、杯、瓶、食物、垃圾、枕头；
- `target_surface`：鞋类收纳、垃圾桶、软装表面、杯具表面、食品收纳；
- `other`：其他物体。

审查结果持久化为 `vlm_semantic_type` 或 `vlm_target_type`。后续原始帧仍返回 Unknown 时，不会重新开始审查。
放置仍须通过公开 `place_location` 或 AABB、语义兼容度和置信阈值，不能任意编造坐标。

### 3. 主动放置复核

- 放置 API 成功且手已空后，优先检查新鲜物体坐标是否落在目标容差内。
- 若 TongSIM 不再返回该物体坐标，状态机主动对目标位置执行两次 `look_at_location`。
- 两次目标复核均无矛盾证据后，以“放置成功 + 手空 + 两次目标复核”作为降级验证依据，并写入
  `placement_evidence`；不会再无期限停在 `PLACED`。
- 若新鲜坐标明确显示物体远离目标，仍判定失败，不会用降级证据覆盖矛盾事实。

### 4. 错误与版本可追溯

- 不可抓取识别支持结构化错误码、常见英文变体以及“不可拾取/无法拿取”等中文消息。
- 已确认为不可抓取的对象加入持久集合，后续帧即使仍有 VLM 标签也不会重新加入候选。
- 每局日志记录真实 Git SHA；可用 `AGENT_BUILD_SHA` 在打包环境注入提交版本。

## 自动验证

```text
uv run pytest -q                         143 passed
uv run python -m compileall -q ...      PASS
Ruff E/F（本轮修改文件）                 PASS
git diff --check                        PASS
```

测试新增覆盖：未审查 Unknown 禁止抓取、聚焦后才调用视觉模型、Unknown 目标持久化、两次模糊审查有界退出、
放置目标主动复核、结构化/中文不可抓取错误、不可抓取状态跨帧保持、构建 SHA 可追溯。

## 今日真实环境最短验收流程

```powershell
git clone https://github.com/jackzhu119/baseline-agent.git
cd baseline-agent
git checkout fix/tidyroom-deadline-hardening
git pull --ff-only
git rev-parse HEAD
uv sync --extra dev
uv run python -m pytest -q
```

启动比赛前确认日志首部出现：

```text
Agent build: competition-runtime-v4+<与 git rev-parse HEAD 前 12 位相同>
```

真机至少观察以下链路：

```text
Unknown -> look_at_object -> focused review -> clutter/target_surface
clutter + semantic target -> move_and_take_object -> hand=true
put_down_sth -> hand=false -> fresh target observation or 2 x look_at_location
placement_evidence -> VERIFIED -> finish_task -> evaluate_subject
```

最终使用官方 `package-results` 命令生成提交文件，并确认五个题型结果文件齐全。

## 证据边界

本轮机器没有可连接的官方 TongSIM 服务和正式赛题实例，因此不能诚实声称真实得分或“近乎完美”已经验证。
当前可以确认的是代码级状态机、故障边界和 143 项自动回归全部通过。最终提交前必须在官方环境跑至少一局
TidyRoom；若失败，请保留完整 `logs/metrics/episode_*.json`、`failure_report.json`、控制台日志和构建 SHA。

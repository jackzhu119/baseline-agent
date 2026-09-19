# 高校组：通用智能体任务挑战赛

通用智能体任务挑战赛面向通用智能体在具身仿真环境中的感知、决策与执行能力评测，参赛者需要构建能够完成指定任务的Agent。
本仓库提供比赛 baseline、运行脚本、模型配置和上手文档，帮助参赛者快速搭建环境、调试方案并提交结果。

## 比赛系统下载
- 完整比赛系统下载链接
  - 夸克网盘：https://pan.quark.cn/s/b81737fe757a?pwd=8bcQ
  - 百度网盘：https://pan.baidu.com/s/1uvbGpc3UQIibR_RvJ9grBQ?pwd=bj6f

根据你的系统类型选择对应的比赛系统进行下载，不支持MacOS

## 比赛系统说明
1. 系统支持Windows和Linux平台，位于Linux目录和Windows目录，如图所示：
![](docs/screenshot-20260601-115527.png)
请根据你的电脑类型下载对应比赛系统，不支持MacOS。

1. 比赛系统包含基准Agent代码、仿真客户端、赛题系统、文档，如图所示
![](docs/screenshot-20260601-120436.png)
- 基准Agent：就是本仓库代码，提供比赛baseline，帮助选手快速上手比赛，获得比赛结果。
- 客户端：比赛需要的具身仿真环境
- 赛题系统：负责向智能体出题，并评判最终结果
- 文档：比赛系统说明文档

## 如何开始？
开始前你需要将比赛系统下载到本地，并仔细阅读下载资料文档中的《【挑战赛】初赛系统使用指南.pdf》，根据指南运行比赛系统和baseline agent。

## 进一步阅读

下载资料文档中的《ArenaAgentPro上手指南》包含完整上手指南，包含 Agent 工作流程、感知层、动作层、Prompt 调试和提分思路，参赛同学请重点阅读。

## 声明
我们不强制要求必须使用大模型，你可以使用任意算法实现任务。

## 竞赛优化分支

`competition-optimization` 分支在不修改官方协议、不读取隐藏状态的前提下加入了：

- 动作 schema、当前可见 object ID、坐标、手持状态和终止动作校验；
- 基于稳定 object ID 的世界状态、计数去重、观察差异、NPC 事实与步数预算；
- 重复动作/停滞检测和失败分类；
- 每局 JSON 指标、`failure_report.json` 和 benchmark 汇总；
- 官方判分轨迹的严格训练集导出（未验证回合不会混入）；
- 模型超时、重试耗尽和感知异常的可恢复处理，不再伪造 `finish_task`；
- Raven 本地模型直连，避免额外 VLM 决策；
- Counting 有界扫描、episode 对象注册表和确定性计数；
- Tidyroom 抓取/移动/放置验证状态机、NPC 结构化记忆、Jigsaw 公共坐标网格推断；
- 一次有界动作修复、显式 RECOVERY 状态和任务级 FinishGuard；
- 公开轨迹重放、官方回合限定的失败频率分析和模型矩阵脚本；
- 安全的模型 JSON 解析以及可安装环境下的 protobuf 导入修复。

Windows 上可以执行：

```powershell
uv sync --extra dev
uv run python scripts/generate_pb2.py
uv run python -m pytest -q
uv run python scripts/preflight.py --release-dir "C:\path\to\official\release"
.\scripts\run_preliminary.ps1 -Model VLMGPT5Config -RunTimes 5
uv run python scripts/benchmark.py
uv run python scripts/failure_analyzer.py
uv run python scripts/replay_episode.py examples/public_trace_example.json
uv run python scripts/export_verified_trajectories.py
```

最后一条命令只接受由官方任务服务判分并标记为 verified 的回合，生成
`training/verified_actions.jsonl` 和带来源 SHA-256 的 manifest。真实得分只会从官方任务服务返回的评测结果生成。
完整架构、57 项回归、真实环境边界和后续 A/B 流程见
[COMPETITION_HIGH_SCORE_REPORT.md](COMPETITION_HIGH_SCORE_REPORT.md)。

Round 4 的换机安装、环境验收、真实评测与结果打包步骤见
[docs/ROUND4_WINDOWS_HANDOFF_GUIDE.md](docs/ROUND4_WINDOWS_HANDOFF_GUIDE.md)，本轮证据边界和优化明细见
[OPTIMIZATION_ROUND_4_REPORT.md](OPTIMIZATION_ROUND_4_REPORT.md)。

下一位开发者继续 Round 5 时，请直接使用
[ROUND5_CONTINUATION_PROMPT.md](ROUND5_CONTINUATION_PROMPT.md)，其中列出了相对上游的增量、当前未验证项和可复制的后续优化指令。

每题结束后额外等待 15 秒的客户端生命周期分析及修复证据见
[15_SECOND_TIMEOUT_ANALYSIS.md](15_SECOND_TIMEOUT_ANALYSIS.md)。

# TongSIM 每题额外 15 秒等待分析

分析基线：`1129ab28f08ac00649e15677e0f195ce2700558f`

结论状态：`AGENT_BUG_WITH_SERVER_FALLBACK`

## 1. Root cause

Counting 的最终动作是 `submit_answer`。修复前，`VLMAgent._handle_submit_answer()` 只把答案包装为任务服务器 `action_space["key"]` 对应的 payload，没有把 `AgentBase.subject_finished` 设为 `True`。因此：

1. `update_action` 已经让任务本体进入 finished；
2. Agent 本地仍停留在 `_run_subject()` 的 answering 循环；
3. 客户端尚未走到循环后的 `evaluate_subject` RPC；
4. 服务端 `task_flow` 观察到“task finished + agent still ANSWERING”，进入 15 秒兜底等待；
5. 超时后服务端日志显示 `agent_start_evaluation`，随后 Agent 才离开 ANSWERING。

根因不是 Counting 扫描、模型调用、retry、等待 observation 或 FinishGuard。它发生在最终答案已经生成之后，是终止动作没有结束客户端 subject 循环。

## 2. Agent state flow

修复前的 Counting 路径：

```text
get_subject
→ run_step
→ acquire_first_person_perception
→ CountingSolver bounded scan / registry deduplication
→ deterministic count
→ submit_answer
→ _handle_submit_answer returns {action_space_key: answer}
→ update_action
→ subject_finished remains False
→ subject loop does not immediately reach evaluate_subject
→ server fallback timeout
```

修复后的路径：

```text
submit_answer selected
→ subject_finished = True
→ update_action sends the unchanged final answer
→ agent loop exits
→ evaluate_subject RPC
→ server can leave ANSWERING and evaluate immediately
```

`AgentBase._run_subject()` 现在记录 `time.perf_counter()` 生命周期事件：

```text
subject_loop_started
action_ready
action_applied
agent_loop_exit
evaluate_subject_started
evaluate_subject_returned
```

这些日志可以在真实 release 中直接计算 `action_applied → evaluate_subject_started` 和 `evaluate_subject_started → returned` 的延迟。

## 3. Server state flow

仓库中的公开 protobuf 只暴露：

- `update_action`
- `is_current_subject_finished`
- `evaluate_subject`
- `evaluate_task`
- `get_task_status`

没有 `FinishSubject`、`FinishAnswering`、`Done` 或携带 agent completion status 的独立 RPC。`SessionStatus.FINISHED` 是整个 session/task 状态，不是截图中的内部 agent `ANSWERING` 枚举。

从真实运行日志可直接确认服务端分支条件不是无条件等待，而是：

```text
Task is finished but at least one agent is still ANSWERING;
waiting 15.0s before forcing evaluation.
```

超时后紧接：

```text
agent_start_evaluation called
All agents finished answering. Evaluating subjects.
```

因此 15 秒是多人 Agent task-flow 的兜底等待窗口；只有 task 已结束、但至少一个 agent 仍为 ANSWERING 时触发。当前工作区没有官方 release/server 源码，`E:\比赛\release` 也不在本机，无法引用 `arena.server.task_flow` 的实现行号；这里没有伪造未取得的服务端源码证据。服务端状态结论由协议调用面、客户端调用顺序和用户提供的真实服务端日志共同支持。

## 4. `finish_task` semantics

`finish_task` 不是 TongSIM protobuf RPC，也不是 task server 的独立 finish RPC。

它是 Agent 动作词表中的本地 pseudo-action：

1. `_do_action()` 把它路由到 `AgentBase._handle_finish()`；
2. `_handle_finish()` 设置本地 `subject_finished = True`；
3. 它把完成说明包装到 `action_space["key"]`；
4. `AgentBase` 再通过通用 `update_action` RPC 提交 payload；
5. 循环退出后，真正显式请求判分的是 `evaluate_subject` RPC。

所以它既不是单纯的 TongSIM 动作，也不会单独直接写服务端 agent 状态。它的关键作用是结束本地循环，使正式的 `evaluate_subject` 调用能够发生。

## 5. Counting final action sequence

`TaskStrategyRouter` 对 Counting 先依次发出有限角度的 `turn_in_degree`。覆盖完成后调用 `CountingSolver.solve()`；只有 `result.confident` 且 `answer is not None` 时才构造：

```python
{
    "action": "submit_answer",
    "parameters": {},
    "output": result.answer,
}
```

最后一条 Agent 语义动作因此是 `submit_answer`，线上 payload 是 `{action_space_key: str(answer)}`。补丁不改变扫描次数、registry 去重、过滤表达式、置信条件或最终数字，只修复该终止动作之后的生命周期。

## 6. 15 秒触发条件

触发条件是：

```text
task semantic result is finished
AND any connected agent state == ANSWERING
```

`wait_after_first_agent_secs=15.0` 是服务端在多人 Agent 场景中等待其余 Agent 正常进入 evaluation 的 grace period；本次单 Agent 仍触发，说明唯一 Agent 没有及时执行正式 evaluation 路径。它不是每题固定、不可避免的等待政策。

用户提供的真实 Counting 日志中，多题从 warning 到 `agent_start_evaluation` 约为 15–17 秒，和 15 秒窗口加轮询/调度开销一致。

## 7. Agent bug or official design

判定：`AGENT_BUG_WITH_SERVER_FALLBACK`。

- 15 秒数值和强制 evaluation 是官方服务端的安全兜底机制；
- 进入该兜底分支的原因是 Agent 的 `submit_answer` 未结束本地 subject answering 生命周期；
- 这不是 FinishGuard 引入的回归；最初提交 `59d291a` 的 `AgentBase` 和 `_handle_submit_answer` 已有同样的生命周期缺口；
- 当前 deterministic Counting 只是稳定、必然地走到了这个旧路径，所以问题在真实 Counting 中每题稳定复现；
- 官方 baseline 也包含这个缺口，但这不表示依靠 timeout 是有意设计。协议已经提供 `evaluate_subject` 正常路径。

## 8. Modified files

- `arenaagent/vlm_agent/vlm_agent.py`
  - `submit_answer` 被选中时设置 `subject_finished = True`；
  - 记录 `submit_answer_selected` 高精度时间。
- `arenaagent/agent_base.py`
  - 添加 subject/action/update/evaluation 生命周期时间日志；
  - 循环结构等价展开，以区分服务端完成和本地终止动作完成。
- `tests/test_agent_lifecycle.py`
  - 验证最终答案 payload 不变；
  - 验证 `submit_answer → update_action → evaluate_subject`，且中间不再次轮询或调用 `run_step`。
- `15_SECOND_TIMEOUT_ANALYSIS.md`
  - 本报告。

没有修改 protobuf、TongSIM 接口、官方 server timeout、FinishGuard 或任务求解器。

## 9. Why this modification

这是协议边界内的最小修复：让问答题的明确最终动作拥有与行为题 `finish_task` 相同的本地终止语义，然后继续使用既有官方 `evaluate_subject` RPC。没有把 timeout 改成 0，也没有绕过服务端状态机。

Raven 的 `solve_raven` 路径没有在本轮强制改成首答即终止，因为其代码保留 ranked candidate retry；在没有真实 Raven 服务端证据前改变它可能损害正确性。

## 10. Tests

环境：CPython 3.12.14。

```text
uv run --python 3.12.14 python -m pytest -q
111 passed in 2.89s

uv run --python 3.12.14 python -m compileall -q arenaagent scripts tests
PASS

git diff --check
PASS（仅有 Git 的 LF/CRLF 工作区提示）

uv run --python 3.12.14 ruff check --select F,E9 arenaagent/agent_base.py arenaagent/vlm_agent/vlm_agent.py tests/test_agent_lifecycle.py
PASS
```

首次 pytest 尝试因新环境未安装 dev extra 而未执行；运行 `uv sync --extra dev` 后按要求的原命令重跑并通过。

## 11. Real TongSIM before/after latency

BEFORE（真实比赛证据）：用户截图中的 Counting subject 多次出现约 15–17 秒的 `warning → agent_start_evaluation` 延迟。

AFTER（当前机器）：`NOT VERIFIED IN REAL TONGSIM`。

只读预检结果：

- task service `127.0.0.1:50051`：不可达；
- TongSIM proxy `127.0.0.1:50060`：不可达；
- official release：本机未提供；
- model credential：未配置。

因此本报告不声称已经取得真实 AFTER latency。离线生命周期测试证明代码会在同一调用链中直接从 `update_action` 进入 `evaluate_subject`，不经过 sleep、下一 observation、LLM、retry 或再次 subject poll。真实 release 复测时应以新增日志验证：

```text
action_applied
→ agent_loop_exit
→ evaluate_subject_started
→ server agent_start_evaluation
```

若 `action_applied` 自身仍阻塞约 15 秒，则说明官方 `update_action` RPC 服务实现还存在服务端同步阻塞，需要取得 release 源码/trace 继续定位；当前仓库无法证明或修改该服务端行为。

## 12. Correctness impact

- Counting 仍完成全部有界扫描后才提交；
- 只有 solver 已判定 `confident` 的最终答案会走 deterministic `submit_answer`；
- 最终答案字段和值完全不变；
- Tidyroom/Jigsaw 的 FinishGuard 和验证条件不变；
- NPC 仍只在 required facts 齐全后由策略提交；
- Raven retry 路径不变；
- 没有提前执行 evaluation 的非终止动作。

预期不影响最终任务正确性，只消除最终答案已经提交之后的无意义 ANSWERING 停留。

## Final conclusion

结论：这个 15 秒来自官方服务端在“task 已 finished、但 Agent 仍为 ANSWERING”时启用的 fallback 等待窗口；触发它的根因是 baseline Agent 的 `submit_answer` 漏掉本地 subject 完成信号，导致 `evaluate_subject` 未及时调用。

状态：已修复客户端生命周期缺口；真实 TongSIM AFTER 延迟尚待有官方 release 的环境验证。

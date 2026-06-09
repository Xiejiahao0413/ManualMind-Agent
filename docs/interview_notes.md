# 面试讲解笔记

这份笔记用于复习 ManualMind-Agent 的项目讲解，不是面向用户的产品文档。

## 项目背景

ManualMind-Agent 是一个面向复杂设备手册的多 Agent 故障诊断系统。目标场景包括工业设备手册、故障码表、维护说明、安全规则和历史故障案例。

项目当前用空压机 A100 的 demo 手册展示端到端流程：文档入库、混合检索、工具调用、诊断报告生成、SSE 流式输出和 Trace 查询。

## 为什么不是普通 RAG

普通 RAG 往往是：

```text
用户问题 -> 向量检索 -> 拼 prompt -> LLM 回答
```

这个项目更强调工程控制：

- 文档先经过解析、脱敏、结构化切分和元数据构建。
- 检索不是单一路径，而是 BM25 + dense + rerank。
- Agent 不能直接调用工具，必须经过 Router、Guard、Retry/Fallback 和 Validator。
- 系统记录 Tool Memory，避免重复工具调用和死循环。
- 高风险维修场景可以触发人工接管。
- Trace 记录每个节点和工具调用，方便排错和评测。

可以总结为：这个项目不是“问答 demo”，而是一个带安全边界、工具控制和可观测性的诊断工作流。

## 为什么使用多 Agent

设备诊断任务不是单一生成任务，至少包含：

- 意图判断
- 故障码和症状抽取
- 手册证据检索
- 安全规则检查
- 报告生成
- 熔断和人工接管

用 LangGraph 的 Supervisor-Worker 模式，可以把这些步骤拆成明确节点：

- Supervisor Node 负责路由。
- Diagnosis Node 负责结构化诊断状态。
- Retrieval Node 负责通过工具层检索证据。
- Safety Report Node 负责安全检查和报告生成。
- Circuit Breaker Node 负责中止异常循环。
- Handoff Node 负责生成人工接管 payload。

这样做的好处是状态流转清晰、节点职责明确、每个阶段都可以测试和 Trace。

## 为什么引入 MCP

MCP 的价值是把工具调用变成统一协议和统一边界。

在当前项目中，先实现了本地 MCP-style Tool Server：

- Tool Registry
- Tool Executor
- Tool Schemas
- Structured ToolExecutionResult

Agent 不直接调用检索器或函数，而是通过：

```text
Tool Intent -> Tool Router -> Tool Call Guard -> MCP Tool Executor -> Tool Result Validator -> Memory Manager
```

这为后续替换成真实 MCP transport 做准备，同时保留了当前测试的可运行性。

## 为什么使用混合检索

设备手册里有大量精确信号：

- 故障码，例如 E03、F12、P001
- 型号，例如 A100、MX100
- 参数名，例如温度、电压、压力
- 章节标题，例如安全注意事项、维护周期

BM25 适合这些精确匹配。

但用户问题经常是口语化描述，例如“机器过热停机”“压力上不去”。这类问题更适合 dense retrieval。

所以项目使用：

```text
BM25 sparse retrieval
-> Dense retrieval interface
-> candidate merge and chunk_id dedup
-> rerank interface
```

当前 dense 和 rerank 是 mock/in-memory 实现，方便本地测试。后续可以替换为 Milvus、BGE embedding 和 BGE-rerank。

## 如何避免 Agent 工具调用死循环

主要靠 Tool Control Layer 和 Tool Memory。

Tool Call Guard 检查：

- 工具白名单
- 必填参数
- 敏感参数
- retry_count
- max_tool_calls_per_task
- normalized args_signature
- duplicate successful signature

如果同一个 tool_name + normalized args 已经成功调用过，系统复用缓存结果，不重复执行工具。

如果 retry_count 达到 3，或者工具调用预算超限，触发 Circuit Breaker，并进入 Handoff。

## 如何做敏感信息脱敏

Sensitive Data Guard 当前使用 regex + dictionary 检测：

- 手机号
- 邮箱
- IP 地址
- 设备编号
- 工单号
- 内部 URL
- 位置关键词

覆盖位置：

- 用户输入
- 文档入库
- MCP 工具参数
- SSE 流式输出

StreamingOutputGuard 使用小 buffer，避免手机号、邮箱等敏感文本被拆成多个 chunk 后漏检。

## 如何做重试、降级和人工接管

RetryFallbackManager 把错误类型映射到降级动作：

- Milvus 超时 -> BM25-only
- BM25 失败 -> Dense-only
- Rerank 失败 -> fused score
- LLM 失败 -> template report
- 缺少设备型号 -> clarification
- 高风险且证据不足 -> human handoff
- 三次失败 -> circuit breaker handoff

当前项目没有接真实外部服务，但控制逻辑已经可测试。

人工接管 payload 包括：

- task_id
- handoff_reason
- risk_level
- device_name
- fault_code
- symptoms
- tool_trace
- retrieved_sources

## 如何做 Trace 和 Evaluation

Trace 和 Memory 是分开的。

Memory 用于运行时状态，例如 task、session、tool memory。

Trace 用于观测和排错，例如：

- supervisor_started / completed
- diagnosis_completed
- tool_call_started / completed / failed
- retrieval_trace
- fallback_decision
- circuit_breaker_triggered
- handoff_created
- final_answer_generated

Evaluation 使用当前 workflow 输出和 trace events 做基础指标：

- 工具选择准确率
- 故障码抽取准确率
- 接管判断准确率
- 来源覆盖率
- 安全提醒覆盖率

这让项目不只是能跑，还能被持续评测。

## 面试时可以强调的亮点

- 不是单纯堆 LLM，而是把 Agent 工具调用做成可控状态机。
- FastAPI 保持 stateless，状态进入 Memory Manager。
- 检索层可替换，当前 mock/in-memory，后续可接 Milvus 和 BGE。
- 工具调用有白名单、参数校验、敏感信息检查、重复签名检查和熔断。
- SSE 输出经过流式脱敏。
- Trace 和 Evaluation 给系统可观测性和回归测试能力。
- demo 可以本地一条命令跑通，适合展示工程闭环。

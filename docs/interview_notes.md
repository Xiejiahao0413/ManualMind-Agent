# 面试讲解笔记

这份文档是给自己面试复盘用的，语气尽量接近现场口述，不是论文式项目说明。

## 1. 项目一句话介绍

ManualMind-Agent 是一个面向复杂设备手册问答和故障诊断的多 Agent 系统。它用合成设备手册做本地 demo，把文档入库、混合检索、受控工具调用、安全脱敏、Trace、Evaluation 和人工接管串成一条完整链路。

## 2. 最复杂的 Agent 工作流怎么设计

我把诊断拆成 Supervisor、Diagnosis、Retrieval、Safety Report、Circuit Breaker 和 Handoff 几个节点。Supervisor 负责看状态决定下一步，Diagnosis 做规则抽取，Retrieval 只能通过工具控制层查证据，Safety Report 负责安全判断和报告生成，异常循环会进熔断和人工接管。

## 3. 为什么它不是简单 RAG

简单 RAG 通常是“检索几个 chunk，然后拼 prompt 让模型回答”。这个项目更强调工程控制：工具不能被 Agent 直接调用，检索结果要校验，重复工具调用会复用，敏感信息会脱敏，高风险问题会触发 handoff，还能用 Trace 和 Evaluation 检查每一步是否合理。

## 4. 如何保证工具调用可靠性

工具调用必须走 Tool Router、Tool Call Guard、Retry/Fallback Manager、MCP Tool Executor 和 Validator。Guard 会检查白名单、参数 schema、敏感参数、retry_count、max tool calls 和重复签名。成功调用会记录到 Tool Memory，下次相同参数可以复用，避免死循环。

## 5. 为什么引入 MCP-style 工具层

我没有让 Agent 直接依赖检索器或业务函数，而是做了一个 MCP-style Tool Registry 和 Executor。这样工具的输入输出 schema、错误处理、耗时、状态都能统一管理。当前是本地实现，后续如果换真实 MCP transport，边界也比较清楚。

## 6. 混合检索怎么做

设备手册里有很多精确信号，比如 E03、F12、型号、参数名，这些更适合 BM25；用户口语化描述，比如“机器跑一会儿就停”，更适合 dense retrieval。项目里做了 BM25、dense interface、metadata filter、chunk_id 去重和 rerank interface，默认本地跑，后续可接 Milvus 和 BGE。

## 7. 敏感信息保护怎么做

Sensitive Data Guard 覆盖用户输入、文档入库、工具参数、检索结果和 SSE 输出。当前用 regex 和词典先识别手机号、邮箱、IP、设备编号、工单号和内部 URL。检测到后只记录类型、mask 和事件，不记录原始敏感值，避免日志里二次泄露。

## 8. Streaming Output Guard 为什么需要 rolling buffer

SSE 是分块输出的，手机号、邮箱、IP 可能被拆到多个 chunk 里。如果每个 chunk 单独检测，就会漏掉。v0.2 加了 rolling buffer：先保留尾部片段，和下个 chunk 拼起来检测，只释放确认安全的 prefix，flush 时再处理剩余内容。

## 9. Milvus 和 memory fallback 怎么设计

v0.5 做的是可选 Milvus 后端，不改变默认检索链路。没有配置 Milvus、没装依赖或连接失败时，系统继续用 LocalBM25Retriever、InMemoryDenseRetriever 和 MockReranker。这样本地 demo 和 pytest 不依赖外部服务，也方便后续做真实向量库实验。

## 10. LLM 报告生成为什么要 fallback

工业诊断里报告生成不能因为模型超时或 API key 缺失就中断，所以 v0.4 在原模板报告外包了一层 ReportGenerationService。默认走模板；显式开启 OpenAI 才调用真实 LLM。LLM 失败、空输出或输出不合格时，会自动回到模板报告。

## 11. Trace 和 Evaluation 怎么做

Trace 记录节点开始结束、工具调用、检索结果、fallback、熔断和 handoff。Evaluation 用合成样本检查工具选择、故障码识别、证据覆盖、安全提醒和人工接管决策。现在有标准、对抗边界、多设备三组样本，总计 130 条，用来验证链路而不是宣称真实生产准确率。

## 12. 当前边界和后续生产化方向

当前项目是本地 synthetic demo，不包含真实厂家手册和生产数据。后续生产化可以补真实权限体系、持久化 memory/trace、真实 MCP 服务、Milvus 集群、生产 embedding/rerank、扫描 PDF/OCR、模型治理和安全审计。这个项目主要展示的是架构和控制链路。

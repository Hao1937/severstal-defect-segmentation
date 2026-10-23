# BigData 项目索引

更新：2026-10-23。项目：Severstal 钢板表面四类缺陷分割，服务课程展示、报告和答辩。

## 入口

- 人类阅读入口：[README](README.md)。
- 协作维护规范：[AGENTS.md](AGENTS.md)，规则统一维护于此。
- 作业原始要求：[作业要求](作业要求.md)。
- 当前阶段与方案：[当前方案](docs/当前方案.md)。

## 文档结构

```mermaid
flowchart TD
    R[README：项目入口] --> C[docs/当前方案：当前设计]
    R --> S[Severstal调研：来源与推导]
    C --> D[docs/决策记录：选择与变化]
    D --> E[docs/实验记录：计划与实测]
    S --> D
    E --> C
```

## 模块索引

| 模块 | 用途 | 本地索引 |
| --- | --- | --- |
| `docs/` | 当前方案、决策与实验 | [docs/CLAUDE.md](docs/CLAUDE.md) |
| `Severstal调研/` | 赛题事实、公开方案观察与设计依据 | [调研/CLAUDE.md](Severstal调研/CLAUDE.md) |

代码入口为 `severstal/`，成员配置在 `configs/`，相关测试在 `tests/`；环境与命令见[运行说明](docs/运行说明.md)，实际运行状态见[实验记录](docs/实验记录.md)。

`.spec-workflow/` 为通用模板，不属于项目模块。后续新增模块时更新本索引，具体配置与实验结果仍写入对应文档。

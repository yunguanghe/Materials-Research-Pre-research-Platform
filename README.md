# HYG Materials Agent

这是一个面向材料研究场景的智能工作台项目，包含：

- 本地合金材料 JSON 检索
- CIF 文件下载与解析
- 初始结构与弛豫后结构对比
- 基于 MatterSim 的结构弛豫与候选快排
- 网页聊天前端

## 主要文件

- `chat.py`
  前端与后端总控入口，负责对话、路由、结果展示
- `matesim_dft.py`
  候选结构弛豫、数值指标提取、候选快排
- `parse_cif.py`
  单个 CIF 解析与两个 CIF 的横向对比
- `download_mp_cif.py`
  从 Materials Project 按体系或化学式下载 CIF
- `pdf_table_pipeline.py`
  表格图片转结构化 JSON
- `bigmodle/extract_alloy_tables_with_gemini.py`
  使用大模型抽取合金表格内容

## 项目能力

### 1. 本地材料检索

系统可从本地 JSON 材料库中检索：

- 合金牌号
- 名义化学成分
- 主要成分
- 杂质要求

### 2. 结构文件处理

系统支持：

- 下载 CIF 到本地
- 解析单个 CIF
- 对比 `input.cif` 与 `relaxed.cif`

### 3. 数值计算

当前数值计算重点是：

- 结构弛豫
- 每原子能量提取
- 最大受力提取
- 收敛判断
- 候选优先级排序

当前排序本质上是“稳定性代理快排”，适用于：

- 前期候选预筛
- 继续研究优先级推荐

## 环境变量

公开版本不包含任何明文 API key，请通过环境变量配置：

```bash
export YUNWU_API_KEY="your_yunwu_api_key"
export YUNWU_BASE_URL="https://yunwu.ai/v1"
export YUNWU_MODEL="gpt-5.4-mini"
export MATERIALS_PROJECT_API_KEY="your_materials_project_api_key"
```

## 运行

### 启动聊天前端

```bash
python3 chat.py
```

### 下载 CIF

```bash
python3 download_mp_cif.py --query Nb-Ti --top-k 5
python3 download_mp_cif.py --query FeSe --top-k 3
```

### 解析 CIF

```bash
python3 parse_cif.py /path/to/file.cif --text
```

## 目录说明

- `PNG/`
  示例图片与前端 logo
- `ilovepdf_structured_json/`
  本地结构化材料数据
- `bigmodle/`
  大模型表格抽取脚本

## 项目边界

当前项目适合：

- 材料研究前期检索与预筛
- 候选结构快排
- 智能材料助手

当前项目不直接等同于：

- 完整 DFT 工作流
- 声子与电子声子耦合全流程
- 真实 Tc 最终预测

## 文档

- `PROJECT_OVERVIEW.md`

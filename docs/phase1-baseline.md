# Phase 1 派生基线

## Identity

- Source repository: `paper-agent`
- Source commit: `7c69adf` (`release: pagent 0.7.0 web agent and catalog grounding`)
- Derived repository: `paper-research-agent`
- Distribution: `paper-research-agent`
- Import package: `pragent`
- CLI: `pra`
- Version: `0.1.0`
- Default data directory: `~/.pragent`
- Environment prefix: `PRA_`

原 `paper-agent/` 与学习项目 `pagent-java/` 未被修改；PRAgent 使用独立数据目录，不会自动读取或写入旧 Pagent 数据。

## Reproducible environment

- macOS arm64
- CPython `3.11.15`（由 `uv` 管理）
- 依赖约束：`requirements-dev.lock`

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -c requirements-dev.lock -e '.[dev]'
```

## Automated validation

```text
pytest: 235 passed
compileall: passed
pip/uv dependency check: passed
check_tmp_space.py: passed
wheel: paper_research_agent-0.1.0-py3-none-any.whl
isolated wheel assets/import/entry-point/Web smoke: passed
```

已验证 wheel 只暴露 `pra` 命令，并包含 `pragent/web/index.html`、`app.js`、`style.css`。

## Runtime smoke

使用临时 `PRA_DATA_DIR`、无 API key：

- `pra --version` → `pra 0.1.0`
- `pra --help` → 成功
- `pra status` → 空库、独立临时数据目录、离线 LLM 状态
- `pra chat` → 以明确错误拒绝启动（未配置 `PRA_LLM_API_KEY`）
- `pra serve` → loopback Uvicorn 成功启动
- `GET /` → 包含 `PRAgent` 品牌
- `GET /api/status` → `papers=0, chunks=0, llm_configured=false`

## Scope boundary

本基线没有调用真实 DeepSeek、arXiv 或本地 embedding 模型，也没有执行产品 Phase 2+ 功能；这些不能从本记录推断为已验证。

# 项目约定

- 本项目专属的约定、架构说明、构建/测试要求写在这里。
- 用户级共享约定（`~/.omp/agent/AGENTS.md`）与硬性规则（`~/.omp/agent/RULES.md`）已自动注入，无需重复。
- 项目级 `.omp/config.yml` 可覆盖全局设置（数组类设置会整体替换，注意）。

## 测试后磁盘残留检查（必须执行）

- 每次运行 pytest 之后，必须执行 `.venv/Scripts/python scripts/check_tmp_space.py` 检查临时目录占用。
- 背景：测试 fake 死循环曾在系统 Temp 的 `pytest-of-Glow` 写入 78G 残留（已修复并清理）。
- pytest 临时目录已固定到项目内 `.pytest-tmp`；检查脚本同时监控系统 Temp 的旧残留位置。
- 脚本告警（>500MB）时：确认残留来源后删除，禁止带病继续。

# PRAgent 来源抓取安全边界

PRAgent 只在用户显式提交 URL 时抓取普通网页。Semantic Scholar/Crossref/arXiv adapter 与 arXiv PDF 下载只访问主机白名单内的固定官方 host，并复用 `ingestion/safe_fetch.py` 的 `pinned_get` 单跳 SSRF 防护（协议/凭据/私网地址校验 + DNS pinning），重定向仅允许在白名单主机之间逐跳重新校验；任意 URL 导入必须经过完整 `SafeFetcher` 流程（MIME/2xx/redirect 决策）。

## SSRF 防护

每一跳都重新执行以下检查，重定向不能继承上一跳的信任：

1. 仅允许 `http` / `https`，拒绝 URL credentials、控制字符、反斜杠、无效 host/port 和 IPv6 zone identifier；
2. 使用系统 resolver 取得所有 A/AAAA 最终地址；没有地址、无效地址，或任意地址不是公网 global unicast 时整次请求 fail closed；
3. 明确拒绝 loopback、RFC1918/private、link-local、multicast、unspecified、reserved、IPv4-mapped 私网 IPv6 以及云 metadata 地址；
4. transport 直接连接已验证并固定的 IP。HTTP `Host` 与 HTTPS SNI/证书校验仍使用逻辑 hostname，连接阶段不会再次按 hostname 解析，从而关闭 DNS check/use rebinding 窗口；
5. 301/302/303/307/308 的 `Location` 使用 `urljoin` 后进入下一轮完整校验；重定向循环和超过上限均拒绝。

抓取使用总 deadline、响应 byte 上限、`Accept-Encoding: identity` 和显式 redirect 上限。只接受 `text/html` / `application/xhtml+xml`；错误 MIME、负数/无效/过大 `Content-Length`、流式读取超过上限、空正文及非 2xx 状态都会返回稳定错误码。生产 transport 不读取无限 redirect body，也不采用宿主 HTTP proxy 自动重写目标。

这里防止的是应用主动访问内网目标，不保证外部网页可信。网页文本进入检索和模型上下文后仍属于 untrusted prompt data，不能提升为指令或绕过工具确认。

## Snapshot 与正文

- 原始响应 bytes 以 SHA-256 命名为 `<sha>.html.gz`，保存在 `PRA_DATA_DIR/snapshots/`；gzip `mtime=0`，同一 bytes 得到相同文件。
- 写入使用同目录临时文件、`fsync`、`0600` 权限和原子 rename；已有同 hash 文件必须先解压并复核 hash，损坏时拒绝覆盖。
- SQLite 只保存 snapshot 相对文件名、hash、最终 URL、抽取正文和规范化 metadata，不保存主机绝对路径。
- Trafilatura 只产出文本和 metadata。原始 HTML 不写入 provider record、不放入 API JSON，也绝不注入应用 DOM 或执行其中 script。
- UI/API 对来源只返回题录、状态和安全标识；snapshot 相对路径、内部 locator、抽取全文仍是 storage boundary 内部数据。

## 明确限制

首版不执行 JavaScript、不登录、不处理付费墙、不递交网页表单，也不做通用爬虫。DNS 解析仍依赖操作系统 resolver 的可用性；总 deadline 在每次已解析连接上强制执行，但无法保证所有平台的底层 resolver 都支持同样精度的取消。TLS 使用系统 CA 与 hostname verification。

自动测试全部注入 DNS 和 transport fixture，不访问真实网站。Live smoke 必须由用户显式执行并单独记录，不能把 fixture 结果描述为实时 provider 可用性。

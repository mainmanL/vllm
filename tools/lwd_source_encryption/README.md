# LWD 源码加密 PoC（静态保护 / AES-256-GCM）

本目录是两仓（`vllm` + `vllm-ascend`）LWD 源码加密 PoC 的全部交付物：构建/安装/启动/校验/卸载工具，
以及由它实际生成的两个模块密文。目标是让部署镜像里的这两个模块**不能以明文 `.py` 存在**，同时保持
普通 Python 对象语义（`__file__`、`__spec__.origin`、`co_filename`、`inspect.getsource`、`linecache`、
`multiprocessing` spawn 等）。

> **这是静态保护 PoC，不是长期密钥保密方案。** 密钥 `runtime_key.b64` 与工具一起分发；运行时
> `loader.get_source()` 按设计返回明文（这是明确接受的边界，见「安全边界」一节）。

## 1. 保护对象

| 模块 | 仓库 | 仓库相对路径 | 密文（构建后） |
| --- | --- | --- | --- |
| `vllm.v1.lwd_control.lwd_comm.lwd_control_plane` | `vllm` | `vllm/v1/lwd_control/lwd_comm/lwd_control_plane.py` | 同目录 `lwd_control_plane.py.lwdc` |
| `vllm_ascend.distributed.lwd_data_plane.lwd_transfer_queue` | `vllm-ascend` | `vllm_ascend/distributed/lwd_data_plane/lwd_transfer_queue.py` | 同目录 `lwd_transfer_queue.py.lwdc` |

构建后这两个 `.py` 被删除，旁边留下 `.py.lwdc`；其余业务模块不受影响。

本 PoC 使用的基线：

- vLLM：分支 `feat/lwd-dsv41-apc`，commit `9d77763c63eb729c2776af512282d7b9dcb098b9`
- vllm-ascend：分支 `debug/lwd-h2d-async-after-first-verify`，commit `a68c9509b3342f9bafd84feb396c784aabc033c3`
  （这是创建 PoC worktree 时该仓的实际工作分支；`release/lwd-v0.30.0` 当时停在 `a06430314`）

两个 PoC worktree 都从上述 HEAD 拉出同名本地分支 `debug/lwd-source-encryption-poc`（**未 commit、未 push**；
本机路径为 `C:\Users\root\projects\lwd\vllm-lwd-source-encryption-poc-wt` 与
`C:\Users\root\projects\lwd\vllm-ascend-lwd-source-encryption-poc-wt`）。当前交付的只是**未提交的工作树
diff**，远端拿不到；验收通过后按 §6.1 的两阶段流程：按路径 `git add` → commit → push 两个仓的同名
`debug` 分支，再由远端 `git fetch` + `checkout`。验收前不要用 `git bundle` 搬运——bundle 只含已提交
对象，会漏掉全部 diff 与 untracked 密文/工具。

## 2. 交付文件

| 文件 | 说明 |
| --- | --- |
| `lwd_poc.py` | 命令行入口：`build` / `install` / `launch` / `verify` / `uninstall` |
| `lwd_source_runtime.py` | 远端真正安装的标准 import hook（`MetaPathFinder` + `Loader`）；也支持 `python -m lwd_source_runtime --uninstall` |
| `manifest.json` | 两个模块的模块名、repository、仓库相对 source/payload 路径、明文 SHA-256/字节数、密文 SHA-256/字节数、canary、格式版本；**不含任何本地绝对路径** |
| `runtime_key.b64` | 本次 PoC 的 AES-256 密钥（base64，32 字节），由 `build` 首次运行时生成 |
| `<module>.py.lwdc` | 构建产物，两个仓库各一个 |

`manifest.json` 是整个工具的唯一「保护清单」：`build` 读它、填哈希；`install`/`verify` 读它决定要
服务哪些模块。新增受保护模块时把它加进 `modules`，并补 `verification` 段即可，工具代码无需改动。

## 3. 安全边界（务必先读）

- **静态保护**：删掉明文字节、按模块加密、运行时内存中解密并编译。能挡住 `grep`/`cat`/`tar`
  镜像里的源码、`inspect.getsource` 之外的直接文件读取，挡不住有权限的进程在运行时 dump 内存。
- **分支与 `.git` 历史里仍有明文**：本次 PoC 的 `debug/lwd-source-encryption-poc` 分支只是从**工作树**
  删除了两个 `.py`；用 `git show <rev>:<path>`、翻分支历史、或读远端 `.git` 都能还原明文。
  所以它**只能做功能 PoC，不能当安全交付**。正式镜像必须不带 `.git`，也不能带任何含明文的构建层/
  缓存/构建上下文，并应从干净上下文重新构建；仅靠本工具删除工作树里的 `.py` 不构成交付级保护。
- **密钥不保密**：`runtime_key.b64` 与工具同目录分发，能读它就能解密任意密文。这是 PoC 的既定边界，
  不是缺陷；长期方案需要 KMS/TPM/远程 attestation 之类的外部信任根。
- **明文可通过 `loader.get_source()` 获得**：为让 `inspect.getsource`/`linecache` 正常工作，loader 实现了
  标准 `get_source()`，它返回明文源码。这是明确接受的边界。
- **直接 `open(module.__file__)` 按预期失败**（`FileNotFoundError`）。`verify` 会显式检查这一点，
  不会去规避它。
- 不 patch `open` / `inspect` / `linecache` / CPython；不写目标模块的 `.pyc`；任何损坏、哈希不符、
  密钥错误、payload 缺失都**直接报错，绝不回退明文**（明文本来就不在）。
- 本地开发机没有可用的 Docker/NPU 远端环境，因此真实 `vllm`/`vllm_ascend` 的**运行时导入**在本地
  没有验证过（缺 `torch`/`torch_npu` 等依赖）。loader/verify 的正确性用独立合成包在临时目录里端到端
  验证（见 §9），真实两个文件的构建前后 SHA-256 与密文解密一致性则用真实产物验证（见 §10）。

## 4. 密文格式（固定 34 字节头 + 模块名 + 密文）

| 偏移 | 长度 | 字段 |
| --- | --- | --- |
| 0 | 8 | magic `LWDCSRC1` |
| 8 | 2 | format version（BE，当前 `1`） |
| 10 | 2 | flags（BE，必须 `0`，预留） |
| 12 | 2 | 模块名 UTF-8 字节长度（1..255） |
| 14 | 12 | nonce（每模块独立随机，`os.urandom(12)`） |
| 26 | 8 | ciphertext+tag 长度（BE，含 16 字节 GCM tag） |
| 34 | n | 模块名 UTF-8 |
| 34+n | m | `AESGCM.encrypt` 输出（密文 ‖ tag） |

- 算法：`cryptography.hazmat.primitives.ciphers.aead.AESGCM`，AES-256-GCM，无附加明文后缀。
- AAD = `magic ‖ version ‖ flags ‖ name_len ‖ 模块名`，即格式版本与**完整模块名**都绑定进 GCM 认证。
- 解析是严格的：magic/version/flags/长度范围/UTF-8/点名匹配逐项校验；**多一个字节（trailing）或
  少一个字节（truncated）都报错**，空文件、非 UTF-8 模块名、`ciphertext_len <= 16` 同样拒绝。
- 校验顺序：`manifest.plaintext_sha256`（解密后）与 `manifest.payload_sha256`（原始密文）都核对；
  `install` 预检、运行时安装、每次 import 解密都会核。

## 5. 本地构建

依赖：Python 3.10+（工具本身兼容 3.10，不依赖 3.14 专有 API）、`cryptography`（AESGCM）、
`pybase64`、`regex`。后两者是 vLLM 自身的依赖（`requirements/common.txt`），远端环境本来就有；
本仓库的 `check-forbidden-imports` 规则禁止直接 `import base64` / `import re`，所以本工具按仓库规则写
`import pybase64 as base64` / `import regex as re`，不要改回标准库同名模块（pre-commit 会拦）。

在 vLLM worktree 里执行（`build` 只写两个仓库的文件 + 本目录的 `manifest.json`/`runtime_key.b64`，
不 import vLLM、不需要 NPU）：

```bash
cd /path/to/vllm-worktree
python tools/lwd_source_encryption/lwd_poc.py build \
    --vllm-root    /path/to/vllm-worktree \
    --ascend-root  /path/to/vllm-ascend-worktree
```

行为：

1. 读取 `manifest.json`，按 `--vllm-root` / `--ascend-root` 解析每个模块的绝对 source/payload 路径
   （路径必须落在对应仓库根内；`..`、绝对路径、反斜杠一律拒绝）。
2. 密钥：`manifest.json` 的 `key_file` 指向的 `runtime_key.b64` 不存在时生成 32 字节随机密钥
   （原子写入，权限 0600）；已存在则校验必须是 32 字节 base64。
3. 每个模块独立随机 nonce → 加密 → **原子写** payload → 重新读回、解密、比对 SHA-256 与逐字节内容。
4. 选择 canary：从明文里挑最长且**在两仓其它文件里都不存在**的行，记录其 `sha256(去空白行)` 到
   manifest；用于 `verify` 发现「明文被复制/漏删」。
5. 所有 payload 校验通过后，**先原子写 manifest**，再删除两个明文 `.py` 并清理它们的旧 `__pycache__` pyc。

安全语义（防止把「已删明文/已有密文」误当成新源）：

- 明文在、payload 在、解密结果与明文一致：默认跳过（`--force` 可用新 nonce 重新加密）。
- 明文不在、payload 在、解密结果与 manifest 记录一致：报告 `already built`，不做任何事（幂等）。
- 明文不在、payload 在、与 manifest 记录**不一致**：直接失败，提示 `git checkout -- <source>` 恢复后重建。
- 明文不在、payload 也不在：直接失败。

## 6. 远端部署（仅支持 editable）

假定远端标准目录 `/vllm-workspace/vllm` 与 `/vllm-workspace/vllm-ascend`，两台都由 `--vllm-root` /
`--ascend-root` 覆盖。

### 6.1 取代码（两阶段：当前只交待验收 diff）

**阶段 1 —— 现在（待验收）**：只交付**未提交的工作树 diff**。两个仓当前都是：

- 删除：`vllm/v1/lwd_control/lwd_comm/lwd_control_plane.py`、
  `vllm_ascend/distributed/lwd_data_plane/lwd_transfer_queue.py`；
- 新增（untracked）：两个 `.py.lwdc` 密文；vLLM 仓的 `tools/lwd_source_encryption/`
  （`lwd_poc.py`、`lwd_source_runtime.py`、`manifest.json`、`runtime_key.b64`、`README.md`）。

这些内容**还没有 commit/push**，所以远端此时拿不到任何产物；也**不要**用 `git bundle`——bundle
只打包已提交对象，未提交的 diff 和 untracked 文件不会被带走。

**阶段 2 —— 验收通过后（由提交者执行）**：在两个仓分别**按路径** `git add` 指定文件、commit，
push 同名 `debug/lwd-source-encryption-poc` 分支，然后在远端 `git fetch` + `git checkout`：

```bash
# 1) vLLM 仓：只 add 这里列出的内容
cd /path/to/vllm-worktree
git add vllm/v1/lwd_control/lwd_comm/lwd_control_plane.py \
        vllm/v1/lwd_control/lwd_comm/lwd_control_plane.py.lwdc \
        tools/lwd_source_encryption
git commit -m "debug(lwd): add source-encryption PoC for the control-plane module"
git push origin debug/lwd-source-encryption-poc

# 2) vllm-ascend 仓
cd /path/to/vllm-ascend-worktree
git add vllm_ascend/distributed/lwd_data_plane/lwd_transfer_queue.py \
        vllm_ascend/distributed/lwd_data_plane/lwd_transfer_queue.py.lwdc
git commit -m "debug(lwd): add source-encryption PoC for the transfer-queue module"
git push origin debug/lwd-source-encryption-poc

# 3) 远端机器
cd /vllm-workspace/vllm
git fetch origin debug/lwd-source-encryption-poc
git checkout -B debug/lwd-source-encryption-poc FETCH_HEAD
cd /vllm-workspace/vllm-ascend
git fetch origin debug/lwd-source-encryption-poc
git checkout -B debug/lwd-source-encryption-poc FETCH_HEAD
```

> 不要用 `git add -A` / `git add .`：会把 `artifacts/`、临时目录等无关内容一起带进分支。
>
> 即使完成阶段 2，分支的父 commit 仍包含明文（见 §3 安全边界）：这个分支只能用于功能验证，
> 正式镜像要从干净上下文重建且不带 `.git`。

### 6.2 环境前提（已有环境不要重装）

`install` 只做**只读预检**，不会重装任何包。如果目标环境里 `vllm` / `vllm_ascend` 已经能从当前
解释器导入（editable 部署），**跳过本节直接做 6.3**——不需要、也不应该为了跑 PoC 重新
`pip install -e .`。

预检会用 `importlib.util.find_spec("vllm")` / `find_spec("vllm_ascend")` 取顶层包的
`submodule_search_locations`，要求与 `--vllm-root` / `--ascend-root` 指向的包目录
（`<root>/vllm`、`<root>/vllm_ascend`）realpath 一致；不一致（wheel/镜像内副本安装、另一棵
checkout、参数指错）会在**任何删除/写入之前** fail-fast。**非 editable 部署就停在这里**，
按实际部署方式处理，不要为了绕过预检去改部署或重装。

只有确实要从零搭一个 editable 环境时，才沿用仓库原有流程：

```bash
cd /vllm-workspace/vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd /vllm-workspace/vllm-ascend
pip install -v -e . --no-build-isolation --no-deps
```

`install` 会做**包根一致性预检**：用 `importlib.util.find_spec("vllm")` /
`find_spec("vllm_ascend")` 取顶层包的 `submodule_search_locations`，必须与 `--vllm-root` /
`--ascend-root` 指向的包目录（`<root>/vllm`、`<root>/vllm_ascend`）realpath 一致；不一致（例如另一个
editable 安装、或参数指错）会在**任何删除/写入之前** fail-fast（见 6.2）。

### 6.3 install

```bash
cd /vllm-workspace/vllm
python tools/lwd_source_encryption/lwd_poc.py install \
    --vllm-root   /vllm-workspace/vllm \
    --ascend-root /vllm-workspace/vllm-ascend
```

预检（写任何文件之前全部完成）：

1. 依赖可导入（`cryptography` / `pybase64` / `regex`）；manifest schema/format_version/cipher/nonce_bytes 正确；
2. `runtime_key.b64` 是 32 字节；
3. 两份 payload 存在、大小与 manifest 一致、密文 SHA-256 与 manifest 一致、按模块名严格解析通过、
   解密后明文 SHA-256 与字节数一致；
4. 两个模块的明文 `.py` **必须不存在**（否则普通 path finder 会赢过密文，直接拒绝）；
5. 顶层包 spec 与 `--*-root` 一致（见 6.2）；
6. 目标 purelib 可写。

然后只写三个文件到 `sysconfig.get_paths()["purelib"]`：

| 文件 | 作用 |
| --- | --- |
| `lwd_source_runtime.py` | 标准 Finder/Loader（与工具目录同源复制） |
| `lwd_source_protection.json` | 运行配置：manifest 绝对路径、key 绝对路径、两个仓库根、被保护模块清单 |
| `lwd_source_protection.pth` | 单行：`import lwd_source_runtime ...; install(suppress_errors=True)` |

`.pth` 由 `site` 在每个普通新解释器启动时执行，因此包括 `multiprocessing` 的 **spawn** 子进程、
`subprocess` 起的普通 Python 都会自动安装同一个 finder。重复 `install` 是幂等的（内容相同则不改写，
只重新清理目标模块的旧 `__pycache__` pyc）。

`--purelib` 只用于预发/测试覆盖，正常远端不要传。

### 6.4 launch（可选）

```bash
python tools/lwd_source_encryption/lwd_poc.py launch \
    --vllm-root   /vllm-workspace/vllm \
    --ascend-root /vllm-workspace/vllm-ascend \
    -- vllm serve <原参数...>
```

先做与 `install` 完全相同的预检 + 安装，然后 `os.execvp` 执行 `--` 之后的原始命令（参数原样，不用
`runpy`，不留包装父进程，退出码/信号语义与直接执行一致）。`launch` 面向 Linux；Windows 上
`os.execvp` 的语义不同，不建议使用。

### 6.5 verify

```bash
python tools/lwd_source_encryption/lwd_poc.py verify \
    --vllm-root   /vllm-workspace/vllm \
    --ascend-root /vllm-workspace/vllm-ascend
```

默认（不做篡改）检查项：

- 静态：两个明文 `.py` 不存在；两份 `.py.lwdc` 存在且大小/SHA-256 与 manifest 一致；目标模块
  `__pycache__/*.pyc` 不存在；canary 扫描（**排除 `.git`**）没有发现任何明文行泄漏；manifest 内无绝对路径。
- 预检 + 安装（幂等），并报告写了几个文件、清了几个旧 pyc。
- **全新解释器**（不是父进程）里：finder 已由 `.pth` 安装、finder 服务的模块名集合与 manifest 完全一致；
  两个模块都由 `LwdSourceProtectionLoader` 导入，`__file__` / `__spec__.origin` / `get_filename`
  与原始 `.py` 路径一致；`loader.get_source()` 的 SHA-256 与 manifest 一致；`inspect.getsource(module)`、
  `linecache.getlines(__file__)` 与源码一致；`get_code()` 是普通 `CodeType` 且 `co_filename` 正确；
  关键类/方法（来自 manifest 的 `verification.module_attributes`）存在且方法 `__code__.co_filename`
  指向原路径；`open(__file__)` 抛 `FileNotFoundError`；导入后仍无 pyc。
- `multiprocessing.get_context("spawn")` 子进程（新 PID）重新导入两个模块并校验哈希/`__file__`/loader。
- 继承检查：明文 `vllm_ascend...host.lwd_host_transfer_queue.LwdHostTransferQueue` 仍能继承加密后的
  `LwdTransferQueue`（`issubclass` + 基类方法 `CodeType`）。

结果逐条打印 `[PASS]/[FAIL]` 与 `RESULT: PASS|FAIL (n/m)`；任一条失败退出码非 0。

`--tamper`：**只在服务停止时使用**。备份第一份 payload，翻转最后一个字节，用全新解释器导入并要求失败
（错误必须来自本 hook：`payload authentication failed` / `is not servable` / `LwdSourceProtectionError`），
`finally` 恢复原密文并核对 SHA-256。默认 verify 不做篡改。

## 7. uninstall 与恢复明文

```bash
# 方式一：CLI
python tools/lwd_source_encryption/lwd_poc.py uninstall

# 方式二：运行时模块自带入口（任意能 import 到 lwd_source_runtime 的地方）
python -m lwd_source_runtime --uninstall
python -m lwd_source_runtime --check     # 查看已安装状态
```

`uninstall` **只**删除本工具安装的 `lwd_source_runtime.py`、`lwd_source_protection.json`、
`lwd_source_protection.pth` 及其 `__pycache__` 下的 runtime pyc；删除前按内容标记确认是自己装的，
不会碰别人的 `.pth` 或同名文件，也不会操作其它包。

它**不会**自动把明文写回仓库。恢复明文的方式是切回原分支：

```bash
cd /vllm-workspace/vllm        && git checkout feat/lwd-dsv41-apc
cd /vllm-workspace/vllm-ascend && git checkout debug/lwd-h2d-async-after-first-verify
# 或按目标发布分支：
#   git checkout release/lwd-v0.30.0
```

恢复分支后 `.py` 回来、`.py.lwdc` 消失（未跟踪文件请自行清理），再按正常流程重装即可。

## 8. 运行时实现要点

- `LwdSourceProtectionFinder` 是精确匹配：只对 `manifest` 里列出的完整模块名返回 spec，其它模块一律
  返回 `None`，交回 `PathFinder`。安装幂等：`sys.meta_path` 里已存在同类 finder 时原地更新。
- `spec.has_location = True`，`spec.origin` = 原始 `.py` 绝对路径，因此 `module.__file__`、
  `__spec__.origin` 与普通导入一致。
- `Loader.exec_module` 每次从 payload 解密（校验明文 SHA-256）→ `compile(source, origin, "exec",
  dont_inherit=True)` → `exec`。`get_source()`/`get_code()`/`get_filename()`/`get_data()`/`is_package()`
  都按标准语义实现，`get_data()` 只对原始路径返回明文，其它路径抛 `OSError`。
- 不实现 `set_data()`，也不通过 `SourceLoader` 的 pyc 缓存路径，因此不会写目标模块的字节码。
- 启动期（`.pth`）安装时若某模块 payload 有问题：向 stderr 打一条 `[LWD] source protection: ...`
  警告并把错误记在 finder 里；该模块那次 `import` 会抛出带具体原因的 `ImportError`（绝不回退明文）。
  配置/密钥整体不可用时同样只警告、不禁用解释器（避免 `.pth` 让整台机器的 Python 起不来）。

## 9. 本地自测（合成包，无 vLLM 依赖）

自测不使用 pytest/UT，而是在临时目录里造两个合成仓库（`synth_control.plane`、`synth_data.queue` 与
`synth_data.host.queue`），直接跑真实 CLI。为验证真实部署路径，它把 hook 装进**运行它的解释器**的
`sysconfig` purelib —— 与真实 `install` 完全相同，因此之后启动的新解释器（包括 venv 自己的
`site-packages` 和 `multiprocessing` spawn 子进程）都会自动加载 `.pth`。测试在 `finally` 里卸载，
并把测试开始前就存在的同名文件恢复原样，不留残留。

```bash
# 用仓库自己的 venv（cryptography 已装好），不要另外找系统 python
.venv/bin/python /path/to/selftest.py --workdir /tmp/lwd_poc_selftest
# Windows: .venv\Scripts\python.exe selftest.py --workdir %TEMP%\lwd_poc_selftest
```

（该脚本不属于交付物，仅用于验收。`--workdir` 只决定合成仓库放哪，不决定安装位置——安装位置始终是
当前解释器的 purelib。若工作目录不可写，脚本会把合成仓库放到 `mkdtemp` 实际给出的目录并打印出来。）

## 10. 已知限制

- 静态保护，运行期明文可被 `loader.get_source()` 取到；`--tamper` 只改密文最后一个字节，属最小验证。
- 只支持 editable 部署；非 editable（wheel/安装副本）不在本 PoC 范围。
- `install` 写的是**当前解释器**的 purelib；重建 venv 后需要重新 `install`（旧目录里的三份文件随
  venv 一起消失）。
- `.pth` 是唯一自动加载机制：如果运行时用 `python -S`（不加载 `site`），finder 不会被安装，此时
  两个模块会直接 `ModuleNotFoundError`（仍然是 fail-closed，不会回退明文）。
- 性能未评估：每进程启动多一次 manifest/key/payload 校验（两个小文件解密，毫秒级），每个模块 import
  时解密一次；相对 vLLM 启动开销可忽略，但没有实测数据。
- 本地无 Docker/NPU 环境，真实两仓的**运行时导入**未验证（见 §3）。

## 11. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `ERROR: top-level package 'vllm' is not importable ...` | 该解释器里没有可导入的 vllm；确认它是 editable 部署（见 6.2），非 editable 就停下 |
| `... differs from the repository root ...` | 参数指到了另一棵 checkout；`--vllm-root` 必须是当前解释器实际 import 的那棵树 |
| `plaintext source ... is still present` | 这棵 checkout 里还有明文；先在本地 `build` 或 `git checkout` 原分支再决定 |
| `payload authentication failed ...` | 密文被改/密钥不对：从 git 恢复 `.py.lwdc` 或重新 `build` |
| import 报 `ModuleNotFoundError` 且 stderr 有 `[LWD] source protection:` 警告 | `.pth` 装上了但配置/密钥/payload 有问题；按警告修好后重新 `install` |
| `inspect.getsource` 返回空 | 说明 finder 没装（`python -m lwd_source_runtime --check` 确认），或模块不是由本 hook 导入的 |

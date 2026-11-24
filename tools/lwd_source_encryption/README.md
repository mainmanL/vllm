# LWD 全量运行时源码加密 PoC（AES-256-GCM / memfd / aarch64 ELF）

本目录提供两仓 LWD 运行时 Python 源码的发现、清单、加密、静态校验、安装、启动、真实导入验证、卸载，
以及单文件 aarch64 ELF 构建工具。构建后工作树只保留 `*.py.lwdc`，对应明文 `.py` 被删除；运行时由标准
`MetaPathFinder`/`Loader` 在内存中解密、校验并编译，不回写明文或受保护模块的 pyc。

> **这是静态保护 PoC，不是密钥保密或运行时防 dump 方案。** Git 历史仍含明文和早期 PoC 密钥；
> `loader.get_source()` 为兼容 inspect 按设计返回明文；ELF 中的 raw key 和进程环境对同权限用户可读。
> 正式交付必须从不含 `.git`、明文构建层和缓存的干净上下文重建，并使用外部信任根管理长期密钥。

## 1. 全量范围与计数

范围由 [`lwd_source_scope.json`](lwd_source_scope.json) 审核，不由手工模块列表维护。发现脚本只扫描
`vllm/`、`vllm_ascend/` 包根内的运行时 `.py` 和 `.py.lwdc`：

- 包根下任一目录名以 `lwd` 开头，则包含该目录全树的 `.py`，包括所有 `__init__.py` 和不以
  `lwd` 开头的文件名；
- 或文件 stem 以 `lwd` 开头，则包含该独立模块；
- 排除 `tests`、`docs`、`tools`、`__pycache__` 等目录，不扫描上游核心文件。

必须存在的范围根包括：

| 仓库 | 范围根 |
| --- | --- |
| vLLM | `vllm/v1/lwd_control`、`vllm/v1/lwd_comm`、`vllm/config/lwd`、`vllm/distributed/lwd` |
| vLLM | `vllm/entrypoints/openai/lwd`（包括 `__init__.py` 和 `frontend_attach.py`） |
| vLLM | `vllm/lwd_logger.py`、`vllm/model_executor/model_loader/lwd_weight_filter.py` |
| Ascend | `vllm_ascend/distributed/lwd_data_plane` |
| Ascend | `vllm_ascend/worker/{lwd_base_model,lwd_cloud_model,lwd_edge_model,lwd_hash,lwd_debug}` |

当前固定计数为 **97 = vLLM 60 + Ascend 37**。此前按分仓 59/38 的估算与通用规则不一致；实际通用范围
总计仍为 97，现以可重复发现的 60/37 为准。计数、缺失必需根、缺失/孤儿 payload、重复模块、大小写路径
冲突、manifest 与发现结果不一致、构建后意外明文都会 fail-fast。

发现同时识别 `.py` 与 `.py.lwdc`，所以全量构建后的重跑仍发现 97 个，不会变成 0。未变化源码复用已有
payload；只有源码变化或 `--force` 时才生成新的随机 12 字节 nonce。

## 2. 工具文件

| 文件 | 作用 |
| --- | --- |
| `lwd_source_scope.json` | 可审查的通用范围规则、必需路径、60/37/97 固定计数 |
| `discover_lwd_sources.py` | 独立发现 CLI 和 manifest/discovery reconciliation |
| `build_lwd_source_encryption.py` | 一键 wrapper：discover → manifest → encrypt → static，可选 ELF/QEMU |
| `lwd_poc.py` | `discover/manifest/keygen/build/static/install/launch/verify/uninstall` |
| `lwd_source_runtime.py` | 安装到 purelib 的 Finder/Loader、AES-GCM payload 和配置校验 |
| `manifest.json` | 确定排序的 97 项清单；POSIX 相对路径、plaintext/payload SHA-256 和 bytes、repository、canary、原两模块 attribute/subclass 验证 |
| `build_lwd_source_poc.py` | 默认读取本目录资产，生成静态 aarch64 ELF，可做 QEMU 检查 |
| `lwd_source_poc_launcher.c` | raw key[32] + Python bundle 的 memfd 启动器模板 |
| `runtime_key.b64` | **仅本地构建/ELF provisioning；精确 ignore，不提交，运行时不得读取它作为回退** |
| `lwd_source_poc` | 可选 ELF 输出；精确 ignore，不提交 |

`manifest.json` 由发现结果生成，不应手改模块项。排序键为 `(repository, source)`；`source` 和 `payload`
均为仓库相对 POSIX 路径，`payload == source + ".lwdc"`。空/短 `__init__.py` 也参与相同哈希和加密流程。

## 3. 密钥规则与安全边界

### 3.1 本地生成和提供

`keygen` 只在文件不存在时生成 32 字节随机 key；已存在时只校验，不覆盖：

```bash
cd /path/to/vllm
uv run --no-project --with cryptography --with pybase64 --with regex \
  python tools/lwd_source_encryption/lwd_poc.py keygen
```

Bash 在当前 shell 提供 key（不会打印 key）：

```bash
export LWD_SOURCE_KEY_B64="$(< tools/lwd_source_encryption/runtime_key.b64)"
```

PowerShell：

```powershell
$env:LWD_SOURCE_KEY_B64 = (Get-Content -Raw tools/lwd_source_encryption/runtime_key.b64).Trim()
```

远端应从 secret manager 或 checkout 外的受限文件注入，例如：

```bash
export LWD_SOURCE_KEY_B64="$(< /run/secrets/lwd_source_key)"
```

**install/launch/verify 和已安装 runtime 只接受 `LWD_SOURCE_KEY_B64`。没有文件回退。** 本地
`runtime_key.b64` 仅用于生成环境变量和构建 ELF；配置中禁止 `key_path`，purelib 不安装 key 文件。
`.gitignore` 精确包含：

```text
/tools/lwd_source_encryption/runtime_key.b64
/tools/lwd_source_encryption/lwd_source_poc
```

### 3.2 不可忽略的边界

- 从 Git 跟踪移除 key **不会改写历史**。本分支父提交及早期 PoC 提交仍可能含 key 和全部明文；
  能访问历史的人可恢复它们。正式安全交付不能只依赖当前工作树删除。
- AES-256-GCM AAD 绑定 magic、格式版本、flags 和完整模块名；payload 严格检查截断、尾随字节、模块名、
  ciphertext SHA-256、plaintext SHA-256 和 bytes。
- `inspect.getsource`、`linecache` 和 `loader.get_data(origin)` 需要明文，故 `get_source()` 返回明文是
  明确接受的边界。空模块的 `inspect.getsource(module)` 与 CPython 原生空模块一样可抛
  `OSError("could not get source code")`；`linecache` 和 `get_data` 分别返回空行/空 bytes。
- 不 patch `open`、`inspect`、`linecache` 或 CPython。直接 `open(module.__file__)` 按预期失败。
- payload 缺失、错误 key、认证失败、哈希不符、语法错误、意外明文均 fail-closed，绝不回退明文。

## 4. 本地全量构建

依赖 Python 3.10+、`cryptography`、`pybase64`、`regex`。以下命令在 vLLM worktree 执行；wrapper 会从
本地 ignored key 文件或已有环境变量向子进程提供 `LWD_SOURCE_KEY_B64`，不会打印 key：

```bash
cd /path/to/vllm
uv run --no-project --with cryptography --with pybase64 --with regex \
  python tools/lwd_source_encryption/build_lwd_source_encryption.py \
    --vllm-root /path/to/vllm \
    --ascend-root /path/to/vllm-ascend
```

可选同时构建 ELF 并跑 QEMU：

```bash
# Windows 默认委托 WSL Ubuntu-22.04；Linux/WSL 中直接执行同一 wrapper。
uv run --no-project --with cryptography --with pybase64 --with regex \
  python tools/lwd_source_encryption/build_lwd_source_encryption.py \
    --vllm-root /path/to/vllm \
    --ascend-root /path/to/vllm-ascend \
    --elf --qemu
```

未指定 `--elf-output` 时输出到工具目录的精确 ignored 文件 `lwd_source_poc`。

`--force` 强制全部重新加密（产生新 nonce）；默认复用未变化 payload。重复执行完整 wrapper 后，发现数量、
manifest 和未变化 payload 的哈希保持不变。

也可分步运行（先按 §3.1 export key）：

```bash
TOOL=tools/lwd_source_encryption/lwd_poc.py
ROOTS="--vllm-root /path/to/vllm --ascend-root /path/to/vllm-ascend"
python "$TOOL" discover $ROOTS
python "$TOOL" manifest  $ROOTS
python "$TOOL" build     $ROOTS
python "$TOOL" static    $ROOTS
```

独立发现 CLI：

```bash
python tools/lwd_source_encryption/discover_lwd_sources.py $ROOTS
python tools/lwd_source_encryption/discover_lwd_sources.py $ROOTS --require-built --json
```

### 4.1 `static` 与真实 import 严格区分

`static` **不导入 vLLM/vllm-ascend，不需要 NPU**：重新发现并对账 97 项，拒绝意外明文/孤儿/重复，
检查 payload 文件哈希，并对全部 97 项执行 **decrypt + plaintext SHA-256 + compile**，另检查 canary 和 pyc。
其成功输出为 `RESULT: STATIC PASS`，不能解释为真实运行时导入成功。

真实环境再运行 §5.4 的 `verify`；本地缺 `torch`、`torch_npu`、UCX、NPU 驱动时不得把 synthetic/static
结果写成真实 NPU 验证结果。

## 5. 远端 checkout / install / launch / verify / uninstall

假设两仓已发布同名 `debug/lwd-source-encryption-poc` 分支，目标目录为 `/vllm-workspace/vllm` 和
`/vllm-workspace/vllm-ascend`：

```bash
cd /vllm-workspace/vllm
git fetch origin debug/lwd-source-encryption-poc
git checkout -B debug/lwd-source-encryption-poc FETCH_HEAD
cd /vllm-workspace/vllm-ascend
git fetch origin debug/lwd-source-encryption-poc
git checkout -B debug/lwd-source-encryption-poc FETCH_HEAD
```

### 5.1 环境前提

当前仅支持 editable 部署。`install` 不会重装包；它用 `find_spec` 核对顶层包位置与命令行仓库根一致，
不一致或不可写时 fail-fast。远端已有环境不要为重跑 PoC 重装。确需从零搭建时才使用仓库既有 editable
流程。所有命令先按 §3.1 注入 `LWD_SOURCE_KEY_B64`。

### 5.2 install

```bash
cd /vllm-workspace/vllm
.venv/bin/python tools/lwd_source_encryption/lwd_poc.py install \
  --vllm-root /vllm-workspace/vllm \
  --ascend-root /vllm-workspace/vllm-ascend
```

预检包括 97 项 discovery/reconcile、payload 认证与 SHA-256、全部源码 compile、包根一致性和 purelib
可写性。安装仅写：

- `lwd_source_runtime.py`；
- `lwd_source_protection.json`（manifest 路径或 inline manifest、仓库根、模块清单、`key_source=env`，不含 key）；
- `lwd_source_protection.pth`（普通 Python/site 启动自动注册 finder，spawn 子进程同样生效）。

### 5.3 launch

```bash
.venv/bin/python tools/lwd_source_encryption/lwd_poc.py launch \
  --vllm-root /vllm-workspace/vllm \
  --ascend-root /vllm-workspace/vllm-ascend \
  -- vllm serve <原参数...>
```

`launch` 先做与 install 相同的预检和安装，再 `execvp` 原命令，面向 Linux。

### 5.4 真实运行时 verify（需要完整依赖；NPU 环境执行）

```bash
.venv/bin/python tools/lwd_source_encryption/lwd_poc.py verify \
  --vllm-root /vllm-workspace/vllm \
  --ascend-root /vllm-workspace/vllm-ascend
```

它在 fresh interpreter 中真实导入 97 个模块，并检查：

- loader、`__file__`、`__spec__.origin`、解密源码 SHA-256/bytes、`get_code().co_filename`；
- 加密 `__init__.py` 的真实 `is_package`、`submodule_search_locations`、`__path__` 和子模块 import；
- `inspect`、`linecache`、`get_data(origin)`、直接 open 失败、受保护模块无 pyc；
- multiprocessing **spawn** 新 PID 的导入、包语义和源码哈希；
- 原 `lwd_control_plane` / `lwd_transfer_queue` 的 attribute 检查，以及
  `LwdHostTransferQueue -> LwdTransferQueue` subclass 检查。

服务停止时可加 `--tamper`：翻转第一项 payload 的一个字节，要求 fresh import 失败，随后恢复并核对哈希。
本地无完整 NPU 依赖时不要运行全量 `verify` 冒充通过；应运行 `static` 和独立 synthetic 验证。

### 5.5 uninstall

```bash
.venv/bin/python tools/lwd_source_encryption/lwd_poc.py uninstall
# 或
.venv/bin/python -m lwd_source_runtime --uninstall
.venv/bin/python -m lwd_source_runtime --check
```

只删除带本工具标记的 runtime/config/pth 和 runtime pyc，不删除 payload，也不恢复明文。恢复开发明文请在
独立工作树使用原分支/基线；不要在交付目录重新写入 `.py`。

## 6. aarch64 单文件 ELF

在 WSL `Ubuntu-22.04` 安装 `aarch64-linux-gnu-gcc` 和 `qemu-aarch64` 后：

```bash
python3 tools/lwd_source_encryption/build_lwd_source_poc.py \
  --output /tmp/lwd_source_poc --qemu
```

`--source-dir` 和 `--launcher` 默认都是工具目录内资产，无需指定。构建 C/header 等中间文件只存在于
`TemporaryDirectory`。输出是静态 aarch64 ELF，不提交到仓库。

ELF 内含：

- `lwd_poc.py`、`lwd_source_runtime.py`、`discover_lwd_sources.py`、manifest 和 scope 的 Python bundle；
- **raw 32 字节 key**；构建脚本额外断言 base64 key 文本未出现在 ELF 中；
- 启动时通过匿名 `memfd` 和 `/proc/self/fd/<N>` 执行 bundle，并自行导出 `LWD_SOURCE_KEY_B64`，不落临时
  Python 文件，也不读取本地 key 文件。

```bash
qemu-aarch64 /tmp/lwd_source_poc --lwd-print-meta
qemu-aarch64 /tmp/lwd_source_poc --lwd-help
# QEMU 检查还实际经 memfd 启动 bundled Python，执行一次无副作用 uninstall。

# 在真实 aarch64 环境中可直接使用：
./lwd_source_poc install --vllm-root /vllm-workspace/vllm \
  --ascend-root /vllm-workspace/vllm-ascend
./lwd_source_poc verify --vllm-root /vllm-workspace/vllm \
  --ascend-root /vllm-workspace/vllm-ascend
./lwd_source_poc launch --vllm-root /vllm-workspace/vllm \
  --ascend-root /vllm-workspace/vllm-ascend -- vllm serve <原参数...>
./lwd_source_poc uninstall
```

固定工具/payload/scope/key 输入时，`-Wl,--build-id=none` 和无时间戳资产生成使重复 ELF 构建哈希一致；
`--qemu` 检查 metadata、help 和 memfd Python bundle。QEMU 成功仍不等于真实 aarch64 NPU 运行时验证。

## 7. 运行时实现与限制

- `LwdSourceProtectionFinder` 精确匹配 manifest 模块；`__init__.py.lwdc` 对应 package，`find_spec` 设置
  `submodule_search_locations=[原始包目录]`，loader 的 `is_package` 返回真实布尔值。
- 非 package 的 search locations 为 `None`。package loader 的 `get_data` 支持解密后的自身源码和包目录内
  非 Python 资源；路径逃逸和其他 `.py`/`.lwdc` 直接读取被拒绝。
- 每次 import 解密并校验明文哈希，按原始 `.py` 路径 compile/exec；不实现 `set_data`，不走 pyc 写入路径。
- `.pth` 安装阶段发生配置/key/payload 错误时输出 `[LWD] source protection:` 警告；受保护 import 给出具体
  错误，绝不回退。`python -S` 不加载 site/pth，因此不受本工具支持。
- 每进程启动需校验并 compile 全量清单，性能影响需在目标环境评估；当前没有伪称性能数据。
- QEMU/静态/synthetic 都是不同层级的本地验证；真实 aarch64、真实 NPU import、模型 serve 和性能仍需目标
  环境验收。

# LWD GPU Edge Upstream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the LWD edge execution path into vLLM, add GPU Edge → NPU Cloud over UCX with rank-replay, and preserve NPU Edge → NPU Cloud over HCCL and UCX.

**Architecture:** Extract protocol/data-plane code into vLLM, keep only device operations in GPU/NPU backends, and build a dedicated `LwdEdgeModel` containing only embed, optional vision modules, and lm_head. Use `LwdBaseGPUModelRunner → LwdEdgeGPUModelRunner`, without a Runtime layer or a zero-decoder-layer full-model fallback.

**Tech Stack:** Python, PyTorch, vLLM V1 worker/model runner, UCX/UHC1, CUDA, Ascend NPU/HCCL, pytest, ruff.

## Global Constraints

- DOWN uses post-norm hidden states plus `top_id_ths`; production never sends token IDs.
- `LwdBaseGPUModelRunner` is the interim base name; no `LwdEdgeRuntime`.
- No full `ForCausalLM` construction and no `make_layers`, decoder placeholders, zero-layer MoE fixes, `hc_head`, final norm, MTP, or KV cache on the edge.
- GPU → NPU supports host UCX only; NPU → NPU retains device HCCL and host UCX.
- Cloud Engine/Scheduler/Worker/ModelRunner stays in vllm-ascend and only consumes shared code.
- GPU edge is V1, single-card, BF16, `in_process`; unsupported configurations fail at startup.
- Do not commit or push unless the user explicitly asks.

---

### Task 1: Share UP payload, staging, and scrambling

**Files:**
- Create: `vllm/v1/lwd_comm/lwd_up_payload.py`
- Create: `vllm/v1/lwd_comm/lwd_pinned_staging.py`
- Create: `vllm/distributed/lwd/lwd_payload_scrambler.py`
- Modify: `vllm-ascend/vllm_ascend/worker/lwd_cloud_model/lwd_cloud_worker.py`
- Modify: `vllm-ascend/vllm_ascend/worker/lwd_base_model/lwd_base_model_runner.py`
- Modify: `vllm-ascend/vllm_ascend/worker/lwd_cloud_model/lwd_cloud_model_runner.py`
- Test: `vllm/tests/v1/lwd_comm/test_lwd_up_payload.py`
- Test: `vllm/tests/v1/lwd_comm/test_lwd_pinned_staging.py`

**Interfaces:**
- Produces: `LwdUpPayloadLayout.pack/unpack`, `LwdPinnedStaging.stage_to_device`, `LwdPayloadScrambler.scramble/descramble`; vllm-ascend imports these exact public names from vLLM.

- [ ] **Step 1: Write failing shared-payload tests**

Cover BF16 embeddings, expert IDs greater than 255, Engram dimensions, absent segments, wrong frame lengths, and byte-for-byte scramble inversion on CPU. Reuse the existing Ascend payload/staging tests as the behavioral baseline instead of inventing new semantics.

- [ ] **Step 2: Run the tests before moving code**

Run: `uv run pytest tests/v1/lwd_comm/test_lwd_up_payload.py tests/v1/lwd_comm/test_lwd_pinned_staging.py -v`
Expected: FAIL because the vLLM modules do not exist.

- [ ] **Step 3: Move implementations and switch Ascend imports**

Move the current implementations without changing wire layout or ring semantics. Replace Ascend imports with:

```python
from vllm.v1.lwd_comm.lwd_up_payload import LwdUpPayloadLayout
from vllm.v1.lwd_comm.lwd_pinned_staging import LwdPinnedStaging
from vllm.distributed.lwd.lwd_payload_scrambler import (
    lwd_get_payload_scrambler,
    lwd_payload_scrambler_initialized,
)
```

- [ ] **Step 4: Run shared and existing Ascend tests**

Run the new vLLM tests and the existing Ascend UP payload, pinned staging, scrambler, cloud pre-recv, and host-stage tests. Expected: PASS with no payload or staging behavior change.

- [ ] **Step 5: Review the diff**

Confirm there is exactly one production implementation of each shared component and no forwarding module or duplicate wire layout.

---

### Task 2: Extract transfer types, future, FIFO, and data plane

**Files:**
- Create: `vllm/distributed/lwd/lwd_types.py`
- Create: `vllm/distributed/lwd/lwd_transfer_future.py`
- Create: `vllm/distributed/lwd/lwd_transfer_queue.py`
- Create: `vllm/distributed/lwd/lwd_data_plane.py`
- Modify: `vllm-ascend/vllm_ascend/distributed/lwd_data_plane/lwd_transfer_queue.py`
- Modify: `vllm-ascend/vllm_ascend/distributed/lwd_data_plane/lwd_data_plane.py`
- Modify: `vllm-ascend/vllm_ascend/distributed/lwd_data_plane/host/lwd_host_transfer_queue.py`
- Test: `vllm/tests/distributed/test_lwd_data_plane.py`

**Interfaces:**
- Produces: `LwdTransferRequest`, `LwdTransferFuture`, `LwdBaseTransferQueue`, `LwdDataPlane`; backend supplies `device`, current stream/event operations, and queue factory. Existing `submit_send`, `submit_recv`, `skip_seqno`, `abort_connection`, `poll_completions`, and `shutdown` signatures remain stable.

- [ ] **Step 1: Write failing FIFO and future tests**

Test out-of-order hold/release, duplicate and skipped seqnos, deferred bind, contiguous watermark, abort, shutdown, and error propagation. Use a fake queue `_execute` and host completion event, not a physical NPU.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/distributed/test_lwd_data_plane.py -v`
Expected: FAIL because shared classes are absent.

- [ ] **Step 3: Extract protocol code and leave HCCL execution in Ascend**

`LwdBaseTransferQueue` owns sequencing/snapshot/reap. `LwdTransferFuture.wait_on_current_stream` calls an injected stream hook. `LwdDataPlane` creates queues through the backend selected during worker initialization; no request-path `get_current_vllm_config` or backend import from vLLM.

- [ ] **Step 4: Run vLLM and Ascend transfer tests**

Expected: new shared tests and existing Ascend queue, host queue, transfer staging, abort, and early-recv tests PASS.

- [ ] **Step 5: Check dependency direction**

Scan vLLM shared modules for `vllm_ascend` and `torch_npu`; expected: no matches. HCCL group/channel creation remains only in vllm-ascend.

---

### Task 3: Implement GPU host UCX backend

**Files:**
- Create: `vllm/distributed/lwd/host/ucx_block_policy.py`
- Create: `vllm/distributed/lwd/host/lwd_host_channels.py`
- Create: `vllm/distributed/lwd/host/lwd_host_transfer_queue.py`
- Create: `vllm/distributed/lwd/host/ucx_io.py`
- Modify: `vllm-ascend/vllm_ascend/distributed/lwd_data_plane/host/lwd_host_transfer_queue.py`
- Test: `vllm/tests/distributed/test_lwd_host_data_plane.py`

**Interfaces:**
- Consumes: shared transfer types/future/FIFO and existing `vllm.v1.lwd_comm.data_comm.LwdUcxTransportPort`.
- Produces: `LwdGpuUcxChannelIo`, host queue allocation on `device="cuda"`, GPU stream/event completion, and host-channel initialization from `LwdConfig`.

- [ ] **Step 1: Write failing GPU IO tests**

Mock CUDA stream/event and transport port. Verify send staging waits for the producer stream, D2H completes before send, receive performs H2D before recording the event, oversized frames fail before queueing, and receive tensors use `record_stream(torch.cuda.current_stream())`.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/distributed/test_lwd_host_data_plane.py -v`
Expected: FAIL because the GPU IO/backend does not exist.

- [ ] **Step 3: Implement shared host policy and GPU device assembly**

Use the existing UHC1 port without changing its protocol. Bind CUDA device in sender and receive-callback threads. Reuse `LwdUpPayloadLayout.bytes_per_token` for automatic UP sizing, including optional Hash and Engram segments.

- [ ] **Step 4: Run host policy and mocked GPU IO tests plus Ascend host regression**

Expected: PASS. NPU host continues to pass its existing tests, proving the extraction did not alter host behavior.

- [ ] **Step 5: Verify real hardware coverage separately**

Record the commands for a two-process GPU/NPU loopback of BF16 UP/DOWN frames, aborts, and shutdown. Do not mark real UCX or rank-replay verified without a GPU/NPU machine.

---

### Task 4: Move edge runner and worker into vLLM

**Files:**
- Create: `vllm/v1/worker/lwd_edge/lwd_edge_worker.py`
- Create: `vllm/v1/worker/lwd_edge/lwd_base_gpu_model_runner.py`
- Create: `vllm/v1/worker/lwd_edge/lwd_edge_gpu_model_runner.py`
- Create: `vllm/v1/worker/lwd_edge/lwd_edge_mm_cache.py`
- Create: `vllm/v1/worker/lwd_edge/lwd_hash_tables.py`
- Modify: `vllm/platforms/cuda.py`
- Modify: `vllm-ascend/vllm_ascend/worker/lwd_edge_model/lwd_edge_model_runner.py`
- Test: `vllm/tests/v1/worker/lwd_edge/test_lwd_edge_model_runner.py`
- Test: `vllm/tests/v1/worker/lwd_edge/test_lwd_edge_worker.py`

**Interfaces:**
- Produces: `LwdBaseGPUModelRunner(GPUModelRunner)`, `LwdEdgeGPUModelRunner.execute_lwd_embed`, `execute_lwd_unembed`, `_lwd_watermarked_output`, `lwd_release_finished`; worker dispatches only `LwdBatchType`.

- [ ] **Step 1: Write failing runner/worker tests**

Use a fake model and fake data plane to assert request-row ordering, stable-sort replay, UP watermark, MM lifecycle, empty batches, finished-request cleanup, empty KV spec, skipped warmup, and `data_plane_comm=device` rejection.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/v1/worker/lwd_edge -v`
Expected: FAIL because the classes are absent.

- [ ] **Step 3: Implement runner and worker**

Move the existing edge business logic into `LwdEdgeGPUModelRunner`; keep only shared staging and device hooks in `LwdBaseGPUModelRunner`. Do not call native `_prepare_inputs`, full forward, sampler, or CUDA graph capture. Ascend reuses the public execution class and overrides NPU hooks.

- [ ] **Step 4: Run runner/worker tests and existing edge tests**

Expected: new tests and existing edge picker/client/executor proxy/style tests PASS. GPU modules contain no Ascend imports.

- [ ] **Step 5: Review rank-replay**

Assert hidden is used directly by `compute_logits`, no norm runs on the edge, and `LwdC2eNotify.token_ids` is not read by production execution.

---

### Task 5: Build a dedicated LwdEdgeModel without decoder layers

**Files:**
- Create: `vllm/model_executor/models/lwd/lwd_edge_model.py`
- Create: `vllm/model_executor/models/lwd/lwd_edge_model_loader.py`
- Create: `vllm/model_executor/models/lwd/deepseek_v4.py`
- Modify: `vllm/model_executor/models/utils.py`
- Modify: `vllm/model_executor/model_loader/lwd_weight_filter.py`
- Modify: `scripts/lwd/split_lwd_checkpoint.py`
- Modify: `vllm/models/deepseek_v4/nvidia/model.py`
- Modify: `vllm/models/deepseek_v41/nvidia/model.py`
- Test: `vllm/tests/model_executor/models/lwd/test_lwd_edge_model.py`

**Interfaces:**
- Produces: `LwdEdgeModel.embed_input_ids`, `compute_logits`, `load_weights`; `LwdEdgeModelLoader(BaseModelLoader)` overrides only `create_model`; per-architecture builders reuse native `VocabParallelEmbedding`, `ParallelLMHead`, `LogitsProcessor`, quantization, and mapper helpers.

- [ ] **Step 1: Write failing dedicated-model tests**

Build DeepSeek V4 and V4.1 text edge models with dummy configs. Assert there are no decoder, `PPMissingLayer`, norm, `hc_head`, MTP, MoE, or KV modules; assert `get_expert_mapping` and full-model finalize are never called; assert embed/logits and filtered weight loading work.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/model_executor/models/lwd/test_lwd_edge_model.py -v`
Expected: FAIL because the dedicated builder and loader do not exist.

- [ ] **Step 3: Implement the dedicated builder and loader**

The loader delegates `download_model` and `load_weights` to the selected existing loader and constructs only `LwdEdgeModel`. Reuse native mapper builders. Make `hc_head` cloud-only in ownership/filter/split rules. Other architectures remain rejected until they have a dedicated builder; never fall back to a zero-layer full model.

- [ ] **Step 4: Run dedicated-model and checkpoint tests**

Expected: dedicated tests and existing weight-filter/split tests PASS after updating the expected `hc_head` ownership. Confirm no `make_layers` call occurs in the new edge path.

- [ ] **Step 5: Update affected design documentation**

Align `lwd_role_module_skip.md`, `lwd_weight_split.md`, `lwd_gpu_edge_upstream.md`, and applicable `AGENTS.md` ownership descriptions with the dedicated edge-model behavior.

---

### Task 6: Integration, regression, and support matrix

**Files:**
- Modify: `vllm/tests/v1/lwd_control/test_lwd_public_surface.py`
- Modify: `vllm-ascend/tests/ut/worker/test_lwd_style_guard.py`
- Modify: `vllm/docs/design/lwd_gpu_edge_upstream.md`

**Interfaces:**
- Produces: verified common-code imports and explicit supported/rejected deployment matrix; real GPU/NPU results are reported separately from mocked/unit results.

- [ ] **Step 1: Add dependency and support-matrix tests**

Assert GPU → NPU allows only host, NPU → NPU allows device and host, V2 and sidecar fail fast, and production never enables a DOWN token-id path.

- [ ] **Step 2: Run vLLM focused tests**

Run all tests added above plus existing LWD config/control/model-loader tests and `git diff --check`. Expected: PASS.

- [ ] **Step 3: Run vllm-ascend focused regression**

Run existing worker, payload, staging, data-plane, host-channel, pre-recv, cloud early-recv, and style-guard tests. Expected: PASS.

- [ ] **Step 4: Run pre-commit on changed files**

Run both repositories' configured pre-commit/ruff checks on changed files. Expected: PASS.

- [ ] **Step 5: Verify real GPU/NPU rank-replay when hardware is available**

Compare cloud-sampled tokens and edge-replayed tokens for text, long chunks, concurrency, abort, prefix/refill, structured output, multimodal, and MTP. Run NPU → NPU once with HCCL and once with UCX. Record TTFT/TPOT and any unverified scenario explicitly rather than claiming it works.

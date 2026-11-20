# LWD Host Staging

This document describes the host-staging implementation on `feat/lwd-main-migration`. Uncached UP prompt data has one arrival path: an early receive hint posts a receive into a cloud device slot, segmented D2H copies populate per-request pinned buffers, and only the resulting arrival reports advance the host-arrival watermark. Prefix-hit rows are seeded at their negotiated boundary instead of being transferred again.

The cloud DP endpoint rank owns the cross-node UP receive and `LwdHostStager`; other cloud TP ranks receive the assembled window through the existing TP broadcast. Prefix caching uses nonce-scoped reservations and the refill protocol below. Cloud-side preemption is supported through the minimal recompute semantics below.

## Storage and watermarks

The first hint for a request allocates one pinned BF16 buffer for the **whole prompt**, including any DeepSeek V4 tail. Later chunks reuse that allocation at absolute prompt offsets. Full consumption marks the request done but does not free its buffer. Storage is released by `release_request` when the request reaches `finished_req_ids`, not when its last row is consumed.

There is no aggregate host-memory capacity or per-request byte cap. The configured transport windows below are not host-storage limits.

For request `r`, define:

- **W1** (`recv_rows`): the contiguous prompt high watermark available for scheduling. A negotiated prefix boundary seeds it; completed and published D2H arrivals advance it thereafter.
- **W2**: the scheduler-side check that native prefill scheduling stays within W1 and within the selected arrival window.

The stager maintains `0 <= consumed <= W1 <= prompt_rows`. Arrival gaps are rejected, repeated ranges are idempotent, and reads cannot cross W1. W2 requires `start + scheduled <= W1` and `scheduled <= window_rows`; native under-scheduling is allowed and retried later. At dispatch, the runner additionally requires `consumed == num_computed_tokens == window.start`.

The fixed **10/2/2** pipeline limits are internal implementation constants, not topology settings:

| Limit | Value | Meaning |
| --- | ---: | --- |
| `LWD_MAX_PENDING_CHUNKS` | 10 | Per-cloud-DP edge gate on announced batches not yet covered by request settlement or the UP send watermark |
| `_LWD_EARLY_RECV_MAX_INFLIGHT` | 2 | Cloud communication-thread limit on submitted, not yet reported UP receives |
| `_LWD_HOST_STAGE_SLOTS` | 2 | Reusable device receive slots |

A device slot is bounded by the maximum batched UP payload and is released after its D2H staging completes. It is distinct from the uncapped, request-lifetime host allocation.

## Single-path arrival flow

```mermaid
sequenceDiagram
    participant ES as Edge scheduler
    participant EW as Edge worker
    participant CE as Cloud engine
    participant CT as Cloud endpoint comm thread
    participant HS as Host stager / copy streams
    participant CS as Cloud scheduler
    participant CR as Cloud model runner

    ES->>CE: RequestNotify(prompt metadata, hashes, optional nonce)
    opt Prefix caching enabled
        CE-->>ES: READY(nonce, H) after pinned reservation
    end
    ES->>CE: RangeNotify(seqno, request rows and offsets)
    CE->>CT: Sideband hint(seqno, rows, offsets, prompt lengths)
    CT->>HS: Allocate device slot and full-prompt host buffers
    CT->>HS: Plan per-request D2H segments and post UP irecv
    ES->>EW: EMBED batch with the same seqno
    EW->>HS: UP embeddings and optional DSV4 tail
    HS->>HS: recv done_event -> segmented D2H -> staged_event
    CT->>CE: Report (UP, seqno) and cumulative rows offset + rows
    CE->>CS: Advance channel watermark and per-request W1 by max
    CS->>CS: Select windows and verify W2
    CS->>CR: Dispatch native mixed prefill/decode batch
    CR->>HS: H2D selected host windows; advance consumed
    CR->>CR: TP broadcast and inject remote embeddings / DSV4 tails
```

A hint is geometry and an early-posting trigger, **not** proof of arrival. `RangeNotify` alone never advances W1. The arrival report is emitted only after both the receive completion event and the D2H staging event complete. The engine drains that report at the start of a step, merges each cumulative request row count by maximum, and separately advances the UP channel watermark.

### Hint and arrival contract

Each hint carries `seqno`, total rows, and aligned request-ID, chunk-row, prompt-offset, and prompt-length tables. Duplicate hints and range notifications do not submit another receive. The stager plans source segments for the entire batch so that skipping an ended request does not shift the source offsets of live requests.

After D2H completion, `publish_received` advances each live request's contiguous W1 and releases the device slot. The completion report contains `(request_id, offset + rows)`, so the scheduler receives cumulative rather than per-chunk row counts.

Two different UP watermarks must not be conflated with W1:

- The **cloud receive watermark** is the highest staged-receive seqno reported on the channel, initially `-1`. It retires per-seqno engine bookkeeping; per-request W1, not this channel scalar, drives prefill readiness.
- The **edge send watermark** is the contiguous successful send-completion prefix per cloud DP. The edge worker returns it with normal and placeholder outputs, and the edge scheduler uses it to retire pending-batch backpressure. An out-of-order or failed send cannot advance it.

## D2H and H2D event ordering

For each staged receive, the UP channel stream records `done_event` after HCCL completion and any descrambling. A dedicated D2H stream waits for that event, issues non-blocking copies into the planned per-request host views, and records `staged_event`. The communication thread polls both events without blocking, publishes arrivals only after staging completes, and then drops its reference to the device receive tensor. The next channel operation waits on the predecessor's receive event, not on its D2H event.

For a scheduled prompt window, the endpoint runner selects one of two rotating device windows. Its dedicated H2D stream first waits for the window's previous-use event, then copies the selected embeddings and any DSV4 tails and records a ready event. The main stream waits for that ready event before injection and TP consumption. After dispatch, the stager advances `consumed`; injection records the window's new used event so a later H2D cannot overwrite storage still in use.

## Minimal cloud preemption and recompute

The cloud scheduler allows native preemption. Preemption rolls back `num_computed_tokens` but retains both W1 and the full-prompt host buffer, so a resumed request recomputes from rows that have already arrived.

Before native `_update_states`, the endpoint runner compares each scheduled cached request's host `consumed` pointer with its new computed-token count. If `consumed` is ahead, it rewinds only that pointer to the new count; W1 is unchanged. For a re-admission self-hit that moves computed tokens forward, it instead skips `consumed` forward, never beyond W1. Either direction restores `consumed == num_computed_tokens == window.start` before the next H2D window.

During visible-set queue reassembly, preempted requests return to the waiting-queue head in their native post-scheduling order, ahead of non-preempted waiting requests. MTP follows the native preemption path; this minimal unit adds no separate MTP mechanism.

## Prefix caching and refill

### Nonce-scoped READY and fail-closed reservations

With prefix caching enabled, the edge rebuilds the prompt hash chain using the cloud's hash block size and assigns a monotonically increasing nonce. It announces the request but holds it outside the native queue. The cloud reserves the hit before native admission and replies `READY(request_id, H, nonce, error)`. Only a current, matching READY enqueues the request. Stale replies or timeouts are ignored; a current error or timeout aborts the waiting request. The same generation nonce is carried through refill requests, range items, host staging, and `refill_done`.

The initial reservation pins real blocks, freezes the block-group snapshot and **H** (the initial READY hit), and sets the staging floor **S = H** through `lwd_resume_cap_tokens`. Native allocation must consume that exact snapshot; reservations are released after successful scheduling or request cleanup. If a trimmed request's reservation snapshot is unavailable, lookup fails closed rather than silently renegotiating a different prefix and consuming mismatched embeddings. Re-admission reservations pin the new hit without changing S.

### H, H', S, and refill dispatch

Within this protocol, the three boundaries are:

- **H**: the initial prefix hit returned by READY. A first UP chunk at offset H seeds W1, `consumed`, and the host stage floor at H, so rows below H need no transfer.
- **H'**: the cache hit found on re-admission, clamped to W1 rounded down to the scheduler block alignment; the reserved block groups are truncated to the same boundary.
- **S**: the current staging floor, initially H. It separates the prefix region covered by the current cache negotiation from ordinary UP arrival accounting.

If H' >= S, the request resumes at H' as a self-hit and the runner skips host `consumed` forward. If H' < S, the cloud requests embeddings for `[H', S)` and gates that request's prefill windows. Host staging tracks the contiguous refill low watermark and emits `refill_done` only after the hole is filled. A matching completion lowers S to H' and reopens scheduling; stale completions cannot change the gate or floor.

The edge splits the requested refill range into chunks bounded by `min(chunk_unit, max_num_batched_tokens)` and dispatches them before ordinary EMBED work. Building or queuing a refill does **not** announce it. At each budget-sized dispatch, the edge revalidates the nonce and route, sends `RangeNotify`, then allocates the seqno and registers its pending batch. This keeps announcement, UP transfer, and backpressure on the same dispatch path.

### Prompt-only caching, MM filtering, and guards

The cloud adopts the edge's initial prompt hashes. Output tokens never extend that hash chain, and KV block-cache writes are capped at `num_prompt_tokens`; hashing and cache commits are therefore prompt-only. LWD reservation lookups return `shared_prefix_boundary = 0`.

Normal and refill embedding paths process only MM features whose placeholder spans intersect the current chunk. Features wholly covered by a prefix hit or located in a future chunk need no data or hash lookup for this dispatch. Refill temporarily supplies the request's positional features for intersection gathering, then restores the live positional state while retaining the encoded-output cache.

With prefix caching enabled, the MTP guard validates computed-token offsets and requires authoritative prompt embeddings for continued prefill. It rejects token-ID fallback when those embeddings are unavailable; pure decode batches keep the native behavior.

The native `enable_mamba_fine_grained_prefix_cache` mechanism is separate from the LWD reservation result and remains **off by default**. When explicitly enabled with Mamba `align`, EAGLE on the Mamba group, and a prefix match unit smaller than the Mamba block, it can register a checkpoint at the shared-prefix junction where an EAGLE/MTP sibling resumes.

## Request completion, abort, and tombstones

Normal request completion follows the native `finished_req_ids` path. The cloud worker releases runner bookkeeping and the request's full-prompt host allocation, then removes batch metadata once no live request buffer references it.

Abort has two ordered effects:

1. The cloud engine retires the request from arrival accounting and enters the native abort path. Late arrival reports for that request are ignored.
2. An abort notice attached to an EMBED scheduler output calls `abort_request` in the stager. This records an ended-request tombstone and marks an existing buffer aborted, but does not free it immediately.

The tombstone suppresses new D2H destinations and late arrival publication for that request. Existing host storage remains frozen until the normal finish-release signal. A release also leaves a tombstone, so a late hint cannot recreate the request. The complete seqno hint and receive geometry remain intact for UP FIFO pairing; ended requests simply contribute no host segments or arrivals.

## Full-prompt M-RoPE

For multimodal requests, the edge computes the full-prompt `int64 [3, prompt_len]` M-RoPE positions once at admission and serializes them in `RequestNotify`. The cloud stores them on the request, attaches them to `scheduled_new_reqs`, and the runner validates `3 * prompt_len * 8` bytes before priming the full position tensor and its completion-position delta.

Per-chunk `LwdRangeItem.mrope_positions` remains empty. M-RoPE does not use the UP auxiliary frame and does not participate in host staging.

## DeepSeek V4 tail

For `L` hash layers and `K` selected experts per token, every UP row occupies `H + 4 * L * K` BF16 elements. The batch payload is `[all embedding rows | all encoded expert-ID bytes]`: each int32 expert ID is transported as four numeric byte values, avoiding BF16 rounding of large IDs.

Each request's host buffer has the analogous full-prompt layout `[prompt embeddings | prompt expert-ID tail]`. D2H planning splits both regions at absolute prompt offsets. H2D staging reassembles the scheduled window as embeddings followed by tails; the runner converts the tail bytes back to int32 IDs and registers them at the same absolute prompt positions for Hash MoE routing.

## Implementation map

| Area | Implementation |
| --- | --- |
| Hint creation and completion drain | `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py` |
| W1 window selection and W2 verification | `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_scheduler.py` |
| Prefix reservations and refill negotiation | `control_cloud_scheduler/lwd_prefix_cache.py`, `control_edge_scheduler/lwd_edge_prefix_control.py` |
| Range, send-watermark, and abort routing | `vllm/v1/lwd_control/control_edge_scheduler/` |
| Hint/arrival sideband protocol | `vllm_ascend/worker/lwd_cloud_model/lwd_pre_recv.py` |
| Per-request buffers and tombstones | `vllm_ascend/worker/lwd_cloud_model/lwd_host_stage.py` |
| Receive/D2H and window/H2D execution | `lwd_cloud_worker.py`, `lwd_cloud_model_runner.py`, and vLLM `distributed/lwd/lwd_transfer_queue.py` / Ascend `lwd_hccl_transfer_queue.py` |

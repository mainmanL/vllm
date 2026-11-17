# PD separation on the host LWD data plane

This implementation migrates the PD behavior to the current LWD control plane,
arrival scheduler and Ascend runners. The protocol version is 6 and PD envelopes
use `lwd-pd-v2`. All edge, prefill and decode processes must use matching versions.
The `basic` trees are references and are not deployment dependencies.

## Ownership and routing

```mermaid
sequenceDiagram
    participant F as Edge frontend
    participant E as Edge engine
    participant X as PD Proxy
    participant P as Selected P DP
    participant D as Selected D DP
    F->>E: Original request + unique attempt
    E->>X: Immutable envelope, no raw prompt
    X->>P: Register with explicit target_dp
    P->>P: Reserve and pin prefix snapshot H
    P->>E: BIND (connection, H, P_end, I, R)
    E->>P: Claim same snapshot; upload [H,P_end)
    P-->>X: Mooncake export metadata
    X->>D: Register same envelope + handoff + target_dp
    D->>E: BIND identifying P binding/export
    par Import KV
        P-->>D: Mooncake transfer through native connector
    and Upload tail
        E->>D: Independent tail [R,N)
    end
    D->>D: Admit tail and existing decodes in native budget
    D->>E: Physical frame + committed token interval + digest
    E->>F: Verified token IDs
    F->>E: Reliable local handoff of immutable K and reason
    E->>D: Retry ResultCommit until receipt/deadline
    D-->>E: ResultReceipt
    D-->>X: Same immutable result and usage in SSE
    X-->>E: Control SSE result
```

The Proxy selects both the cloud and its actual global DP index. HTTP registration
requires `body.target_dp`; the Bridge has only the DP queues managed by that
process and rejects a missing or unmanaged target. It never makes a second
placement decision. A shared HTTP service and a separate endpoint per DP are both
supported. External load balancing uses a distinct control listen port for each
global DP.

Each host edge currently has native DP size 1. Its local DP 0 connects to every
linked cloud DP. P and D DP sizes are independent; the example has P=2 and D=4.
A connection is `(edge_id, cloud_id, cloud_dp_idx)`, never a bare DP index.
Multiple independent edges are supported. PD does not use the device data plane,
shared edge/cloud HCCL worlds, or virtual pipeline stages. Intra-cloud collectives
and the separate Mooncake P-to-D transport retain their native responsibilities.

## Request, prefix and execution state

The frontend allocates an attempt ID after the final internal request ID is known.
The envelope fingerprint covers that attempt, original prompt length N, output
limit, sampling, structured constraints, model and input ABI, hash manifest,
tenant, issuance time and TTL. Expired attempts cannot be registered again after
bounded in-memory records have been removed. The Proxy pins its selection for an
attempt and concurrent retries attach to the same stage execution.

P registration reserves the target prefix snapshot and pins it before publishing
BIND. H is capped at `P_end - 1`. RequestNotify claims that exact reservation; it
must not rerun prefix lookup. Fine-grained compressed-cache hits and native
alignment remain authoritative. H is not rounded to the scheduler block size.
The native allocation takes ownership before reservation references are released.

| Model cache semantics | P execution/export P_end | Imported extent I | D computation starts R |
| --- | --- | --- | --- |
| Normal attention | N | N | N-1 |
| Mamba or compressed state | N-1 | N-1 | N-1 |

The native connector performs any P truncation once. Export sizing uses the
request's actual prompt length; import sizing uses I from the handoff even when
`prompt_token_ids` is absent. A stateful one-token prompt has no supported handoff
and is rejected. D retains the original N. Native remote-KV completion must land
at R before tail execution is allowed.

P uses the existing pinned host prompt staging and arrival windows. D owns a
small, independent copy of its tail payload and its completion event; it neither
allocates an N-row embedding buffer nor retains a view into a receive slot.
KV import can progress before the tail arrives. Tail execution requires KV,
tail readiness, transfer credits and native token budget. Existing decode and MTP
requests remain in that scheduling pass. Input windows use per-request caps.

PD preemption is terminal in this version. The engine fails the request and frees
its execution resources; it does not refill the prompt, recompute, restore state,
or silently select another P/D. The non-PD recovery path remains separate.

The original edge request and MM features survive P completion. The D tail is
sliced from the original prompt; full-prompt MRoPE metadata is sent to D. P consumes
only its execution prefix of those positions. Structured constraints are included
in the fingerprint and initialized on D. P's internal export-driving sample does
not advance the user's grammar or enter the output stream.

## Output and final results

P output suppression belongs to an immutable batch snapshot. D's generated count
G advances from native `EngineCoreOutput.new_token_ids`, after native acceptance
and stop processing. Draft candidates and padding rows are not G.

Every PD DOWN notification carries attempt and binding IDs, physical row counts,
valid output intervals, digests and explicit terminal G. The edge deduplicates a
physical sequence before requesting its tensor, receives the full frame, and
verifies the recovered valid token prefix before delivery. Physical credit is
returned for validly consumed padding frames as well as user-token frames.
Window acknowledgments and output credits are cumulative and replay-safe.

K is the frontend detokenizer's retained token-list length. It is frozen before
RequestState deletion on completion, stop string, cancellation, disconnection,
timeout and errors. The frontend retains an independent record until the local
engine utility accepts ownership. A failed utility call keeps the real K for
bounded retry. Cancellation happens after this local ownership step; failures
cannot replace an unknown K with zero.

The engine's small result outbox retries the immutable commitment independently
of model execution and data-plane slots. D joins it with its terminal G:

- `complete` requires K=G.
- Early stop, cancellation, disconnection, timeout and error allow 0<=K<=G,
  including when native execution has already stopped or reached its length cap.
- K cannot exceed the edge-verified prefix. A request canceled before any forward
  has explicit G=0; missing terminal information remains unknown.
- A duplicate commitment is idempotent; a conflicting commitment is rejected.

The receipt and control HTTP SSE publish the same canonical result. Usage is
`prompt_tokens=N`, `completion_tokens=K`, `total_tokens=N+K`, and cached prompt
tokens equal the P snapshot's H. Usage availability and execution status are
separate fields. P's compatibility response has zero placeholder usage and is
not a second final user charge.

Reliability is bounded and process-local. This implementation does not persist
results across process restarts. Full retention tables fail closed; missing
receipts expire as explicitly unconfirmed. Execution slots, UP/DOWN credits and
large request/MM state are not held until billing confirmation.

## Configuration and deployment

The examples in `examples/lwd/` contain a host P2/D4 topology, edge/P/D entry
configuration, and Proxy target lists for shared endpoints and external DP load
balancing. Replace addresses, model settings, secret paths and topology paths.
Set `pd.tenant_id`, or use the edge coordination `consumer_id` as the tenant ID.
The edge's single `coordination.control_url` must point at the PD Proxy.
PD requires an explicit shared `data_plane_host.up_buffer_size` (MiB). The
current host handshake compares maximum frame capacities; independently sizing
each side from its local scheduling budget can produce incompatible capacities.
The example uses 256 MiB. Size this shared value for the largest complete frame,
including auxiliary expert data. PD upload batches also respect the selected
cloud's token budget; non-PD automatic buffer sizing is unchanged.

Run the Proxy with the same Python environment as vLLM:

```bash
python -m vllm.v1.lwd_control.pd.proxy --config examples/lwd/pd_proxy.json
```

Use normal model/TP/DP options for each process, with native edge DP=1 and the
configured cloud DP sizes. Pass the role's `*_config.json` as
`--additional-config`. On P and D, pass `pd_prefill_kv.json` and
`pd_decode_kv.json` respectively as `--kv-transfer-config`. These Mooncake
producer/consumer examples assume TP=4 on both sides, P DP=2 and D DP=4;
the actual native parallel options must agree with the connector configuration.
Adjust transport endpoints and memory capacity for the model. The LWD host
topology does not configure the P-to-D connector. Scheduler limits may differ
across P and D.

PCP/DCP combinations are not newly enabled by this migration. Unsupported overlap
modes must fail explicitly. Sequence-parallel external embeddings are padded and
sliced consistently; DeepSeek V4 expert tables follow the hidden-state layout,
including all-gather and removal of padding. Request-owned draft and accepted
state is restored only for the same live request identity and is invalidated on
replacement, rollback, preemption or finish.

## Verification boundary

CPU tests execute the protocol, host connection creation with fake transports,
HTTP registration and Proxy routing, lifecycle helpers, frontend ownership,
small tensor buffers, mixed-row layouts and connector sizing. They do not prove
real NPU numerical equivalence, stream/event timing, UCX/Mooncake interoperability,
HCCL collective progress, throughput or TPOT. Those require deployment on the
intended Ascend hardware with the same model and load, including multimodal,
structured-output, MTP, cancellation, preemption and asymmetric DP cases.

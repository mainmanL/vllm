# CPU validation of the host PD migration

Validation uses source code from `target/vllm` and `target/vllm-ascend`. No NPU,
real UCX link, Mooncake peer or hosted model is used. The `basic` trees are not
modified. The implementation and deployment contract are documented in
[lwd_pd_host.md](lwd_pd_host.md).

## Reproduction

The checked environment is `/private/tmp/dcn-pd-venv`, created with `uv venv`
and Python 3.12.12. Its main packages are torch 2.11.0, pytest 9.2.0,
pytest-asyncio 1.5.0, pydantic 2.13.5, msgspec 0.22.0, PyYAML 6.0.3,
httpx 0.28.1, FastAPI 0.163.1, aiohttp 3.14.5, numpy 2.5.2,
pybase64 1.5.0 and Ruff 0.16.6. Dependency installation used compatible cached
wheels when the package network was unavailable. No system Python installation
was changed.

From `target/vllm`, run:

```bash
LWD_PD_PYTHON=/private/tmp/dcn-pd-venv/bin/python tools/lwd_validate_pd_cpu.sh
```

For a different checkout layout, set `LWD_ASCEND_SOURCE` to the Ascend source
directory. The default Python path is `.venv-lwd-cpu/bin/python`. An equivalent
uv-managed environment needs the packages listed above.

## Recorded results (2026-11-16)

The final consolidated script completed successfully:

| Suite group | Result |
| --- | --- |
| Protocol, edge/cloud contracts, Proxy and frontend | 69 passed |
| PD topology/configuration and shipped examples | 22 passed |
| Host channel assembly | 3 passed |
| Ascend PD runtime and selected existing regression suites | 111 passed, 10 subtests passed |
| **Script total** | **205 passed, 10 subtests passed** |

An additional existing message/handshake run had **15 passed, 1 skipped**.
All **62 changed Python files** (41 vLLM, 21 Ascend) passed parsing, Ruff
checks and formatting checks. Both `git diff --check` runs and the validation
script's shell syntax check passed. The protected team-standards sections of
both AGENTS files were verified unchanged. No commits or pushes were made.

## What the tests execute

| Area | Evidence |
| --- | --- |
| Topology and configuration | Real schema loading, shared host capacity, P2/D4 connection expansion, exact control queue indices, shipped topology and role examples |
| Host transport assembly | Real channel initialization with fake devices/transports; full connection lookup and one listener per cloud DP |
| Proxy and HTTP | FastAPI/ASGI, real Bridge queues and dispatch, P export and D control SSE; exact P1/D3 routing, retries, TTL and capacity failures |
| Prefix and handoff | Frozen fine-grained H, claim without re-query, disabled cache reads, one truncation, I/R sizing, small-window progress and input credit return |
| Edge lifecycle | Real runtime and scheduler methods at fake engine boundaries; original N survives, only the D tail is resent, DOWN/upload fairness and independent budgets |
| Results | Serialization, digest verification, duplicate/gap/conflict cases, native G vs frontend K, K<G stop, explicit 0/0 cancellation, receipts and bounded backpressure |
| Frontend | Real OutputProcessor/AsyncLLM method bodies with fake engine utilities; detokenizer K frozen before deletion/abort, local handoff failure retains K |
| Ascend runners | CPU tensor ownership, physical/valid rows, request draft state, SP and expert layouts, MM retention and MRoPE boundary handling |
| Native integration seams | Real source methods for batch queue/executor/worker cleanup and Mamba splitting, with hardware operations replaced at their boundaries |

Some existing architecture guards are AST checks. They are included alongside
behavioral tests, not presented as device execution. Targeted tests isolate the
source modules from optional model/plugin imports so that they can run without
installing NPU libraries.

## Existing suite failures and excluded checks

The complete `tests/config/lwd` suite was compared against an archive of
vLLM HEAD `fa61204aa642f6f475c83ddf33b3ed7cf69e1f8f`:

- HEAD: 49 failed, 147 passed.
- Working tree at that comparison point: the same 49 failed, 164 passed.
- The sorted failed-test identities are identical (`diff -u` exits 0).
- Later focused tests added tenant, actual-example and shared-capacity coverage.
  The complete configuration suite is not claimed to pass.

Evidence and exact commands are recorded in
`/private/tmp/dcn-pd-config-baseline-evidence.txt`; stdout is in
`/private/tmp/dcn-pd-baseline-results.txt` and
`/private/tmp/dcn-pd-working-results.txt`.

The existing message/handshake tests additionally passed with 15 passed and one
optional real-SamplingParams import case skipped. The protocol package was
loaded directly, without vLLM's optional model initialization.

No full repository CI or pre-commit hook suite was run. Changed Python files
were parsed and checked with Ruff, and both repositories were checked with
`git diff --check`. There is no claim of NPU numerical equivalence, true
UCX/Mooncake/HCCL progress or a TPOT performance result.

The inherited edge DOWN recovery path can wait for its receive future to bind.
PD does not add that wait; this migration does not redesign the entire DOWN
transport scheduler. D's *new-request* KV/tail admission is independently
progressed and tested without a pending connection blocking another tail.

## Hardware acceptance still required

Use the shipped asymmetric topology with the intended model and TP/DP settings.
Check normal attention and stateful handoffs, partial prefix hits, image/MRoPE,
structured outputs, MTP acceptance/rejection, SP and DeepSeek V4 wide EP. Exercise
cancel/disconnect, delayed KV/tail arrival, duplicate notifications, connection
failure and explicit preemption failure. Compare existing requests' TPOT under
a continuous new-request load against the same baseline hardware/model/load;
record means and tail percentiles, batch composition and MTP behavior. CPU
checks cannot establish a performance threshold or prove there is no regression.

## New or modified test functions for review

Parameterized functions expand into multiple test cases. This list contains
new and changed functions; unmodified tests in the selected suites also run.

### vllm/tests/config/lwd/test_pd_topology.py

- `test_host_edge_reaches_every_asymmetric_stage_engine`
- `test_entry_exposes_stage_and_pd_limits_from_its_topology`
- `test_pd_rejects_invalid_deployment`
- `test_non_pd_host_keeps_all_cloud_connections`
- `test_non_pd_host_retains_automatic_up_buffer_sizing`
- `test_edge_control_plane_advertises_each_clouds_real_dp_count`
- `test_host_native_dp_validation_is_instance_local`
- `test_shipped_host_p2_d4_example_and_all_process_entries_are_loadable`

### vllm/tests/v1/lwd_control/test_lwd_frontend_result.py

- `test_finished_request_freezes_real_k_before_cleanup`
- `test_abort_freezes_existing_token_count_before_removing_external_mapping`
- `test_stale_attempt_cannot_supply_tokens_to_a_reused_request_id`
- `test_unknown_frontend_count_is_not_replaced_with_zero`
- `test_local_handoff_failure_retains_immutable_k_for_idempotent_retry`
- `test_async_abort_hands_result_to_engine_before_aborting_execution`
- `test_expired_local_handoff_is_explicitly_unconfirmed_and_keeps_k`

### vllm/tests/v1/lwd_control/test_lwd_messages.py

- `test_request_notify_round_trips_full_prompt_mrope_positions`

### vllm/tests/v1/lwd_control/test_lwd_pd_cloud_contracts.py

- `test_p_registration_pins_stage_bound_and_claim_reuses_snapshot`
- `test_p_cancel_before_claim_releases_pin_and_never_adds_request`
- `test_fine_grained_snapshot_uses_p_end_bound_and_survives_owner_release`
- `test_http_skip_prefix_cache_is_respected_before_bind`
- `test_cloud_pd_send_backpressure_never_sleeps_and_expires_closed`
- `test_decode_import_progresses_without_tail_then_joins_existing_mtp`
- `test_pd_preemption_fails_without_reimport_or_refill`
- `test_http_bridge_preserves_proxy_dp_and_attempt_result`
- `test_http_export_to_decode_c2e_commit_receipt_share_one_result`
- `test_commit_before_decode_claim_explicitly_confirms_zero_output`
- `test_real_cloud_asgi_validates_frozen_identity_and_explicit_dp`
- `test_late_cancelled_range_does_not_activate_reused_request_binding`
- `test_bind_advertises_effective_cloud_window_budget`
- `test_pd_single_receive_credit_advances_inside_large_alignment_block`
- `test_reused_id_waits_for_native_cleanup_and_unidentified_late_output_is_ignored`
- `test_cancel_before_queued_native_add_does_not_resurrect_stage`

### vllm/tests/v1/lwd_control/test_lwd_pd_edge_contracts.py

- `test_add_builds_real_parseable_envelope_and_waits_for_proxy_bind`
- `test_bind_p1_d3_preserves_separate_routes_caps_and_tail`
- `test_down_frame_received_once_and_only_verified_rows_reach_k`
- `test_late_attempt_and_corrupt_rows_cannot_reactivate_or_advance_output`
- `test_wrong_bind_dp_is_rejected_and_no_local_load_balance_fallback`
- `test_finalize_rpc_keeps_result_after_abort_and_late_bind_does_not_execute`
- `test_tail_and_existing_down_frames_alternate_without_starvation`
- `test_decode_default_credit_covers_its_own_spec_width_with_smaller_edge_budget`
- `test_upload_batch_obeys_selected_cloud_budget_across_requests`

### vllm/tests/v1/lwd_control/test_lwd_pd_protocol.py

- `test_envelope_binds_attempt_constraints_and_expiry`
- `test_handoff_boundary`
- `test_stateful_one_token_and_arbitrary_plan_rejected`
- `test_result_order_independent_and_usage_uses_frontend_prefix`
- `test_unknown_g_not_zero_and_cancel_before_forward_is_explicit`
- `test_invalid_counts_never_publish_a_result`
- `test_output_digest_gap_overlap_replay_and_physical_rows`
- `test_zero_live_padding_can_close_same_endpoint`
- `test_result_retries_identity_receipt_and_expiry`
- `test_asymmetric_bindings_and_cumulative_credits`
- `test_decode_cannot_change_frozen_handoff`
- `test_finalization_freezes_verified_count_before_late_tokens`
- `test_required_pd_wire_fields_and_roundtrip`
- `test_result_receipt_rejects_changed_usage_and_digest`
- `test_old_padding_endpoint_cannot_terminate_a_later_prefix`
- `test_control_backpressure_preserves_peer_fifo_without_blocking_other_peers`
- `test_control_backpressure_is_bounded_and_fails_explicitly`

### vllm/tests/v1/lwd_control/test_lwd_pd_proxy.py

- `test_proxy_p2_d4_routes_distinct_dp_and_preserves_envelope_and_kv`
- `test_same_attempt_http_retries_share_one_execution_and_identical_sse`
- `test_proxy_cache_never_evicts_live_attempt_to_make_room`
- `test_proxy_failure_is_unconfirmed_and_does_not_reselect_or_invent_usage`
- `test_expired_envelope_is_rejected_before_any_stage_selection`

### vllm-ascend/tests/ut/distributed/lwd_data_plane/test_lwd_host_topology.py

- `test_edge_dp_zero_connects_to_every_p_and_d_engine`
- `test_cloud_dp_reuses_one_listener_across_edges`
- `test_host_queue_looks_up_the_full_stage_connection`

### vllm-ascend/tests/ut/worker/test_lwd_cloud_early_recv.py

- `test_d_pending_or_failed_connection_does_not_block_another_tail`

### vllm-ascend/tests/ut/worker/test_lwd_edge_mm_refill.py

- `test_pd_preserves_media_coordinates_after_p_until_d_tail_and_final_finish`

### vllm-ascend/tests/ut/worker/test_lwd_pd_runtime.py

- `test_native_batch_queue_executes_zero_token_finished_flush_and_releases_edge_media`
- `test_d_tail_copies_actual_rows_and_rejects_foreign_or_retired_binding`
- `test_d_submit_skips_full_prompt_staging_and_accepts_only_the_planned_tail`
- `test_edge_drains_full_down_frame_but_only_recovers_valid_rows`
- `test_cloud_physical_carrier_to_edge_recovery_respects_native_commit_prefix`
- `test_d_meta_placeholder_preserves_prompt_shape_without_prompt_storage`
- `test_p_full_prompt_mrope_is_sliced_at_actual_computation_end`
- `test_hash_experts_follow_unequal_dp_hidden_padding_and_sp_shard`
- `test_request_owned_spec_state_survives_departure_but_not_replacement_or_rollback`
- `test_d_tail_mixes_with_decode_without_full_prompt_hash_storage_or_cpu_roundtrip`
- `test_p_suppression_is_immutable_for_the_output_generation`
- `test_mooncake_embeddings_import_uses_i_once_and_export_uses_truncated_p_end`

### vllm-ascend/tests/ut/worker/test_lwd_pre_recv.py

- `test_make_hint_preserves_complete_row_tables`

### vllm-ascend/tests/ut/worker/test_lwd_s1s2_refactor_guard.py

- `test_watermark_mrope_and_expert_failfast_contracts`

### vllm-ascend/tests/ut/worker/test_lwd_v4_mixed_migration.py

- `test_main_and_mtp_shard_only_embedding_rows`

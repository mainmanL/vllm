# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.lwd.lwd_transfer_future import LwdTransferFuture
from vllm.distributed.lwd.lwd_transfer_queue import LwdBaseTransferQueue
from vllm.distributed.lwd.lwd_types import (
    ConnectionId,
    LwdChannelType,
    LwdTransferRequest,
)
from vllm.v1.lwd_comm.data_comm.ucx_channel import LwdHostDoneEvent


class _ImmediateQueue(LwdBaseTransferQueue):
    def _execute(self, request, predecessor, into=None):
        del predecessor
        event = LwdHostDoneEvent()
        event.set()
        tensor = request.tensor if request.op == "recv" else None
        if into is None:
            return LwdTransferFuture(request, [], event, tensor=tensor)
        into._bind(handles=[], done_event=event, tensor=tensor)
        return into


def _request(seqno: int) -> LwdTransferRequest:
    return LwdTransferRequest(
        channel=LwdChannelType.UP,
        op="recv",
        num_elements=1,
        tensor=torch.tensor([seqno], dtype=torch.bfloat16),
        seqno=seqno,
    )


def test_out_of_order_request_is_bound_when_its_predecessor_arrives() -> None:
    queue = _ImmediateQueue(ConnectionId(0, 0, 0), LwdChannelType.UP, "recv")
    deferred = queue.submit(_request(1))
    assert not deferred.done()

    queue.submit(_request(0))

    assert deferred.done()
    assert deferred.result().tensor.item() == 1
    assert len(queue.reap()) == 2


def test_skip_and_duplicate_seqnos_do_not_block_later_requests() -> None:
    queue = _ImmediateQueue(ConnectionId(0, 0, 0), LwdChannelType.UP, "recv")
    queue.skip_seqno(0)

    completed = queue.submit(_request(1))
    duplicate = queue.submit(_request(1))
    later = queue.submit(_request(2))

    assert completed.done()
    assert duplicate.done()
    with pytest.raises(RuntimeError, match="op failed.*seqno=1"):
        duplicate.result()
    assert later.done()
    assert later.result().tensor.item() == 2

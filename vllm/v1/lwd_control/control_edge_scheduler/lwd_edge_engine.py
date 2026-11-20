"""边侧引擎子类:run_engine_core 类选择点注入的边 EngineCore。

边引擎职责:通信面装配(POST_OUT bind/接收线程/PRE_OUT 延迟连接)、
调度器注入(LwdEdgeScheduler)、请求入口/步进/关停的引擎接口覆写。
全部状态收进子类字段,不外挂引擎属性。

构造约束:
  ① 引擎主体先行(super() 最先调):接收线程 WAKEUP 敲门的
     self.input_queue 由 super 创建,后启动线程即无构造期竞态;代价是
     云缺失时引擎主体启动白费,超时路径只清理通信面,主体随进程退出
     兜底回收。通信面随后构建:bind POST_OUT、建云载荷队列与接收线程,
     阻塞等云首拍 HELLO——HELLO 在云引擎全量初始化(权重/KV/图编译)
     完成后才发出,等到了它才允许边侧对外就绪;超时(hello_timeout_s,
     默认 600s,须覆盖云全量启动时长)fail-fast。
     HELLO 首拍一次、无周期重发,不考虑任一侧重启自愈:重启即整组重拉,
     构造期等待是边侧唯一的发现窗口。
  ② 调度器经 scheduler_cls 注入裸类,须在 super() 之前设值——super 内
     构建调度器时一次性消费该配置,后设无效;引擎构造完成后回填
     publisher(早于任何请求,等价构造注入)。

步进编排(embed 优先):步首收割在飞批(unembed 交付 token/请求终结,
embed 登记进度)-> prefill 编排(单请求组批 -> 范围预告 -> executor
异步执行,优先占深度配额)-> 云载荷消费(c2e -> UNEMBED 批,剩余配额)。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_coord_client import (
    LwdCoordClient,
)
from vllm.v1.lwd_debug import LwdDebug, LwdLogBase
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_NOT_FINISHED,
    LWD_WIRE_VERSION,
    LwdC2eNotify,
    LwdHelloNotify,
    LwdPrefixReadyNotify,
    LwdRefillNotify,
    lwd_decode_cloud_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
    LwdConfig,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_prefix_control import (
    LwdEdgePrefixControl,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
    lwd_build_unembed_batch,
)

logger = init_logger(__name__)


def _lwd_disable_down_enabled() -> bool:
    """VLLM_ASCEND_LWD_DISABLE_DOWN=1:云不 collect/不发 DOWN,边直通交付。"""
    return os.environ.get("VLLM_ASCEND_LWD_DISABLE_DOWN") == "1"


# 云->边载荷队列容量:队满时接收线程阻塞在 put,背压沿 ZMQ 直达云侧
# 步发送循环(载荷不可丢)
LWD_C2E_META_QUEUE_MAX = 1000

# 批队列在飞深度上限:派发(embed+unembed 合计)不收割的批数上限,
# 突发积压时连续派发喂饱 worker(参照仓经验 >=4 才能维持流水对齐)
LWD_EDGE_BATCH_QUEUE_DEPTH = 4


class LwdEdgeEngineCore(EngineCoreProc):
    """边 PO 引擎:通信面装配 + 调度器注入 + step/add/abort/shutdown 覆写。"""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        # 调度器注入须赶在 super() 之前:super 内构建 self.scheduler 时
        # 一次性消费 scheduler_cls,后设无效(注入失效,首请求即崩)
        vllm_config.scheduler_config.scheduler_cls = LwdEdgeScheduler
        super().__init__(*args, **kwargs)
        config = LwdConfig.from_env_and_config(vllm_config)
        self._lwd_config = config
        self._lwd_prefix_enabled = bool(
            vllm_config.cache_config.enable_prefix_caching
        )
        self._lwd_prefix_control = (
            LwdEdgePrefixControl(self, config.prefix_timeout_s)
            if self._lwd_prefix_enabled else None
        )
        self._lwd_pending_refills: deque = deque()
        # 层日志总开关:env 已开则不动,config 段开则补开(仅本进程)
        LwdLogBase.set_debug(config.debug)
        # 云 engine 数(= 云侧 data_parallel_size):边侧收齐 M 个 HELLO
        # 才放行服务;每个 HELLO 携带 engine_index 与各自的 PRE_OUT 端点
        self._lwd_expected_engines = (
            vllm_config.parallel_config.lwd_config.cloud_dp_size
        )
        # 通信面:bind POST_OUT 订阅面(M 个云 engine 来连,fan-in);
        # PRE_OUT 发布面 per-engine——端点由各 engine 的 HELLO 通告决定
        # (边不预知云地址),每个新 HELLO 到达即建一条定向 publisher
        self._edge_receiver = self._lwd_build_post_out(config)
        self._edge_senders: dict[int, LwdControlPublisher] = {}
        self._lwd_peer_hellos: dict[int, LwdHelloNotify] = {}
        self._lwd_coord_client: LwdCoordClient | None = None
        # 云->边唯一载荷队列:生产端接收线程,消费端引擎步;数据面经
        # UNEMBED 批的 lwd_c2e_notifies 拿元数据,不直接读队列
        # (单消费者语义)
        self.lwd_c2e_meta_queue = queue.Queue(maxsize=LWD_C2E_META_QUEUE_MAX)
        # 已派发待收割的批队列 (kind, payload, t_dispatch, future):
        # kind="unembed" payload=notify;kind="embed" payload=scheduler_output。
        # 派发不收割,队首 FIFO 收割——必须全局单队列:executor 的 FutureWrapper
        # 按底层队列序排水,交错收割会连带等错批。t_dispatch 供
        # [Lwd][sched] harvest wait(派发→收割)度量批在队列里的滞留时长
        self._lwd_batch_queue: deque = deque()
        # 在飞的 embed 批执行 future(收割点只弹队不等待):水位只能在
        # future 完成后读到,所以挂在这里逐步排空 —— 立刻读会永远读不到
        # (弹队时 worker 往返还没完成),闸门摘除会永久停摆(真机实录)。
        self._lwd_embed_futures_inflight: list = []
        hello_event = threading.Event()
        discovery = threading.Thread(
            target=self._receive_thread,
            args=(hello_event,),
            name="lwd-post-in",
            daemon=True,
        )
        discovery.start()
        if not hello_event.wait(config.hello_timeout_s):
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[Lwd] edge engine init failed: got "
                f"{len(self._lwd_peer_hellos)}/{self._lwd_expected_engines} "
                f"cloud HELLOs within {config.hello_timeout_s}s on POST_OUT "
                f"(bind {config.lwd_post_out_bind_endpoint()}; check cloud "
                f"master_addr connectivity and POST_OUT port)"
            )
        self._lwd_cross_check_hello()
        self._lwd_configure_prefix_cache()

        self.scheduler.lwd_edge_publishers = self._edge_senders
        if self._lwd_prefix_control is not None:
            self.scheduler._lwd_prefix_pending = self._lwd_prefix_control.pending
            self.scheduler._lwd_prefix_hash_block_size = (
                self._lwd_prefix_control.hash_block_size
            )
        # FIRST_REQ 唤醒钩子:仅多云(有 coordinator)时接上;单云 None,
        # 调度器侧跳过
        self.scheduler.lwd_first_req_hook = (
            self._lwd_coord_client.send_first_req
            if self._lwd_coord_client is not None
            else None
        )
        logger.info(
            "[Lwd] edge engine assembled: POST_OUT bind %s, %d cloud "
            "engine(s) discovered, coord_client=%s",
            config.lwd_post_out_bind_endpoint(),
            len(self._edge_senders),
            self._lwd_coord_client is not None,
        )

    # ------------------------------------------------------------------ #
    # 通信面                                                              #
    # ------------------------------------------------------------------ #
    def _lwd_build_post_out(self, config: LwdConfig) -> LwdControlSubscriber:
        """bind POST_OUT 订阅面;云经 master_addr 主动来连。"""
        return LwdControlSubscriber(
            config.lwd_post_out_bind_endpoint(),
            bind=True,
            decoder=lwd_decode_cloud_notify,
        )

    def _receive_thread(self, hello_event: threading.Event) -> None:
        """POST_OUT 接收线程体,按消息类型分发。

        HELLO -> 登记 engine 身份并建该 engine 的定向 PRE_OUT publisher:
        云端点唯一事实源,每 engine 首拍一次;重复 HELLO(同 engine_index)
        忽略(不支持对端重启自愈)。收齐 cloud_dp_size 个才置 hello_event
        (构造线程放行)。HELLO 携带 coordinator stats 地址时建
        LwdCoordClient(FIRST_REQ 唤醒 + stats 对账)。
        LwdC2eNotify(云->边唯一载荷)-> 载荷队列(阻塞 put,不可丢)
        + WAKEUP 唤醒主循环:prefill 全部完成后请求转入 awaiting,
        引擎无排程工作、阻塞在 input_queue.get(),载荷只进队列不会
        唤醒任何线程,必须向 input_queue 敲门;WAKEUP 原生语义即丢弃
        消息体,数据与唤醒分离,多投无害(空 drain 一步即返回)。
        其余帧(坏帧已被订阅层丢弃后仍不认识的类型)告警丢弃。
        """
        receiver = self._edge_receiver
        config = self._lwd_config
        while not receiver.closed:
            msg = receiver.recv(timeout_ms=5000)
            if msg is None:
                continue
            if isinstance(msg, LwdHelloNotify):
                eng = msg.engine_index
                endpoint = f"tcp://{msg.pre_out_host}:{msg.pre_out_port}"
                if eng in self._lwd_peer_hellos:
                    logger.warning(
                        "[Lwd] duplicate HELLO from cloud engine %d ignored", eng
                    )
                    continue
                logger.info(
                    "[Lwd] cloud engine %d discovered via HELLO: PRE_OUT -> %s "
                    "(%d/%d)",
                    eng,
                    endpoint,
                    len(self._lwd_peer_hellos) + 1,
                    self._lwd_expected_engines,
                )
                self._lwd_peer_hellos[eng] = msg
                self._edge_senders[eng] = LwdControlPublisher(
                    endpoint, bind=False, queue_max=config.publish_queue_max
                )
                if len(self._lwd_peer_hellos) > self._lwd_expected_engines:
                    # 配置错配:边侧 --cloud-dp-size 小于云侧实际 engine 数
                    # (世界已按更大的值建成,控制面不能悄悄少管一个 engine)
                    logger.error(
                        "[Lwd] more cloud HELLOs (%d) than expected "
                        "(--cloud-dp-size=%d); check edge launch config",
                        len(self._lwd_peer_hellos), self._lwd_expected_engines,
                    )
                if msg.coord_stats_addr and self._lwd_coord_client is None:
                    self._lwd_coord_client = LwdCoordClient(msg.coord_stats_addr)
                if len(self._lwd_peer_hellos) >= self._lwd_expected_engines:
                    hello_event.set()
            elif isinstance(msg, (LwdPrefixReadyNotify, LwdRefillNotify)):
                self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, msg))
            elif isinstance(msg, LwdC2eNotify):
                logger.info(
                    "[Lwd][edge-ctrl] C2eNotify reqs=%d down_seqno=%s",
                    len(getattr(msg, "req_ids", []) or []),
                    getattr(msg, "down_seqno", None),
                )
                # 队列元素带到达时间戳:消费滞后(c2e_wait=到达→派发)
                # 是"边消费不动 → 云 publisher 队满小睡"闭环的前置指标
                self.lwd_c2e_meta_queue.put((msg, time.monotonic()))
                self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            else:
                logger.warning("[Lwd] drop unexpected POST_OUT frame %r", type(msg))

    def _lwd_shutdown_planes(self) -> None:
        """两面关停(幂等):receiver 先关断输入,publisher 与 coord 收尾。"""
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None:
            control.shutdown()
        receiver = getattr(self, "_edge_receiver", None)
        if receiver is not None:
            receiver.shutdown()
        for sender in getattr(self, "_edge_senders", {}).values():
            sender.shutdown()
        coord = getattr(self, "_lwd_coord_client", None)
        if coord is not None:
            coord.shutdown()

    def _lwd_cross_check_hello(self) -> None:
        """逐 engine HELLO 调度配置互校(构造线程)。

        max_num_batched_tokens/max_num_seqs 是边侧批上限与云侧调度
        的同源输入(设计 §2.2/§5.2 的部署对齐前提),不一致即构造期
        fail-fast——不一致不会让链路报错,只会让云 preflight 恒 hold、
        prefill 被 decode 静默串行(真机实录:边 factor=1/云 factor=4
        时 chunk 8191 + decode 4 > 预算 8192,prefill 积压 280 步)。
        num_speculative_tokens 仅为 wire 兼容保留,边侧不再使用(host 暂存
        后边侧不再为云 decode 预留预算,spec 系数无需边云对齐)。多 engine
        (cloud_dp_size>1)要求 wire_version>=3(engine_index/coord 地址),
        否则端点表无法按身份建立。对端为旧版(数值全 0)时仅告警不拦截。"""
        hellos: dict[int, LwdHelloNotify] = getattr(
            self, "_lwd_peer_hellos", {}
        )
        if not hellos:
            return
        expected = set(range(self._lwd_expected_engines))
        if set(hellos) != expected:
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[LWD] cloud HELLO engine set mismatch: "
                f"expected={sorted(expected)}, got={sorted(hellos)}"
            )
        if self._lwd_expected_engines > 1:
            for eng, hello in hellos.items():
                if hello.wire_version < LWD_WIRE_VERSION:
                    self._lwd_shutdown_planes()
                    raise RuntimeError(
                        f"[LWD] cloud engine {eng} wire_version="
                        f"{hello.wire_version} < {LWD_WIRE_VERSION}: 多 DP "
                        f"部署要求 HELLO 携带 engine_index/coord_stats_addr,"
                        f"两侧必须同版本部署"
                    )
        for eng, hello in hellos.items():
            if (
                self.vllm_config.model_config.uses_mrope
                and hello.wire_version < LWD_WIRE_VERSION
            ):
                self._lwd_shutdown_planes()
                raise RuntimeError(
                    f"[LWD] edge/cloud wire protocol mismatch: model uses "
                    f"mrope but cloud engine {eng} HELLO wire_version="
                    f"{hello.wire_version} < {LWD_WIRE_VERSION}——云侧仍按数据面"
                    f" aux 帧协议运行,mrope positions 将静默缺失(msgspec 容忍"
                    f"未知字段,旧云不会报错),两侧必须同版本部署"
                )
            if hello.max_num_batched_tokens == 0:
                logger.warning(
                    "[Lwd] peer HELLO (engine %d) carries no scheduling triple "
                    "(legacy peer); edge/cloud config alignment UNCHECKED",
                    eng,
                )
                continue
            sched = self.vllm_config.scheduler_config
            local = (
                sched.max_num_batched_tokens,
                sched.max_num_seqs,
            )
            remote = (
                hello.max_num_batched_tokens,
                hello.max_num_seqs,
            )
            if local != remote:
                self._lwd_shutdown_planes()
                raise RuntimeError(
                    f"[LWD] edge/cloud scheduling config mismatch on engine "
                    f"{eng}: (max_num_batched_tokens, max_num_seqs) "
                    f"edge={local} != cloud={remote}; "
                    f"cap/preflight 公式两侧同源输入不一致会让 preflight 恒 "
                    f"hold、prefill 被 decode 静默串行——两侧拉起配置须同值"
                )
        logger.info(
            "[Lwd] HELLO scheduling config verified on %d engine(s)",
            len(hellos),
        )

    def _lwd_configure_prefix_cache(self) -> None:
        """逐 engine 互校 PC 能力,所有 engine 的哈希规则必须一致。"""
        hellos = self._lwd_peer_hellos
        capabilities = set()
        for eng, hello in hellos.items():
            if self._lwd_prefix_enabled != hello.prefix_caching:
                self._lwd_shutdown_planes()
                raise RuntimeError(
                    f"[LWD] prefix caching mismatch on engine {eng}: "
                    f"edge={self._lwd_prefix_enabled} "
                    f"cloud={hello.prefix_caching}"
                )
            if self._lwd_prefix_enabled and hello.wire_version < LWD_WIRE_VERSION:
                self._lwd_shutdown_planes()
                raise RuntimeError(
                    f"[LWD] cloud engine {eng} does not support prefix wire "
                    f"version {LWD_WIRE_VERSION}"
                )
            capabilities.add((
                hello.hash_block_size,
                hello.prefix_alignment_tokens,
                hello.prefix_alignment_required,
                hello.prefix_use_eagle,
            ))
        if self._lwd_prefix_control is not None:
            if len(capabilities) != 1:
                self._lwd_shutdown_planes()
                raise RuntimeError(
                    f"[LWD] inconsistent prefix capabilities across engines: "
                    f"{sorted(capabilities)}"
                )
            self._lwd_prefix_control.configure(next(iter(hellos.values())))

    # ------------------------------------------------------------------ #
    # 引擎接口覆写                                                        #
    # ------------------------------------------------------------------ #
    def add_request(self, request, _request_wave: int = 0) -> None:
        if self._lwd_prefix_control is not None and not request.abort_immediately:
            self._lwd_prefix_control.add_request(request)
        else:
            self.scheduler.lwd_edge_add_request(request)

    def _handle_client_request(self, request_type, request) -> None:
        control = self._lwd_prefix_control
        if control is not None and control.handle_client_request(
            request_type, request
        ):
            return
        if request_type == EngineCoreRequestType.WAKEUP and isinstance(
            request, LwdRefillNotify
        ):
            self._lwd_handle_refill(request)
            return
        super()._handle_client_request(request_type, request)

    def _lwd_handle_refill(self, notify: LwdRefillNotify) -> None:
        scheduler_output = self.scheduler.lwd_edge_build_refill(notify)
        if scheduler_output is None:
            logger.error(
                "[Lwd][edge-ctrl] refill req=%s cannot be served",
                notify.request_id,
            )
            self.abort_requests([notify.request_id])
            return
        self._lwd_pending_refills.append(scheduler_output)

    def abort_requests(self, request_ids: list[str]) -> None:
        """abort 信号先出云,再走原生本地清理。"""
        aborted = set(request_ids)
        self._lwd_pending_refills = deque(
            output for output in self._lwd_pending_refills
            if not aborted.intersection(output.lwd_batch.batch_meta.req_ids)
        )
        if self._lwd_prefix_control is not None:
            self._lwd_prefix_control.prepare_abort(request_ids)
        self.scheduler.lwd_edge_abort(request_ids)
        super().abort_requests(request_ids)

    def step(self):
        return self._lwd_edge_step()

    def step_with_batch_queue(self):
        return self._lwd_edge_step()

    def has_work(self) -> bool:
        """c2e 载荷队列与待收割批队列都计入工作:WAKEUP 只能打断阻塞的
        get,轮询循环是否退出由本判据决定(core.py while not has_work)。
        批队列非空必须计入——否则最后一批派发后引擎睡眠,最终 token
        永不收割交付。"""
        return (super().has_work() or not self.lwd_c2e_meta_queue.empty()
                or self._lwd_batch_queue or self._lwd_pending_refills)

    def _lwd_edge_step(self) -> tuple[dict[int, object] | None, bool]:
        """单步编排(批队列异步化+多批在飞,embed 优先):收割队首直至
        清空 -> prefill 连续派发(优先占深度配额) -> 消费云载荷(剩余
        配额派发 UNEMBED)-> 返回({0: 云侧结果输出} | None,prefill
        是否有派发)。

        embed/unembed 均派发入队不收割,队首全局 FIFO 收割(序与
        executor 排水序同构,不可交错);EMBED 优先派发,有回包排队时
        留一个名额给 UNEMBED。

        [Lwd][sched] 三类日志(饿死分析锚点):每批 harvest(含队列滞留
        wait)、每批 dispatch(含 ahead 队列构成/c2e 水位)、每步汇总。"""
        outputs: list = []
        finished_reqs: set = set()
        prefill_work = False
        h_emb = h_unemb = 0
        # UP send 完成水位:embed 批的执行结果只在 future 完成后才可读,
        # 而收割点只弹队不等待 —— 所以挂起逐步排空,不在弹队时读。
        # 水位是通道级(seqno 流 per-engine),在飞表按 (future, engine)
        # 存档,排空时逐 engine 取 max 分别回报。
        kept = []
        wm_by_engine: dict[int, int] = {}
        for fut, eng in self._lwd_embed_futures_inflight:
            if fut.done():
                wm = getattr(fut.result(), "lwd_up_send_watermark", -1)
                if wm is not None and wm >= 0:
                    wm_by_engine[eng] = max(wm_by_engine.get(eng, -1), wm)
            else:
                kept.append((fut, eng))
        self._lwd_embed_futures_inflight = kept
        for eng, wm in wm_by_engine.items():
            self.scheduler.lwd_edge_note_send_watermark(eng, wm)
        while self._lwd_batch_queue:
            kind, payload, t_dispatch, future = self._lwd_batch_queue.popleft()
            if kind == "unembed":
                self._lwd_deliver_unembed(
                    payload, future, outputs, finished_reqs
                )
                h_unemb += 1
            else:
                self.scheduler.lwd_edge_update_progress(
                    dict(payload.num_scheduled_tokens)
                )
                # UP send 完成水位捎带在 embed 批的执行结果里,但收割点
                # 只弹队不等待 —— future 挂到在飞表(带 engine),步首
                # 排空时再读(立刻读 = 永远读不到,闸门摘除停摆;真机实录)。
                if future is not None:
                    batch = getattr(payload, "lwd_batch", None)
                    meta = getattr(batch, "batch_meta", None)
                    self._lwd_embed_futures_inflight.append(
                        (future, int(getattr(meta, "engine_index", 0) or 0))
                    )
                prefill_work = True
                h_emb += 1
            wait_ms = (time.monotonic() - t_dispatch) * 1000
            if wait_ms > 1.0:
                logger.info(
                    "[Lwd][sched] edge harvest kind=%s seqno=%s wait=%.2fms",
                    kind, self._lwd_batch_seqno(kind, payload), wait_ms,
                )
        d_emb = 0
        while (
            self._lwd_pending_refills
            and self._lwd_can_dispatch_embed()
        ):
            refill_output = self._lwd_pending_refills.popleft()
            future = self.model_executor.execute_model(
                refill_output, non_block=True
            )
            self._lwd_batch_queue.append(
                ("embed", refill_output, time.monotonic(), future)
            )
            prefill_work = True
            d_emb += 1
        while self._lwd_can_dispatch_embed() and self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            if not scheduler_output.num_scheduled_tokens:
                break
            future = self._lwd_dispatch_embed(scheduler_output)
            if future is None:
                break
            self._lwd_batch_queue.append(
                ("embed", scheduler_output, time.monotonic(), future)
            )
            prefill_work = True
            d_emb += 1
        d_unemb = self._lwd_edge_consume_c2e()
        n_emb, n_unemb = self._lwd_queue_mix()
        if (prefill_work or outputs or h_emb or h_unemb
                or d_emb or d_unemb):
            logger.info(
                "[Lwd][sched] edge step harvest_emb=%d harvest_unemb=%d "
                "disp_emb=%d disp_unemb=%d queue=%demb/%dunemb c2e_pending=%d",
                h_emb, h_unemb, d_emb, d_unemb, n_emb, n_unemb,
                self.lwd_c2e_meta_queue.qsize(),
            )
        if outputs:
            step_outputs = EngineCoreOutputs(
                outputs=outputs,
                finished_requests=finished_reqs or None,
            )
            return {0: step_outputs}, prefill_work
        return None, prefill_work

    def _lwd_can_dispatch_embed(self) -> bool:
        remaining = LWD_EDGE_BATCH_QUEUE_DEPTH - len(self._lwd_batch_queue)
        # Recheck before taking the last slot: DOWN can arrive while an earlier
        # EMBED is being dispatched. Refills share the same dispatch limit.
        return remaining > 1 or (
            remaining == 1 and self.lwd_c2e_meta_queue.empty()
        )

    def _lwd_queue_mix(self) -> tuple[int, int]:
        """批队列构成 (embed数, unembed数):dispatch 日志的 ahead_unemb
        = EMBED 批前面压着的 UNEMBED 批数,是"EMBED 被 decode 回程挤住"
        (饿死假说)的直接读数。"""
        n_emb = sum(1 for e in self._lwd_batch_queue if e[0] == "embed")
        return n_emb, len(self._lwd_batch_queue) - n_emb

    @staticmethod
    def _lwd_batch_seqno(kind: str, payload) -> str:
        """harvest 日志的配对号:unembed 用 down_seqno(与云 DOWN 同号),
        embed 用批 seqno(与 RangeNotify/UP 同号)。"""
        if kind == "unembed":
            return str(getattr(payload, "down_seqno", "?"))
        batch = getattr(payload, "lwd_batch", None)
        return str(getattr(batch, "seqno", "?")) if batch else "?"

    def _lwd_edge_consume_c2e(self) -> int:
        """消费 c2e 通告:派发 UNEMBED 批入批队列(异步收割)。
        云侧有活请求才产 meta(build_hidden_payload 空则返回 None)。
        DISABLE_DOWN 诊断:通告只携带 token_ids(hidden=0)时跳过 worker
        派发,以 future=None 直通收割,不挂任何 recv。"""
        quota = LWD_EDGE_BATCH_QUEUE_DEPTH - len(self._lwd_batch_queue)
        notifies: list[tuple[LwdC2eNotify, float]] = []
        while len(notifies) < quota:
            try:
                notifies.append(self.lwd_c2e_meta_queue.get_nowait())
            except queue.Empty:
                break
        for notify, t_arrive in notifies:
            if _lwd_disable_down_enabled() and notify.token_ids:
                logger.info(
                    "[Lwd][sched] edge direct-deliver seqno=%s reqs=%d "
                    "(DISABLE_DOWN, no unembed dispatch)",
                    notify.down_seqno, len(notify.req_ids),
                )
                future = None
            else:
                future = self._lwd_dispatch_unembed(notify, t_arrive)
            self._lwd_batch_queue.append(
                ("unembed", notify, time.monotonic(), future)
            )
        return len(notifies)

    def _lwd_dispatch_embed(self, scheduler_output):
        """EMBED 批提交侧(唯一提交点,同步/异步共用):范围预告(发布
        成功才组批挂 lwd_batch)后以 non_block 提交,返回 future;
        发布失败返回 None,本轮不再提交。

        正确性前提:发布通道不丢消息。失败分支无回退可施——进度
        已按排程乐观推进,该 chunk 随 SO 废弃即永久丢失(进度虚高),
        正确性由通道不丢保证;同步/异步失败语义镜像(均为弃批)。"""
        if not self.scheduler.lwd_edge_notify(scheduler_output):
            return None
        batch = scheduler_output.lwd_batch
        n_emb, n_unemb = self._lwd_queue_mix()
        logger.info(
            "[Lwd][sched] edge dispatch-embed seqno=%d reqs=%s tokens=%d "
            "ahead_unemb=%d ahead_emb=%d c2e_pending=%d",
            batch.seqno, batch.batch_meta.req_ids,
            scheduler_output.total_num_scheduled_tokens,
            n_unemb, n_emb, self.lwd_c2e_meta_queue.qsize(),
        )
        return self.model_executor.execute_model(
            scheduler_output, non_block=True
        )

    def _lwd_dispatch_unembed(self, notify: LwdC2eNotify, t_arrive: float):
        """UNEMBED 批提交侧:组批并以 non_block 提交 worker,立即返回
        future(不等执行)。同步路径提交后立即收割;异步路径
        (batch_queue)将 (notify, future) 入队,收割阶段再等 future——
        提交与等待的边界即本函数返回处。

        c2e_wait(到达→派发)是消费滞后读数:持续偏大说明边引擎步
        循环被收割/派发占住,c2e 在积压,云侧 publisher 队满小睡在即。"""
        unembed_batch = lwd_build_unembed_batch(notify)
        n_emb, n_unemb = self._lwd_queue_mix()
        logger.info(
            "[Lwd][sched] edge dispatch-unembed seqno=%s reqs=%d rows=%d "
            "ahead_unemb=%d ahead_emb=%d c2e_wait=%.2fms c2e_pending=%d",
            notify.down_seqno, len(notify.req_ids),
            sum(notify.num_accepted_tokens), n_unemb, n_emb,
            (time.monotonic() - t_arrive) * 1000,
            self.lwd_c2e_meta_queue.qsize(),
        )
        return self.model_executor.execute_model(
            unembed_batch, non_block=True
        )

    @staticmethod
    def _lwd_row_disposition(
        accepted: int, has_tokens: bool, finished: bool
    ) -> str:
        """逐行交付裁定(pad-and-skip):

        - "skip":零接受活行(accepted=0 且未完结)——mid-prefill 非参与
          行,无内容无终态,通告里保留只为 DOWN 张量对位,不交付、
          不 ERROR、不标记首 token(修复前此形态被 ERROR 杀请求);
        - "error":声明 accepted>0 却无 token = unembed 真失败,ERROR
          优先于云侧完成码(原语义保留);
        - "deliver":其余(含零接受但带完结码的收尾行——终结信号必须
          透传,吞掉会丢完结)。"""
        if accepted == 0 and not finished:
            return "skip"
        if not has_tokens and not finished:
            return "error"
        return "deliver"

    def _lwd_deliver_unembed(
        self, notify: LwdC2eNotify, future, outputs: list, finished_reqs: set,
    ) -> None:
        """UNEMBED 批收割侧:等 worker 应答取采样 token 表,逐请求交付。
        不感知提交时机,只消费 (notify, future)。应答契约:
        token ids 按 ModelRunnerOutput 的 req_ids x sampled_token_ids
        按位对齐还原(批的 req_ids 原样下发,worker 逐位回填);行裁定
        走 _lwd_row_disposition(零接受活行跳过,accepted>0 缺 token 才
        ERROR);迟到载荷幂等丢弃。"""
        _t = time.monotonic()
        if future is None:
            # DISABLE_DOWN 直通:无 worker 批,token 直接取自 c2e 通告。
            sampled_token_map: dict[str, list[int]] = {
                rid: list(toks)
                for rid, toks in zip(notify.req_ids, notify.token_ids)
            }
        else:
            result = future.result()
            # UP send 完成水位随 unembed 结果捎带(decode 期的主要载具)
            wm = getattr(result, "lwd_up_send_watermark", -1)
            if result is not None and wm is not None and wm >= 0:
                # 水位按来源 engine 限定(通道级 seqno 流)
                self.scheduler.lwd_edge_note_send_watermark(
                    int(getattr(notify, "engine_index", 0) or 0), wm
                )
            watermarks = getattr(result, "lwd_up_send_watermarks", None) or {}
            for eng, engine_wm in watermarks.items():
                if engine_wm >= 0:
                    self.scheduler.lwd_edge_note_send_watermark(
                        eng, engine_wm
                    )
            # req_ids x sampled_token_ids 按位对齐:worker lm_head 恢复的采样
            # token,即该请求本步的生成内容;后续仅两处流向——
            # lwd_edge_deliver_tokens(调度器只对账 awaiting 生命周期,
            # 不消费内容)与 EngineCoreOutput 的 new_token_ids(outputs ->
            # EngineCoreOutputs -> 主循环按 frontend 消费 -> 前端解流交付
            # 客户端,即最终输出的生成 token)。
            sampled_token_map = (
                {} if result is None
                else dict(zip(result.req_ids, result.sampled_token_ids))
            )
        # [Lwd][perf] 临时探针:收割时长 = RPC 往返 + worker 执行全长
        # (与 worker 侧 [Lwd][perf] unembed 分段对账,差值即进程往返开销)
        logger.info(
            "[Lwd][perf] harvest dur=%.2fms", (time.monotonic() - _t) * 1000
        )
        # accepted 与行按位对齐(lwd_build_unembed_batch 断言保证);
        # DISABLE_DOWN 旁路不带 accepted,以回填 token 有无代替
        accepts = notify.num_accepted_tokens
        for index, request_id in enumerate(notify.req_ids):
            finish_reason = self._lwd_finish_code(notify, index)
            finished = finish_reason is not None
            sampled_token_ids = (
                list(sampled_token_map.get(request_id, []))
                if sampled_token_map else []
            )
            accepted = (
                accepts[index] if accepts
                else (1 if sampled_token_ids else 0)
            )
            disposition = self._lwd_row_disposition(
                accepted, bool(sampled_token_ids), finished
            )
            if disposition == "skip":
                # 零接受活行:不交付不出空输出,也无谓打调试税
                continue
            LwdDebug.edge_tokens_delivered(  # [lwd-debug]
                request_id, sampled_token_ids, finish_reason,
                self.vllm_config,
            )
            if disposition == "error":
                # 声明接受却无 token = unembed 失败,ERROR 优先于云侧码
                # (已完结的空行属正常,如 DISABLE_DOWN 下的 ABORT)
                finish_reason = FinishReason.ERROR
                finished = True
            if not self.scheduler.lwd_edge_deliver_tokens(
                request_id, sampled_token_ids, finished=finished
            ):
                logger.warning(
                    "[Lwd] drop stale cloud payload for %s (not awaiting)",
                    request_id,
                )
                continue
            if not sampled_token_ids and not finished:
                # 无内容且未完结:不出空输出
                continue
            outputs.append(
                EngineCoreOutput(
                    request_id, sampled_token_ids,
                    finish_reason=finish_reason,
                )
            )
            if finished:
                finished_reqs.add(request_id)

    @staticmethod
    def _lwd_finish_code(
        notify: LwdC2eNotify, index: int
    ) -> FinishReason | None:
        """逐请求完成码:哨兵 = 未完结(None);否则透传 FinishReason。
        与 req_ids 按位严格对齐,错配 IndexError fail-fast。"""
        code = notify.finish_reasons[index]
        return None if code == LWD_NOT_FINISHED else FinishReason(code)

    def shutdown(self) -> None:
        """两面关停后走原生;初始化失败路径两面可能未建,容忍缺省。"""
        self._lwd_shutdown_planes()
        super().shutdown()

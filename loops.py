import asyncio
import inspect
import logging
import multiprocessing as mp
import pickle
import sys
from queue import Empty, Queue
from typing import Dict, List, Optional, Tuple, Union

from fastapi import HTTPException

from .server import BubbleAPI
from .bubble_base import BubbleSpec
from .utils import BubbleAPIStatus

mp.allow_connection_pickling()

logger = logging.getLogger(__name__)

def _inject_context(context: Union[List[dict], dict], func, *args, **kwargs):
    sig = inspect.signature(func)
    if "context" in sig.parameters:
        return func(*args, **kwargs, context=context)
    return func(*args, **kwargs)

async def collate_requests(
    bubble_api: BubbleAPI, request_queue: Queue, max_batch_size: int, batch_timeout: float
) -> Tuple[List, List]:
    payloads = []
    timed_out_uids = []
    entered_at = asyncio.get_event_loop().time()
    end_time = entered_at + batch_timeout
    apply_timeout = bubble_api.request_timeout not in (-1, False)

    while asyncio.get_event_loop().time() < end_time and len(payloads) < max_batch_size:
        remaining_time = end_time - asyncio.get_event_loop().time()
        if remaining_time <= 0:
            break

        try:
            response_queue_id, uid, timestamp, x_enc = await asyncio.get_event_loop().run_in_executor(
                None, request_queue.get, True, min(remaining_time, 0.001)
            )
            if apply_timeout and asyncio.get_event_loop().time() - timestamp > bubble_api.request_timeout:
                timed_out_uids.append((response_queue_id, uid))
            else:
                payloads.append((response_queue_id, uid, x_enc))
        except Empty:
            continue

    return payloads, timed_out_uids

async def run_single_loop(bubble_api: BubbleAPI, bubble_spec: BubbleSpec, request_queue: Queue, response_queues: List[Queue]):
    while True:
        try:
            response_queue_id, uid, timestamp, x_enc = await asyncio.get_event_loop().run_in_executor(
                None, request_queue.get, True, 1.0
            )
        except Empty:
            await asyncio.sleep(0.01)
            continue

        if (bubble_api.request_timeout and bubble_api.request_timeout != -1) and (
            asyncio.get_event_loop().time() - timestamp > bubble_api.request_timeout
        ):
            logger.error(f"Request {uid} timed out.")
            response_queues[response_queue_id].put((uid, (HTTPException(504, "Request timed out"), BubbleAPIStatus.ERROR)))
            continue

        try:
            context = {}
            if hasattr(bubble_spec, "populate_context"):
                bubble_spec.populate_context(context, x_enc)
            x = _inject_context(context, bubble_api.decode_request, x_enc)
            y = await _inject_context(context, bubble_api.predict, x)
            y_enc = _inject_context(context, bubble_api.encode_response, y)
            response_queues[response_queue_id].put((uid, (y_enc, BubbleAPIStatus.OK)))
        except Exception as e:
            logger.exception(f"Error processing request uid={uid}")
            err_pkl = pickle.dumps(e)
            response_queues[response_queue_id].put((uid, (err_pkl, BubbleAPIStatus.ERROR)))

async def run_batched_loop(
    bubble_api: BubbleAPI,
    bubble_spec: BubbleSpec,
    request_queue: Queue,
    response_queues: List[Queue],
    max_batch_size: int,
    batch_timeout: float,
):
    while True:
        batches, timed_out_uids = await collate_requests(
            bubble_api,
            request_queue,
            max_batch_size,
            batch_timeout,
        )

        for response_queue_id, uid in timed_out_uids:
            logger.error(f"Request {uid} timed out.")
            response_queues[response_queue_id].put((uid, (HTTPException(504, "Request timed out"), BubbleAPIStatus.ERROR)))

        if not batches:
            await asyncio.sleep(0.01)
            continue

        response_queue_ids, uids, inputs = zip(*batches)
        try:
            contexts = [{}] * len(inputs)
            if hasattr(bubble_spec, "populate_context"):
                for input, context in zip(inputs, contexts):
                    bubble_spec.populate_context(context, input)

            x = [
                _inject_context(context, bubble_api.decode_request, input)
                for input, context in zip(inputs, contexts)
            ]
            x = bubble_api.batch(x)
            y = await _inject_context(contexts, bubble_api.predict, x)
            outputs = bubble_api.unbatch(y)
            for response_queue_id, y, uid, context in zip(response_queue_ids, outputs, uids, contexts):
                y_enc = _inject_context(context, bubble_api.encode_response, y)
                response_queues[response_queue_id].put((uid, (y_enc, BubbleAPIStatus.OK)))

        except Exception as e:
            logger.exception("Error processing batched request.")
            err_pkl = pickle.dumps(e)
            for response_queue_id, uid in zip(response_queue_ids, uids):
                response_queues[response_queue_id].put((uid, (err_pkl, BubbleAPIStatus.ERROR)))

async def run_streaming_loop(bubble_api: BubbleAPI, bubble_spec: BubbleSpec, request_queue: Queue, response_queues: List[Queue]):
    while True:
        try:
            response_queue_id, uid, timestamp, x_enc = await asyncio.get_event_loop().run_in_executor(
                None, request_queue.get, True, 1.0
            )
            logger.debug("uid=%s", uid)
        except Empty:
            await asyncio.sleep(0.01)
            continue

        if (bubble_api.request_timeout and bubble_api.request_timeout != -1) and (
            asyncio.get_event_loop().time() - timestamp > bubble_api.request_timeout
        ):
            logger.error(f"Request {uid} timed out.")
            response_queues[response_queue_id].put((uid, (HTTPException(504, "Request timed out"), BubbleAPIStatus.ERROR)))
            continue

        try:
            context = {}
            if hasattr(bubble_spec, "populate_context"):
                bubble_spec.populate_context(context, x_enc)
            x = _inject_context(context, bubble_api.decode_request, x_enc)
            y_gen = await _inject_context(context, bubble_api.predict, x)
            y_enc_gen = _inject_context(context, bubble_api.encode_response, y_gen)
            async for y_enc in y_enc_gen:
                y_enc = bubble_api.format_encoded_response(y_enc)
                response_queues[response_queue_id].put((uid, (y_enc, BubbleAPIStatus.OK)))
            response_queues[response_queue_id].put((uid, ("", BubbleAPIStatus.FINISH_STREAMING)))
        except Exception as e:
            logger.exception(f"Error processing streaming request uid={uid}")
            response_queues[response_queue_id].put((uid, (pickle.dumps(e), BubbleAPIStatus.ERROR)))

async def run_batched_streaming_loop(
    bubble_api: BubbleAPI,
    bubble_spec: BubbleSpec,
    request_queue: Queue,
    response_queues: List[Queue],
    max_batch_size: int,
    batch_timeout: float,
):
    while True:
        batches, timed_out_uids = await collate_requests(
            bubble_api,
            request_queue,
            max_batch_size,
            batch_timeout,
        )
        for response_queue_id, uid in timed_out_uids:
            logger.error(f"Request {uid} timed out.")
            response_queues[response_queue_id].put((uid, (HTTPException(504, "Request timed out"), BubbleAPIStatus.ERROR)))

        if not batches:
            await asyncio.sleep(0.01)
            continue

        response_queue_ids, uids, inputs = zip(*batches)
        try:
            contexts = [{}] * len(inputs)
            if hasattr(bubble_spec, "populate_context"):
                for input, context in zip(inputs, contexts):
                    bubble_spec.populate_context(context, input)

            x = [
                _inject_context(context, bubble_api.decode_request, input)
                for input, context in zip(inputs, contexts)
            ]
            x = bubble_api.batch(x)
            y_iter = await _inject_context(contexts, bubble_api.predict, x)
            unbatched_iter = bubble_api.unbatch(y_iter)
            y_enc_iter = _inject_context(contexts, bubble_api.encode_response, unbatched_iter)

            async for y_batch in y_enc_iter:
                for response_queue_id, y_enc, uid in zip(response_queue_ids, y_batch, uids):
                    y_enc = bubble_api.format_encoded_response(y_enc)
                    response_queues[response_queue_id].put((uid, (y_enc, BubbleAPIStatus.OK)))

            for response_queue_id, uid in zip(response_queue_ids, uids):
                response_queues[response_queue_id].put((uid, ("", BubbleAPIStatus.FINISH_STREAMING)))

        except Exception as e:
            logger.exception("Error processing streaming batched request.")
            err_pkl = pickle.dumps(e)
            for response_queue_id, uid in zip(response_queue_ids, uids):
                response_queues[response_queue_id].put((uid, (err_pkl, BubbleAPIStatus.ERROR)))

async def inference_worker(
    bubble_api: BubbleAPI,
    bubble_spec: Optional[BubbleSpec],
    device: str,
    worker_id: int,
    request_queue: Queue,
    response_queues: List[Queue],
    max_batch_size: int,
    batch_timeout: float,
    stream: bool,
    workers_setup_status: Dict[str, bool] = None,
):
    await bubble_api.setup(device)
    bubble_api.device = device

    print(f"Setup complete for worker {worker_id}.")

    if workers_setup_status:
        workers_setup_status[worker_id] = True

    if bubble_spec:
        logging.info(f"bubble_motor will use {bubble_spec.__class__.__name__} spec")

    if stream:
        if max_batch_size > 1:
            await run_batched_streaming_loop(bubble_api, bubble_spec, request_queue, response_queues, max_batch_size, batch_timeout)
        else:
            await run_streaming_loop(bubble_api, bubble_spec, request_queue, response_queues)
    elif max_batch_size > 1:
        await run_batched_loop(bubble_api, bubble_spec, request_queue, response_queues, max_batch_size, batch_timeout)
    else:
        await run_single_loop(bubble_api, bubble_spec, request_queue, response_queues)

# bubble-motor/server.py

import asyncio
import copy
import inspect
import logging
import multiprocessing as mp
import os
import json
import shutil
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from queue import Empty, Queue
from typing import Dict, List, Optional, Sequence, Tuple, Union
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.gzip import GZipMiddleware
from ariadne import QueryType, MutationType, make_executable_schema
from ariadne.asgi import GraphQL
import sys
import signal

from api import BubbleAPI
from auth import router as auth_router, get_authentication_dependency
from connector import _Connector
from example_openai_spec import OpenAISpec
from bubble_base import BubbleSpec
from utils import BubbleAPIStatus, MaxSizeMiddleware, load_and_raise

mp.allow_connection_pickling()

logger = logging.getLogger(__name__)

LONG_TIMEOUT = 100


# Define the GraphQL schema using Ariadne
type_defs = """
    type Query {
        get_result(request_id: String!): PredictionResult
    }

    type Mutation {
        predict(input_data: String!): PredictionResult
    }

    type PredictionResult {
        request_id: String
        status: String
        result: String
    }
"""

query = QueryType()
mutation = MutationType()


@query.field("get_result")
async def resolve_get_result(_, info, request_id):
    server = info.context["request"].app.state.bubble_server
    if request_id not in server.response_buffer:
        logger.warning(f"GraphQL: Result request for unknown request ID: {request_id}")
        return {"request_id": request_id, "status": "not_found", "result": None}
    response, status = server.response_buffer.get(request_id, (None, "processing"))
    if status == BubbleAPIStatus.ERROR:
        return {"request_id": request_id, "status": "error", "result": response.get("error", "Unknown error")}
    elif status == BubbleAPIStatus.OK:
        return {"request_id": request_id, "status": "completed", "result": response}
    else:
        return {"request_id": request_id, "status": "processing", "result": None}


@mutation.field("predict")
async def resolve_predict(_, info, input_data):
    server = info.context["request"].app.state.bubble_server
    request_id = str(uuid.uuid4())
    event = asyncio.Event()
    server.response_buffer[request_id] = (event, BubbleAPIStatus.PROCESSING)
    server.request_queue.put((server.response_queue_id, request_id, asyncio.get_event_loop().time(), {"input": input_data}))
    logger.info(f"GraphQL: Prediction request received. Request ID: {request_id}")
    return {"request_id": request_id, "status": "processing", "result": None}


# Create the executable schema
schema = make_executable_schema(type_defs, query, mutation)


def _inject_context(context: Union[List[dict], dict], func, *args, **kwargs):
    sig = inspect.signature(func)
    if "context" in sig.parameters:
        return func(*args, **kwargs, context=context)
    return func(*args, **kwargs)


def collate_requests(
        bubble_api: BubbleAPI, request_queue: Queue, max_batch_size: int, batch_timeout: float
) -> Tuple[List, List]:
    payloads = []
    timed_out_uids = []
    entered_at = time.monotonic()
    end_time = entered_at + batch_timeout
    apply_timeout = bubble_api.request_timeout not in (-1, False)

    while time.monotonic() < end_time and len(payloads) < max_batch_size:
        remaining_time = end_time - time.monotonic()
        if remaining_time <= 0:
            break

        try:
            response_queue_id, uid, timestamp, x_enc = request_queue.get(timeout=min(remaining_time, 0.001))
            if apply_timeout and time.monotonic() - timestamp > bubble_api.request_timeout:
                timed_out_uids.append((response_queue_id, uid))
            else:
                payloads.append((response_queue_id, uid, x_enc))

        except Empty:
            continue

    return payloads, timed_out_uids


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
    loop = asyncio.get_event_loop()
    await bubble_api.setup(device)
    bubble_api.device = device

    print(f"Setup complete for worker {worker_id}.")

    if workers_setup_status is not None:
        workers_setup_status[worker_id] = True

    if bubble_spec:
        logging.info(f"bubble_motor will use {bubble_spec.__class__.__name__} spec")

    while True:
        batches, timed_out_uids = collate_requests(
            bubble_api,
            request_queue,
            max_batch_size,
            batch_timeout,
        )

        for response_queue_id, uid in timed_out_uids:
            logger.error(f"Request {uid} timed out.")
            response_queues[response_queue_id].put(
                (uid, ({"error": "Request timed out."}, BubbleAPIStatus.ERROR))
            )

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
                _inject_context(
                    context,
                    bubble_api.decode_request,
                    input,
                )
                for input, context in zip(inputs, contexts)
            ]
            x = bubble_api.batch(x)
            y_gen = await _inject_context(contexts, bubble_api.predict, x)
            y_enc_gen = await _inject_context(contexts, bubble_api.encode_response, y_gen)

            for y_enc in y_enc_gen:
                response_queues[response_queue_id].put((uid, (y_enc, BubbleAPIStatus.OK)))

            for response_queue_id, uid in zip(response_queue_ids, uids):
                response_queues[response_queue_id].put((uid, ("", BubbleAPIStatus.FINISH_STREAMING)))

        except Exception as e:
            logger.exception("Error processing batched request.")
            error_response = {"error": str(e)}
            for response_queue_id, uid in zip(response_queue_ids, uids):
                response_queues[response_queue_id].put((uid, (error_response, BubbleAPIStatus.ERROR)))


class BubbleServer:
    def __init__(
            self,
            bubble_api: BubbleAPI,
            accelerator: str = "auto",
            devices: Union[str, int] = "auto",
            workers_per_device: int = 1,
            timeout: Union[float, bool] = 30,
            max_batch_size: int = 1,
            batch_timeout: float = 0.001,
            api_path: str = "/predict",
            stream: bool = False,
            spec: Optional[BubbleSpec] = None,
            max_payload_size: Optional[int] = None,
        ):
        if batch_timeout > timeout and timeout not in (False, -1):
            raise ValueError("batch_timeout must be less than timeout")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")
        if isinstance(spec, OpenAISpec):
            stream = True

        if not api_path.startswith("/"):
            raise ValueError("api_path must start with '/'.")

        self.api_path = api_path
        bubble_api.stream = stream
        bubble_api.request_timeout = timeout
        bubble_api._sanitize(max_batch_size, spec=spec)
        self.app = FastAPI(lifespan=self.lifespan)
        self.app.state.bubble_server = self
        self.response_queue_id = None
        self.response_buffer: Dict[str, Union[Tuple[deque, asyncio.Event, str], Dict]] = {}
        if not stream:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        if max_payload_size is not None:
            self.app.add_middleware(MaxSizeMiddleware, max_size=max_payload_size)
        self.bubble_api = bubble_api
        self.bubble_spec = spec
        self.workers_per_device = workers_per_device
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.stream = stream
        self._connector = _Connector(accelerator=accelerator, devices=devices)

        specs = spec if spec is not None else []
        self._specs = specs if isinstance(specs, Sequence) else [specs]

        # Ensure the request_type is correctly set from the bubble_api's decode_request signature
        decode_request_signature = inspect.signature(bubble_api.decode_request)
        self.request_type = decode_request_signature.parameters.get("request", Request).annotation
        if self.request_type == inspect.Parameter.empty:
            self.request_type = Request

        # Ensure the response_type is correctly set from the bubble_api's encode_response signature
        encode_response_signature = inspect.signature(bubble_api.encode_response)
        self.response_type = encode_response_signature.return_annotation
        if self.response_type == inspect.Signature.empty:
            self.response_type = Response

        accelerator = self._connector.accelerator
        devices = self._connector.devices
        if accelerator == "cpu":
            self.devices = [accelerator]
        elif accelerator in ["cuda", "mps"]:
            device_list = devices
            if isinstance(devices, int):
                device_list = range(devices)
            self.devices = [f"{accelerator}:{el}" for el in device_list]
        else:
            self.devices = ["cpu"]

        self.workers = self.devices * self.workers_per_device
        self.processes: List[mp.Process] = []
        self.manager: Optional[mp.Manager] = None

        # Call setup_server only after request_type is set
        self.setup_server()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        loop = asyncio.get_running_loop()

        if not hasattr(self, "response_queues") or not self.response_queues:
            raise RuntimeError("Response queues have not been initialized.")

        response_queue = self.response_queues[self.response_queue_id]
        response_executor = ThreadPoolExecutor(max_workers=len(self.devices) * self.workers_per_device)
        future = asyncio.create_task(self.response_queue_to_buffer(response_queue, self.response_buffer, self.stream, response_executor))

        try:
            yield
        finally:
            future.cancel()
            logger.debug("Shutting down response queue to buffer task")
            await asyncio.sleep(0.1)

    async def data_streamer(self, q: deque, data_available: asyncio.Event, send_status: bool = False):
        while True:
            await data_available.wait()
            while len(q) > 0:
                data, status = q.popleft()
                if status == BubbleAPIStatus.FINISH_STREAMING:
                    return

                if status == BubbleAPIStatus.ERROR:
                    logger.error("Error occurred while streaming outputs from the inference worker.")
                    if send_status:
                        yield data, status
                    return
                if send_status:
                    yield data, status
                else:
                    yield data
            data_available.clear()

    def setup_server(self):
        workers_ready = False

        # Include the OAuth2 token router
        self.app.include_router(auth_router)

        @self.app.get("/", dependencies=[Depends(get_authentication_dependency())])
        async def index(request: Request) -> Response:
            return Response(content="bubble_server running")

        @self.app.get("/health", dependencies=[Depends(get_authentication_dependency())])
        async def health(request: Request) -> Response:
            nonlocal workers_ready
            if not workers_ready:
                workers_ready = all(self.workers_setup_status.values())

            if workers_ready:
                return Response(content="ok", status_code=200)

            return Response(content="not ready", status_code=503)

        # Use request_type and response_type directly to avoid the attribute error
        async def predict(request: self.request_type,
                          background_tasks: BackgroundTasks) -> self.response_type:
            response_queue_id = self.app.state.bubble_server.response_queue_id
            uid = str(uuid.uuid4())
            event = asyncio.Event()
            self.response_buffer[uid] = (event, BubbleAPIStatus.PROCESSING)
            logger.info(f"Received request uid={uid}")

            payload = request
            if self.request_type == Request:
                if request.headers.get("Content-Type") in (
                    "application/x-www-form-urlencoded",
                    "multipart/form-data",
                ):
                    payload = await request.form()
                else:
                    payload = await request.json()

            self.request_queue.put_nowait((response_queue_id, uid, time.monotonic(), payload))

            await event.wait()
            response, status = self.response_buffer.pop(uid)

            if status == BubbleAPIStatus.ERROR:
                load_and_raise(response)
            return response

        async def stream_predict(request: self.request_type,
                                 background_tasks: BackgroundTasks) -> self.response_type:
            response_queue_id = self.app.state.bubble_server.response_queue_id
            uid = str(uuid.uuid4())
            event = asyncio.Event()
            q = deque()
            self.response_buffer[uid] = (q, event, BubbleAPIStatus.PROCESSING)
            logger.debug(f"Received request uid={uid}")

            payload = request
            if self.request_type == Request:
                payload = await request.json()
            self.request_queue.put((response_queue_id, uid, time.monotonic(), payload))

            return StreamingResponse(self.data_streamer(q, data_available=event))

        if not self._specs:
            stream = self.bubble_api.stream
            endpoint = self.api_path
            methods = ["POST"]
            self.app.add_api_route(
                endpoint,
                stream_predict if stream else predict,
                methods=methods,
                dependencies=[Depends(get_authentication_dependency())]
            )

        for spec in self._specs:
            spec: BubbleSpec
            for path, endpoint, methods in spec.endpoints:
                self.app.add_api_route(
                    path, endpoint=endpoint, methods=methods, dependencies=[Depends(get_authentication_dependency())]
                )

        # Setup GraphQL using Ariadne
        self.app.add_route("/graphql", GraphQL(schema=schema), methods=["GET", "POST"])

    async def launch_inference_worker(self, num_uvicorn_servers: int):
        self.manager = mp.Manager()
        self.workers_setup_status = self.manager.dict()
        self.request_queue = self.manager.Queue()

        self.response_queues = []
        for _ in range(num_uvicorn_servers):
            response_queue = self.manager.Queue()
            self.response_queues.append(response_queue)

        for spec in self._specs:
            server_copy = copy.copy(self)
            del server_copy.app
            try:
                await spec.setup(server_copy)
            except Exception as e:
                raise e

        process_list = []
        for worker_id, device in enumerate(self.devices * self.workers_per_device):
            if isinstance(device, list):
                device = device[0]  # Simplified device handling

            self.workers_setup_status[worker_id] = False

            ctx = mp.get_context("spawn")
            process = ctx.Process(
                target=self.inference_worker_process,
                args=(
                    self.bubble_api,
                    self.bubble_spec,
                    device,
                    worker_id,
                    self.request_queue,
                    self.response_queues,
                    self.max_batch_size,
                    self.batch_timeout,
                    self.stream,
                    self.workers_setup_status,
                ),
            )
            process.start()
            self.processes.append(process)
            process_list.append(process)
        return self.manager, process_list

    @staticmethod
    def inference_worker_process(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(inference_worker(*args, **kwargs))
        finally:
            loop.close()

    def generate_client_file(self):
        src_path = os.path.join(os.path.dirname(__file__), "python_client.py")
        dest_path = os.path.join(os.getcwd(), "client.py")

        if os.path.exists(dest_path):
            return

        try:
            shutil.copy(src_path, dest_path)
            print(f"File '{src_path}' copied to '{dest_path}'")
        except Exception as e:
            print(f"Error copying file: {e}")

    async def response_queue_to_buffer(
            self,
            response_queue: mp.Queue,
            response_buffer: Dict[str, Union[Tuple[deque, asyncio.Event, str], Dict]],
            stream: bool,
            threadpool: ThreadPoolExecutor,
    ):
        loop = asyncio.get_running_loop()
        if stream:
            while True:
                try:
                    uid, response = await loop.run_in_executor(threadpool, response_queue.get)
                except Empty:
                    await asyncio.sleep(0.0001)
                    continue
                stream_response_buffer, event, status = response_buffer[uid]
                stream_response_buffer.append((json.dumps(response), status))
                event.set()
        else:
            while True:
                try:
                    uid, response = await loop.run_in_executor(threadpool, response_queue.get)
                except Empty:
                    await asyncio.sleep(0.0001)
                    continue
                response_buffer[uid] = (response, BubbleAPIStatus.OK)
                event = response_buffer.get(uid)
                if isinstance(event, tuple):
                    _, event = event
                event.set()

    async def run(
            self,
            port: Union[str, int] = 8000,
            num_api_servers: Optional[int] = None,
            log_level: str = "info",
            generate_client_file: bool = True,
            api_server_worker_type: Optional[str] = None,
            **kwargs,
    ):
        if generate_client_file:
            self.generate_client_file()

        port_msg = f"port must be a value from 1024 to 65535 but got {port}"
        try:
            port = int(port)
        except ValueError:
            raise ValueError(port_msg)

        if not (1024 <= port <= 65535):
            raise ValueError(port_msg)

        if num_api_servers is None:
            num_api_servers = len(self.workers)

        self.manager, bubble_server_workers = await self.launch_inference_worker(num_api_servers)

        if sys.platform == "win32":
            api_server_worker_type = "thread"
        elif api_server_worker_type is None:
            api_server_worker_type = "process"

        try:
            servers = await self._start_server(port, num_api_servers, log_level, api_server_worker_type, **kwargs)
            # Register signal handlers for graceful shutdown
            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_event_loop().add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))
            await asyncio.gather(*[s.serve() for s in servers])
        finally:
            await self.shutdown()

    async def _start_server(self, port, num_uvicorn_servers, log_level, uvicorn_worker_type, **kwargs):
        servers = []
        for response_queue_id in range(num_uvicorn_servers):
            self.app.state.bubble_server.response_queue_id = response_queue_id
            if self.bubble_spec:
                self.bubble_spec.response_queue_id = response_queue_id
            app = copy.copy(self.app)

            config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level=log_level, **kwargs)
            server = uvicorn.Server(config=config)
            servers.append(server)
        return servers

    async def shutdown(self, signal=None):
        if signal:
            print(f"Received exit signal {signal.name}...")
        print("Shutting down bubble_server")
        for process in self.processes:
            process.terminate()
            process.join()
        if self.manager:
            self.manager.shutdown()
        print("Shutdown complete.")

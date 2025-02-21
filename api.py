from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel
import inspect
import json

from bubble_base import BubbleSpec


class BubbleAPI(ABC):
    _stream: bool = False
    _default_unbatch: callable = None
    _spec: Optional['BubbleSpec'] = None
    _device: Optional[str] = None
    request_timeout: Optional[float] = None

    @abstractmethod
    async def setup(self, device):
        """Set-up the model so it can be called in `predict`."""
        pass

    def decode_request(self, request, **kwargs):
        """Convert the request payload to your model input."""
        if self._spec:
            return self._spec.decode_request(request, **kwargs)
        return request

    def batch(self, inputs):
        """Convert a list of inputs to a batched input."""
        if hasattr(inputs[0], "__torch_function__"):
            import torch
            return torch.stack(inputs)
        if inputs[0].__class__.__name__ == "ndarray":
            import numpy
            return numpy.stack(inputs)
        return inputs

    @abstractmethod
    async def predict(self, x, **kwargs):
        """Run the model on the input and return or yield the output."""
        pass

    def _unbatch_no_stream(self, output):
        return list(output)

    def _unbatch_stream(self, output_stream):
        for output in output_stream:
            yield list(output)

    def unbatch(self, output):
        """Convert a batched output to a list of outputs."""
        return self._default_unbatch(output)

    @abstractmethod
    async def encode_response(self, output, **kwargs):
        """Convert the model output to a response payload."""
        pass
        # if self._spec:
         #   return self._spec.encode_response(output, **kwargs)
        # return output

    def format_encoded_response(self, data):
        if isinstance(data, dict):
            return json.dumps(data) + "\n"
        if isinstance(data, BaseModel):
            return data.model_dump_json() + "\n"
        return data

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def _sanitize(self, max_batch_size: int, spec: Optional['BubbleSpec']):
        if self.stream:
            self._default_unbatch = self._unbatch_stream
        else:
            self._default_unbatch = self._unbatch_no_stream

        if spec:
            self._spec = spec
            return

        original = self.unbatch.__code__ is BubbleAPI.unbatch.__code__
        if (
            self.stream
            and max_batch_size > 1
            and not all([
                inspect.isgeneratorfunction(self.predict),
                inspect.isgeneratorfunction(self.encode_response),
                (original or inspect.isgeneratorfunction(self.unbatch)),
            ])
        ):
            raise ValueError(
                """When `stream=True` with max_batch_size > 1, `bubble_api.predict`, `bubble_api.encode_response` and
                `bubble_api.unbatch` must generate values using `yield`."""
            )

        if self.stream and not all([
            inspect.isgeneratorfunction(self.predict),
            inspect.isgeneratorfunction(self.encode_response),
        ]):
            raise ValueError(
                """When `stream=True` both `bubble_api.predict` and
             `bubble_api.encode_response` must generate values using `yield`."""
            )

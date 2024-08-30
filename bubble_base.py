from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from .server import BubbleServer


class BubbleSpec:
    """Spec will have its own encode, and decode."""

    def __init__(self):
        self._endpoints = []

        self._server: "BubbleServer" = None

    def setup(self, server: "BubbleServer"):
        self._server = server

    def add_endpoint(self, path: str, endpoint: Callable, methods: List[str]):
        """Register an endpoint in the spec."""
        self._endpoints.append((path, endpoint, methods))

    @property
    def endpoints(self):
        return self._endpoints.copy()

    @abstractmethod
    def decode_request(self, request, meta_kwargs):
        """Convert the request payload to your model input."""
        pass

    @abstractmethod
    def encode_response(self, output, meta_kwargs):
        """Convert the model output to a response payload.

        To enable streaming, it should yield the output.

        """
        pass

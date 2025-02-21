
import asyncio
from typing import Any, Union
from server import BubbleAPI, BubbleServer
from example_openai_spec import OpenAISpec
from bubble_base import BubbleSpec


class MyBubbleAPI(BubbleAPI):
    """
    Example implementation of BubbleAPI.
    Implements setup, predict, and encode_response methods.
    """
    async def setup(self, device: str):
        print(f"Setting up MyBubbleAPI on device: {device}")
        # Load your AI model here. For example:
        # self.model = load_model(device)

    async def predict(self, x: Any, **kwargs) -> Any:
        """
        Simulate a streaming prediction.
        Replace with actual inference logic.
        """
        # Example streaming prediction
        yield f"MyBubbleAPI Prediction part 1 for input: {x['input']}"
        await asyncio.sleep(0.5)  # Simulate processing delay
        yield f"MyBubbleAPI Prediction part 2 for input: {x['input']}"

    async def encode_response(self, output: Any, **kwargs) -> Union[Any, Any]:
        """
        Encapsulate the prediction output into ChatMessage.
        """
        from example_openai_spec import ChatMessage
        return ChatMessage(role="assistant", content=output)


if __name__ == "__main__":
    # Initialize Bubble Spec (e.g., OpenAI Spec)
    openai_spec = OpenAISpec()

    # Initialize Bubble API
    bubble_api = MyBubbleAPI()

    # Initialize Bubble Server with the spec
    server = BubbleServer(
        bubble_api=bubble_api,
        spec=openai_spec,
        api_path="/v1/chat/completions",
        stream=True,  # Enable streaming if desired
        max_batch_size=10,  # Adjust as needed
        batch_timeout=0.05,  # Adjust as needed
        timeout=30,  # Request timeout in seconds
        max_payload_size=10 * 1024 * 1024,  # 10 MB limit
        accelerator="auto",  # Choose accelerator (cpu, cuda, mps, auto)
        devices="auto",  # Number of devices
    )

    # Run the server
    asyncio.run(server.run(port=8000))

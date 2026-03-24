
from typing import Optional
from langfuse.langchain import CallbackHandler
import os
from langfuse import Langfuse


def build_langfuse_handler() -> Optional[CallbackHandler]:
    """Create a Langfuse callback handler when tracing credentials exist.

    Input:
        Reads Langfuse public key, secret key, and host from environment
        variables.
    Output:
        Returns a configured `CallbackHandler` or `None` when tracing is not
        configured.
    Security context:
        Prevents accidental tracing setup with partial credentials and keeps all
        Langfuse secrets environment-scoped.
    """

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    
    if not public_key or not secret_key or not host:
        return None
    
    return CallbackHandler(
        public_key=public_key,
        trace_context={"trace_id": Langfuse.create_trace_id(seed="42")}
    )


from typing import Optional
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse


def build_langfuse_handler(public_key: str, trace_id: str) -> Optional[CallbackHandler]:
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
    return CallbackHandler(
        public_key=public_key,
        trace_context={"trace_id": trace_id or Langfuse.create_trace_id()}
    )

"""Llama, Mistral, Qwen2 and Qwen3 dense decoders: the config-driven reference lives in ikengine/ref_dense.py.
Importing it here registers those architectures when this package discovers its modules."""
from .. import ref_dense  # noqa: F401

"""Register the cache connector in spawned vLLM workers."""

import os

if os.environ.get("CACHEREPAIR_VLLM") == "1":
    from cacherepair.connector import register

    register()

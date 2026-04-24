# Real LLM tests

These scripts call the real LLM API through the OpenAI SDK.  The default runtime
uses your MCP/OpenAI-compatible environment variables:

```bash
export MCP_API_BASE="https://your-openai-compatible-endpoint/v1"
export MCP_API_KEY="..."
export MCP_MODEL="..."
python run_all_tests.py --require-real
```

`OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL` remain compatibility
fallbacks, but `MCP_*` is the preferred configuration path.

The tests are intentionally executable scripts rather than pytest cases.  They
exercise real agent loops, tool calls, file patching, code execution, context
management, memory writing/recall, and an end-to-end repair task.

# Real API scenario runner

This folder replaces the previous scattered pytest/smoke files. It runs the agent as a real API LLM agent and uses assertive scenario checks in normal Python, not pytest and not a mocked LLM.

```bash
export MCP_API_KEY="..."      # or OPENAI_API_KEY
export MCP_MODEL="gpt-4o-mini" # or OPENAI_MODEL
export MCP_API_BASE="https://api.openai.com/v1"
python real_scenarios/run_research_agent_scenarios.py --reset --scenario all
```

Scenario coverage:

1. cross-thread long-term memory recall
2. multi-step tool loop: write_note then read_note
3. tool governance: search and inspect tool descriptors
4. large observation compression through huge_log
5. skill loading and skill-local tool use
6. structured JSON answer using remembered preferences

The runner exits non-zero on failed behavior. It intentionally refuses to run without real API credentials.

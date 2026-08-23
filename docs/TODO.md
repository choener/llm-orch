- [ ] `/v1/info` endpoint should return utilization of each card
- [x] Allow aliases to select between multiple models (design + implementation:
      `docs/003-smart-handling.md`) — ordered `targets` list, `policy: prefer_loaded |
      prefer_order`, `make_room: none | evict_idle | drain_surplus`.
    - Follow-up (not implemented): a global make-room default for *direct* (non-alias)
      requests — doc 002's "load `9` over `2`" scenario.
- [x] Make duplicate unloading work (`docs/003-smart-handling.md`, make-room victim class 3:
      `make_room: drain_surplus` on an alias drains the less-busy duplicate, bounded by
      `drain_timeout`, then loads the preferred model).
- [ ] delay unloading the last model of a type, when "lazy-unload: true" is set. Consider Deepseek
  V4. That model takes a while to load, but also blocks all other models. Keep it loaded, unless a
  request to load another model comes in, once ttl and others would normally trigger unload. We
  don't want this everywhere, since it also prevents deep sleep of the gpu's, which I want on
  cheyenne
- [ ] Need to investigate the interplay between pi, llm-orch, llama-server and deepseek-v4-flash.
  Sometimes, a request just hangs. (a) pi should wait longer, (b) llm-orch should then keep-alive
  ping, (c) llm-orch should check whether llama-server died on a request and restart and re-issue
  the request
- [ ] Need to consider how to create a benchmark for firecrawl testing different models,
  qwen3.8-27b, qwen-moe, gemma-4-moe, gemma-4-12b, etc. Should have "raw" example data and a
  "simulation" of what the firecrawl service will do with the data.

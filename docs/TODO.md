- [ ] `/v1/info` endpoint should return utilization of each card
- [ ] Allow aliases to select between multiple models, in the following order:
    - take the first loaded model with a free slot
    - load the first unloaded model that actually fits
    - take the first loaded model and wait
    - make it possible to set the desired behaviour (first loaded vs. first fitting)
    - Needs some thinking, but here is the use case: I can fit one dense and one MoE model, or two
      of each. Sometime two dense models are loaded and, say, firecrawl wants to summarise with the
      MoE. Now not possible. firecrawl then should load the MoE.
- [ ] Make duplicate unloading work. I currently get errors when Qwen3.8 27B is loaded twice, and
  the MoE want to load. This should make the MoE wait, drain the less busy 27B and then load the MoE
- [ ] delay unloading the last model of a type, when "lazy-unload: true" is set. Consider Deepseek
  V4. That model takes a while to load, but also blocks all other models. Keep it loaded, unless a
  request to load another model comes in, once ttl and others would normally trigger unload. We
  don't want this everywhere, since it also prevents deep sleep of the gpu's, which I want on
  cheyenne

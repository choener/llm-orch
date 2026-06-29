- post-load action: run after the model has started loading. To be used to reduce voltage to the
  gpu.
- on switching gpu affinity, use slots/<id>/save and restore to save the kv-cache
- I want to be able to log-to-disk queries and answers for some models. I.e. be able to define a
  log-level per model.
- It would also be good to have an endpoint where recent query information is return via json.
  Similar to llama-swap's information. Including cached tokens, etc.

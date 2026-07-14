# GPU selection under load.

- Assume LLMs "1" and "2", while "9" is really large (i.e. 9 uses 99% gpu vram, whle 1 uses 11%)
- Also assume two GPUs "A" and "B".

Both GPUs are idle, and a request for "1" comes in.
- Load "1" onto "A" or whereever is more vram free.
- "1" will answer requests, and be unloaded once TTL time has passed without requests.

When should we try to load a copy of "1" onto "B"?
- "waiting > max_concurrent" ?
- load on "B" is small enough ?

I am tempted to go with something like this:
- "waiting > max_concurrent"
- another GPU has the free vram

# Now, what to do if the gpu's are "full"?
- "1" is on "A"
- "2" is on "B"
- "9" is requested
- find the gpu with least gpu load, say "B"
- don't enqueue more requests from the queue onto "B"
- once "2" has answered all outstanding requests
- evict "2" from "B"
- load "9" onto "B"
- start answering requests for "9"
- ("2" will only be loaded onto "A", if there are requests outstanding for "2")

How to handle priority for unloading?
Need function?

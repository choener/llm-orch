# TODOs

## Loading animation

- While a model loads, send periodic dots back to the llm so that it knows work happens.
- Does this end up in the history of messages of the LLM or can be removed once done?

## Summary

The periodic "." messages (or similar progress indicators) sent from llama-server are not meant to
go into the model's context and are handled as server-side operational messages, not as part of the
actual conversation or completion.

Here's what I found from examining the llama.cpp server documentation and discussions:

### Key Points:

1. Server-Sent Events (SSE) Protocol: The llama-server uses SSE for streaming responses. Progress
   indicators like "." are typically sent within the SSE stream as control or status messages, not
   as actual model-generated tokens.

2. Separate Control Channel: The server distinguishes between:
    - Model-generated content: Actual tokens from the LLM that go into the conversation context
    - Server metadata/control messages: Status updates, progress indicators, and operational
      feedback that are handled by the server/harness

3. Built-in Tools Context: From the discussion about --tools all, we can see that llama-server
   handles tool calls and responses at the server level, separate from the model's reasoning. The
   system manages the conversation flow without injecting operational messages into the model
   context.

4. SSE Ping Mechanism: The server has a documented --sse-ping-interval parameter specifically for
   sending keep-alive signals during long operations without affecting the model's context. These
   pings are purely for connection maintenance.

### Answer to Your Question:

If you're implementing a coding harness that receives periodic "." messages from llama-server:

- These messages do NOT end up in the model's context
- They can be (and should be) handled as special server messages that your harness uses for:
    - Progress indication to the user
    - Connection keep-alive
    - Operational status tracking

Your harness should filter out these dots and treat them as metadata/control signals rather than
injecting them into the conversation history sent to the model. The server maintains a clear
separation between actual completion tokens and operational/status messages in its SSE protocol.

To be even more precise, in the SSE stream, actual model tokens are sent as JSON payloads with
content fields, while progress indicators like "." would be sent as comment lines or special event
types that your client can distinguish and handle separately.

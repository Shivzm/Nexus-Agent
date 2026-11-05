# Issue faced and solution

## Groq request rejected after a long conversation

### What happened

A chat request failed with a Groq 413 response: the request contained 8,105 tokens while the applicable token-per-minute limit was 8,000. The question itself was not the problem. The ReAct graph resent the conversation history on each model call, and earlier document-search passages and web-search results accumulated across turns.

### How we traced it

1. Checked the provider error and compared the reported request size with the configured model's limit.
2. Followed the ReAct model call and confirmed it passed the full `messages` history back to the model on every turn.
3. Identified earlier tool messages as the growing part of the history: retrieved document chunks and web results were useful for the turn that requested them, but were repeatedly sent on later turns.
4. Kept the user's questions and the assistant's replies in history, since those provide useful conversational context, and targeted only old tool output for compaction.

### Solution implemented

Before each ReAct model call, `compact_history` replaces tool messages from completed turns with the short placeholder `[earlier tool output omitted]`. It preserves the message's tool-call ID and tool name, and leaves the current turn's tool output and the rest of the conversation intact. This reduces repeated prompt content without changing retrieval, web search, or the user's visible conversation.

### Why this approach

It addresses the source of the growth rather than shortening the user's question or arbitrarily truncating the entire conversation. Older tool results have already been incorporated into the assistant's earlier answer; keeping those answers while omitting repeated raw passages avoids resending bulky evidence on every subsequent turn.

### Where to find it

- The compaction helper and its use before the ReAct model invocation are in [`src/agents/react_agent.py`](./src/agents/react_agent.py).
- The ReAct graph behavior is covered by the tests in [`Tests/test_react_agent.py`](./Tests/test_react_agent.py). There is not currently a dedicated test asserting the contents returned by `compact_history`.

# Human-in-the-Loop

Sometimes you want a human to review or modify the agent's work before it continues. Flowgentra supports this with **interrupt points** — the graph pauses, saves its state, and waits for you to resume it.

This requires **checkpointing** to be enabled so the state is persisted between the interrupt and the resume.

---

## Basic setup

=== "Python"

    ```python
    from flowgentra_ai.graph import StateGraph, END
    from flowgentra_ai import State

    def draft(state):
        state["draft"] = f"Draft article about {state['topic']}"
        return state

    def publish(state):
        state["status"] = "published"
        return state

    builder = StateGraph(dict)
    builder.add_node("draft",   draft)
    builder.add_node("publish", publish)
    builder.set_entry_point("draft")
    builder.add_edge("draft",   "publish")
    builder.add_edge("publish", END)

    # Pause BEFORE "publish" runs — human reviews first
    builder.interrupt_before("publish")

    # Checkpointing is required to persist state across the pause
    builder.set_checkpointer("./checkpoints")

    graph = builder.compile()
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::{StateGraph, DynState};
    use flowgentra_ai::memory::FileCheckpointer;

    let graph = StateGraph::builder()
        .add_node("draft",   draft_fn)
        .add_node("publish", publish_fn)
        .entry("draft")
        .edge("draft",   "publish")
        .edge("publish", "__end__")
        .interrupt_before("publish")
        .with_checkpointer(FileCheckpointer::new("./checkpoints"))
        .build();
    ```

---

## Interrupt and resume

=== "Python"

    ```python
    # First run — executes "draft", then pauses before "publish"
    result = graph.invoke_with_thread("thread-1", State({"topic": "AI"}))

    print(result["draft"])   # "Draft article about AI"
    # Graph is now paused. The draft is in state.

    # Human reviews the draft here...

    # Resume — continues from where it stopped (runs "publish")
    result = graph.resume("thread-1")
    print(result["status"])  # "published"
    ```

=== "Rust"

    ```rust
    let result = graph.invoke_with_thread("thread-1", initial_state).await?;
    println!("{}", result.get_string("draft").unwrap());

    // ... human reviews ...

    let result = graph.resume("thread-1").await?;
    println!("{}", result.get_string("status").unwrap());
    ```

---

## Resume with human edits

Before resuming, you can inject changes into the state:

=== "Python"

    ```python
    # First run — pauses before publish
    result = graph.invoke_with_thread("thread-1", State({"topic": "AI"}))

    # Human edits the draft
    human_edit = State({"draft": "Improved draft with better structure and examples"})

    # Resume with the edited state merged in
    result = graph.resume_with_state("thread-1", human_edit)
    print(result["status"])  # "published" — with the edited draft
    ```

=== "Rust"

    ```rust
    let mut edits = DynState::new();
    edits.set("draft", "Improved draft with better structure");
    let result = graph.resume_with_state("thread-1", edits).await?;
    ```

---

## Interrupt after a node

Interrupt **after** a node runs instead of before. The node executes, then the graph pauses. Useful when you want to review the output before the next step uses it.

=== "Python"

    ```python
    builder.interrupt_after("draft")   # draft runs, then pauses
    ```

---

## Multiple interrupt points

You can have several pauses in one graph. Each `resume()` call advances to the next one.

=== "Python"

    ```python
    builder.interrupt_before("review")
    builder.interrupt_before("publish")

    # First run → pauses at "review"
    result = graph.invoke_with_thread("thread-1", initial_state)

    # First resume → runs "review", then pauses at "publish"
    result = graph.resume("thread-1")

    # Second resume → runs "publish", then finishes
    result = graph.resume("thread-1")
    ```

---

## Full human-in-the-loop example

A content pipeline where humans approve both the outline and the final draft:

=== "Python"

    ```python
    from flowgentra_ai.graph import StateGraph, END
    from flowgentra_ai.llm import LLM, Message
    from flowgentra_ai import State

    client = LLM(provider="openai", model="gpt-4o", api_key="sk-...")

    def create_outline(state):
        response = client.chat([
            Message.system("Create a structured outline for an article."),
            Message.user(f"Topic: {state['topic']}"),
        ])
        state["outline"] = response.content
        return state

    def write_draft(state):
        response = client.chat([
            Message.system("Write a full article based on this outline."),
            Message.user(state["outline"]),
        ])
        state["draft"] = response.content
        return state

    def publish(state):
        state["status"] = "published"
        # In real life: post to CMS, send email, etc.
        return state

    builder = StateGraph(dict)
    builder.add_node("outline", create_outline)
    builder.add_node("draft",   write_draft)
    builder.add_node("publish", publish)
    builder.set_entry_point("outline")
    builder.add_edge("outline", "draft")
    builder.add_edge("draft",   "publish")
    builder.add_edge("publish", END)

    builder.interrupt_after("outline")   # human approves outline
    builder.interrupt_after("draft")     # human approves draft

    builder.set_checkpointer("./checkpoints")
    graph = builder.compile()

    # Step 1: Generate outline
    result = graph.invoke_with_thread("article-1", State({"topic": "Rust for AI"}))
    print("OUTLINE:\n", result["outline"])

    # Human reviews and may edit the outline
    approved_outline = result["outline"] + "\n\n[Editor note: Add more examples]"
    result = graph.resume_with_state("article-1", State({"outline": approved_outline}))
    print("DRAFT:\n", result["draft"])

    # Human approves draft
    result = graph.resume("article-1")
    print("Status:", result["status"])   # "published"
    ```

---

## Tips

- Always use `invoke_with_thread` (not `invoke`) when using interrupts. Without a thread ID, state can't be persisted.
- The checkpoint directory accumulates files over time. Clean it up periodically for long-running deployments.
- If a human takes too long to resume, the state just waits — there's no TTL by default.
- You can build a simple web UI that calls `resume()` or `resume_with_state()` when the human clicks "Approve".

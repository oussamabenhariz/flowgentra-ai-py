"""MockLLM — scripted, offline LLM for tests. No network, no credentials."""

from flowgentra_ai.llm import Message, MockLLM


def test_always_returns_fixed_reply():
    llm = MockLLM.always("hello").as_llm()
    assert llm.chat([Message.user("hi")]).content == "hello"
    assert llm.chat([Message.user("anything")]).content == "hello"


def test_when_contains_and_otherwise():
    mock = MockLLM()
    mock.when_contains("weather", "It is sunny")
    mock.otherwise("I don't know")
    llm = mock.as_llm()

    assert llm.chat([Message.user("what's the weather?")]).content == "It is sunny"
    assert llm.chat([Message.user("random question")]).content == "I don't know"
    assert mock.call_count() == 2


def test_sequence_repeats_last_once_exhausted():
    llm = MockLLM.sequence(["step 1", "step 2"]).as_llm()
    assert llm.chat([Message.user("x")]).content == "step 1"
    assert llm.chat([Message.user("x")]).content == "step 2"
    assert llm.chat([Message.user("x")]).content == "step 2"  # clamps to last


def test_with_usage_reports_token_counts():
    mock = MockLLM.always("one two three")
    mock.with_usage()
    llm = mock.as_llm()
    _, usage = llm.chat_with_usage([Message.user("hi there")])
    assert usage is not None
    assert usage.completion_tokens == 3


def test_chat_stream_emits_reply():
    llm = MockLLM.always("a b c").as_llm()
    chunks = list(llm.chat_stream([Message.user("x")]))
    assert "".join(chunks) == "a b c"


def test_as_llm_drop_in_for_chat_with_tools():
    """A MockLLM should work anywhere a real LLM does, since it just falls
    back to chat() when tools are passed (default trait behavior)."""
    llm = MockLLM.always("no tool needed").as_llm()
    result = llm.chat_with_tools([Message.user("hi")], [])
    assert result.content == "no tool needed"

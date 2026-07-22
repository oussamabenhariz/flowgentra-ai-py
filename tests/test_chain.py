"""Chain composition: `prompt | llm | parser` and the explicit
Chain.sequence([...]) alternative, plus the two-stage flowgentra_ai.llm.Chain."""

from flowgentra_ai.chain import Chain
from flowgentra_ai.llm import Chain as LLMChain
from flowgentra_ai.llm import JsonOutputParser, ListOutputParser, MockLLM, PromptTemplate


def test_pipe_operator_prompt_then_llm():
    prompt = PromptTemplate("Translate '{text}' to French.")
    llm = MockLLM.always("Bonjour").as_llm()

    pipeline = prompt | llm
    result = pipeline.invoke({"text": "Hello"})

    assert result.content == "Bonjour"


def test_pipe_operator_with_json_parser():
    prompt = PromptTemplate("List colors related to {topic}")
    llm = MockLLM.always('["red", "blue"]').as_llm()

    pipeline = prompt | llm | JsonOutputParser()
    result = pipeline.invoke({"topic": "ocean"})

    assert result == ["red", "blue"]


def test_pipe_operator_with_list_parser():
    prompt = PromptTemplate("List fruits")
    llm = MockLLM.always("apple, banana, cherry").as_llm()

    pipeline = prompt | llm | ListOutputParser()
    result = pipeline.invoke({})

    assert result == ["apple", "banana", "cherry"]


def test_chain_sequence_matches_pipe_operator():
    prompt = PromptTemplate("List colors related to {topic}")
    llm = MockLLM.always('["red", "blue"]').as_llm()

    via_pipe = (prompt | llm | JsonOutputParser()).invoke({"topic": "ocean"})
    via_sequence = Chain.sequence([prompt, llm, JsonOutputParser()]).invoke({"topic": "ocean"})

    assert via_pipe == via_sequence == ["red", "blue"]


def test_plain_function_stage():
    prompt = PromptTemplate("Echo {text}")
    llm = MockLLM.always("bonjour").as_llm()

    pipeline = prompt | llm | (lambda msg: msg.content.upper())
    result = pipeline.invoke({"text": "hi"})

    assert result == "BONJOUR"


def test_chain_repr_and_extend_with_or():
    prompt = PromptTemplate("x {a}")
    llm = MockLLM.always("y").as_llm()
    chain = prompt | llm
    extended = chain | (lambda m: m.content + "!")

    assert extended.invoke({"a": "1"}) == "y!"
    assert "Chain" in repr(chain)


def test_llm_chain_two_stage_helper():
    """flowgentra_ai.llm.Chain — the Rust-bound, fixed prompt->LLM helper."""
    prompt = PromptTemplate("Translate '{text}' to French.")
    llm = MockLLM.always("Bonjour").as_llm()

    chain = LLMChain(prompt, llm)
    result = chain.invoke({"text": "Hello"})

    assert result.content == "Bonjour"


def test_llm_chain_invoke_structured():
    prompt = PromptTemplate("List colors related to {topic}")
    llm = MockLLM.always('["red", "blue"]').as_llm()

    chain = LLMChain(prompt, llm)
    result = chain.invoke_structured({"topic": "ocean"})

    assert result == ["red", "blue"]

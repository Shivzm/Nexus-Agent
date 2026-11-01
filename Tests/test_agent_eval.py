import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.run_agent_eval import (
    first_selected_tool,
    load_cases,
    parse_success_judgment,
    run_agent_evaluation,
    save_results,
    summarize_results,
)


class FakeGraph:
    def invoke(self, inputs, config):
        question = inputs["messages"][0]["content"]
        return {
            "messages": [
                HumanMessage(content=question),
                AIMessage(
                    content="The result is 6.",
                    tool_calls=[{"name": "multiply", "args": {"a": 2, "b": 3}, "id": "call-1"}],
                ),
                ToolMessage(content="6", tool_call_id="call-1", name="multiply"),
                AIMessage(content="The result is 6."),
            ]
        }


class FakeJudge:
    def invoke(self, prompt):
        assert "The result is 6." in prompt

        class Response:
            content = json.dumps({"success": True, "explanation": "Correct product."})

        return Response()


def case():
    return {
        "id": "math-example",
        "category": "math",
        "question": "What is 2 times 3?",
        "expected_tool": "multiply",
        "success_criteria": "Answer is 6.",
    }


def test_load_cases_checks_category_tool_consistency(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case()) + "\n", encoding="utf-8")

    assert load_cases(path) == [case()]

    path.write_text(json.dumps({**case(), "expected_tool": "search_web"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expects tool"):
        load_cases(path)


def test_first_selected_tool_uses_first_model_tool_call():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_documents", "args": {"query": "q"}, "id": "1"},
                {"name": "search_web", "args": {"query": "q"}, "id": "2"},
            ],
        )
    ]

    assert first_selected_tool(messages) == "search_documents"
    assert first_selected_tool([]) is None


def test_success_judge_parser_validates_json_contract():
    assert parse_success_judgment('{"success":true,"explanation":"ok"}') == (True, "ok")
    with pytest.raises(ValueError, match="boolean 'success'"):
        parse_success_judgment('{"success":"yes"}')


def test_run_measures_tool_choice_and_task_success_separately():
    details, summary = run_agent_evaluation([case()], FakeGraph(), FakeJudge())

    assert details[0]["tool_correct"] is True
    assert details[0]["task_success"] is True
    assert summary["tool_selection_accuracy"] == 1
    assert summary["task_success_rate"] == 1
    assert summary["by_category"]["math"]["questions"] == 1


def test_summary_rejects_empty_results():
    with pytest.raises(ValueError, match="empty"):
        summarize_results([])


def test_save_results_writes_csv_and_json(tmp_path):
    details, summary = run_agent_evaluation([case()], FakeGraph(), FakeJudge())

    csv_path, json_path = save_results(details, summary, tmp_path)

    assert csv_path.is_file()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["summary"]["tool_selection_accuracy"] == 1
    assert report["details"][0]["id"] == "math-example"

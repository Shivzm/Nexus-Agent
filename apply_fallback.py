"""Adds the eval-only --fallback-top-k option to evals/run_eval.py and two tests.

Run from the repo root:  python apply_fallback.py
Works with LF or CRLF files, all-or-nothing (writes nothing if any spot is not found),
and safe to run twice.
"""
from pathlib import Path
import sys

RUN = Path("evals/run_eval.py")
TEST = Path("Tests/test_eval_resume.py")

EDITS = [
    ("make_target signature",
     "    guard: RunGuard | None = None,\n) -> Callable[[dict[str, Any]], dict[str, Any]]:\n    guard = guard or RunGuard()\n",
     "    guard: RunGuard | None = None,\n    fallback_k: int = 0,\n) -> Callable[[dict[str, Any]], dict[str, Any]]:\n"
     '    """``fallback_k`` > 0 is an eval-only experiment: when the score threshold filters out\n'
     "    every chunk, retry without a threshold and keep the closest ``fallback_k``. It never\n"
     "    changes ``retrieve()`` defaults, so the chatbot behaves exactly as before.\n"
     '    """\n    guard = guard or RunGuard()\n'),
    ("target retrieval",
     '        hits = retrieve_fn(question, top_k=5, search_config=search_config)\n        contexts =',
     '        hits = retrieve_fn(question, top_k=5, search_config=search_config)\n'
     "        if not hits and fallback_k > 0:\n"
     "            hits = retrieve_fn(question, top_k=fallback_k, score_threshold=0.0, search_config=search_config)\n"
     "        contexts ="),
    ("run_evaluation signature",
     "    judge_llm: Any = None,\n    guard: RunGuard | None = None,\n) -> Any:\n    if max_workers",
     "    judge_llm: Any = None,\n    guard: RunGuard | None = None,\n    fallback_k: int = 0,\n) -> Any:\n    if max_workers"),
    ("run_evaluation make_target call",
     "            search_config=search_config,\n            guard=guard,\n        ),\n        data=examples,",
     "            search_config=search_config,\n            guard=guard,\n            fallback_k=fallback_k,\n        ),\n        data=examples,"),
    ("experiment prefix",
     'experiment_prefix=f"{search_config}-baseline",',
     'experiment_prefix=f"{search_config}-baseline" + (f"-fallback{fallback_k}" if fallback_k else ""),'),
    ("save_results signature",
     '    search_config: Literal["dense", "hybrid"] = "dense",\n) -> tuple[Path, Path, dict[str, Any]]:',
     '    search_config: Literal["dense", "hybrid"] = "dense",\n    fallback_k: int = 0,\n) -> tuple[Path, Path, dict[str, Any]]:'),
    ("summary record",
     'summary.update({"questions": len(rows), "model": model, "config": search_config})',
     'summary.update({"questions": len(rows), "model": model, "config": search_config})\n'
     "    if fallback_k:\n        summary[\"fallback_top_k\"] = fallback_k"),
    ("result file names",
     '    csv_path = output_dir / f"{search_config}-{timestamp}.csv"\n    json_path = output_dir / f"{search_config}-{timestamp}.json"',
     '    label = f"{search_config}-fallback{fallback_k}" if fallback_k else search_config\n'
     '    csv_path = output_dir / f"{label}-{timestamp}.csv"\n    json_path = output_dir / f"{label}-{timestamp}.json"'),
    ("CLI option",
     '    parser.add_argument(\n        "--debug-retrieval",',
     '    parser.add_argument(\n        "--fallback-top-k",\n        type=int,\n        default=0,\n'
     '        help="Eval-only experiment (default 0 = off, matches the shipped chatbot): if the score "\n'
     '        "threshold removes every chunk, use the N closest chunks instead.",\n    )\n'
     '    parser.add_argument(\n        "--debug-retrieval",'),
    ("CLI validation",
     "    if args.max_retries < 0:",
     '    if args.fallback_top_k < 0:\n        parser.error("--fallback-top-k cannot be negative")\n    if args.max_retries < 0:'),
    ("main run_evaluation call",
     "        judge_llm=judge_llm,\n        guard=guard,\n    )\n    for name",
     "        judge_llm=judge_llm,\n        guard=guard,\n        fallback_k=args.fallback_top_k,\n    )\n    for name"),
    ("main save_results call",
     "        search_config=args.config,\n    )\n    print(json.dumps(summary",
     "        search_config=args.config,\n        fallback_k=args.fallback_top_k,\n    )\n    print(json.dumps(summary"),
]

TESTS = '''

def _threshold_retriever(calls):
    """Mimics retrieve(): the default threshold filters everything, threshold=0 keeps the closest chunks."""

    def retrieve_fn(question, top_k, search_config, score_threshold=None):
        calls.append((top_k, score_threshold))
        if score_threshold == 0.0:
            return [{"content": f"close chunk {i}", "metadata": {"source": "paper.pdf"}} for i in range(top_k)]
        return []

    return retrieve_fn


def test_fallback_is_off_by_default_and_matches_shipped_behaviour():
    from evals.run_eval import make_target

    calls = []
    target = make_target(FlakyLLM(), retrieve_fn=_threshold_retriever(calls))
    out = target({"question": "q"})
    assert out["contexts"] == [] and out["sources"] == []
    assert calls == [(5, None)]            # one plain retrieval, no second attempt


def test_fallback_top_k_retries_without_threshold_only_when_empty():
    from evals.run_eval import make_target

    calls = []
    target = make_target(FlakyLLM(), retrieve_fn=_threshold_retriever(calls), fallback_k=2)
    out = target({"question": "q"})
    assert len(out["contexts"]) == 2
    assert calls == [(5, None), (2, 0.0)]  # normal attempt first, then the fallback
'''


def read(path):
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    return raw.replace("\r\n", "\n"), crlf


def write(path, text, crlf):
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def main():
    for p in (RUN, TEST):
        if not p.exists():
            sys.exit(f"{p} not found - run this from the repo root (the folder with main.py).")

    run_text, run_crlf = read(RUN)
    test_text, test_crlf = read(TEST)

    if "fallback_k" in run_text:
        print("evals/run_eval.py already has the fallback option - nothing to do for it.")
    else:
        missing = [name for name, old, _ in EDITS if run_text.count(old) != 1]
        if missing:
            sys.exit("Nothing written. These spots in evals/run_eval.py were not found exactly once:\n  - "
                     + "\n  - ".join(missing) + "\nSend me the output of: git diff --stat")
        for _, old, new in EDITS:
            run_text = run_text.replace(old, new, 1)
        write(RUN, run_text, run_crlf)
        print("Updated evals/run_eval.py")

    if "test_fallback_top_k_retries" in test_text:
        print("Tests already added - nothing to do for them.")
    else:
        if "class FlakyLLM" not in test_text:
            sys.exit("Tests/test_eval_resume.py has no FlakyLLM class - was nexus_eval_ratelimit_plus_debug.patch applied?")
        write(TEST, test_text.rstrip("\n") + "\n" + TESTS, test_crlf)
        print("Added 2 tests to Tests/test_eval_resume.py")


main()

"""Download a small, on-theme PDF corpus (RAG, retrieval, LLM agents) from arXiv.

Run from the project root:

    uv run python -m src.scripts.download_corpus

PDFs are saved to <rag_docs_dir>/arxiv/ (default: data/docs/arxiv/) and are NOT
committed to git. Files that already exist are skipped, so it is safe to re-run.

The arXiv IDs below were written from memory: open the first page of a few PDFs
to confirm they match their file names.
"""

import time
import urllib.error
import urllib.request
from pathlib import Path

from config.settings import settings

PAPERS: dict[str, str] = {
    # --- retrieval and RAG
    "rag-lewis-2020": "2005.11401",
    "dense-passage-retrieval": "2004.04906",
    "sentence-bert": "1908.10084",
    "colbert": "2004.12832",
    "beir-benchmark": "2104.08663",
    "hyde-query-expansion": "2212.10496",
    "self-rag": "2310.11511",
    "corrective-rag": "2401.15884",
    "rag-survey": "2312.10997",
    "ragas-evaluation": "2309.15217",
    "lost-in-the-middle": "2307.03172",
    "query-rewriting-for-rag": "2305.14283",
    "bge-m3-embeddings": "2402.03216",
    "graphrag-local-to-global": "2404.16130",
    # --- agents and reasoning
    "react-agents": "2210.03629",
    "toolformer": "2302.04761",
    "chain-of-thought": "2201.11903",
    "tree-of-thoughts": "2305.10601",
    "webgpt": "2112.09332",
    "llm-agents-survey": "2308.11432",
    # --- language model foundations
    "bert": "1810.04805",
    "gpt3-few-shot": "2005.14165",
    "llama": "2302.13971",
    "llama-2": "2307.09288",
    "mistral-7b": "2310.06825",
    "lora": "2106.09685",
    "chinchilla-scaling": "2203.15556",
    "t5": "1910.10683",
}

DELAY_SECONDS = 3  # be polite to arXiv


def download(arxiv_id: str, dest: Path) -> None:
    request = urllib.request.Request(
        f"https://arxiv.org/pdf/{arxiv_id}",
        headers={"User-Agent": "Nexus-Agent corpus downloader (personal project)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise ValueError("response was not a PDF")
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def main() -> None:
    out_dir = Path(settings.rag_docs_dir) / "arxiv"
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded, skipped, failed = 0, 0, []
    for name, arxiv_id in PAPERS.items():
        dest = out_dir / f"{name}.pdf"
        if dest.exists():
            skipped += 1
            continue
        try:
            download(arxiv_id, dest)
            downloaded += 1
            print(f"downloaded {name}.pdf  (arXiv:{arxiv_id})")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failed.append(name)
            print(f"FAILED     {name}.pdf  (arXiv:{arxiv_id}): {exc}")
        time.sleep(DELAY_SECONDS)

    print(f"\nDone: {downloaded} downloaded, {skipped} already present, {len(failed)} failed.")
    if failed:
        print("Re-run to retry the failures, or download them by hand into", out_dir)


if __name__ == "__main__":
    main()

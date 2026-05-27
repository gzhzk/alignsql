"""Self-Consistency inference strategies.

Usage
-----
>>> from alignsql.models.inference import sample_candidates, execute_and_vote
"""

import sqlite3
import time
from typing import Any, Callable, Optional

from vllm import LLM, SamplingParams


def sample_candidates(
    llm: LLM,
    tokenizer,
    prompts: list[str],
    *,
    n: int = 5,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 512,
    system_prompt: str = "",
    extract_sql_fn: Callable[[str], str] = lambda x: x or "",
    strip_tokens: Optional[list[str]] = None,
) -> list[list[str]]:
    """Sample N candidate SQLs per prompt via vLLM.

    Uses vLLM's native ``n`` parameter so the model does one shared prefill
    and only the decode phase is repeated N times — much cheaper than calling
    generate() in a loop.

    Parameters
    ----------
    llm : LLM
        vLLM LLM instance.
    tokenizer
        Tokenizer with ``apply_chat_template``.
    prompts : list[str]
        User-message content, one per sample.
    n : int
        Number of candidates per prompt.
    temperature : float
        Sampling temperature (>0 for diversity).
    top_p : float
        Nucleus sampling threshold.
    max_tokens : int
        Max new tokens per candidate.
    system_prompt : str
        System-level instruction to prepend before each user message.
    extract_sql_fn : Callable
        Post-processing function to extract/clean SQL from raw generation.
    strip_tokens : list[str], optional
        Substrings to remove from the formatted chat template
        (e.g. Qwen think tokens ``<|think|>``).

    Returns
    -------
    list[list[str]]
        Shape ``(len(prompts), n)`` — cleaned candidate SQL strings.
    """
    strip = strip_tokens or []

    formatted = []
    for msg in prompts:
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for tok in strip:
            text = text.replace(tok, "")
        formatted.append(text)

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=n,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(formatted, params, use_tqdm=False)

    return [[extract_sql_fn(out.text) for out in o.outputs] for o in outputs]


def execute_and_vote(
    candidates: list[list[str]],
    db_paths: list[str],
    db_kwargs: Optional[dict] = None,
) -> list[str]:
    """Execute candidates and pick the winner by result-set voting.

    For each question:
      1. Execute every candidate SQL against its database.
      2. Group candidates by their result set (``str(sorted(rows))``).
      3. Discard candidates that raise an execution error.
      4. Select the group with the most members.
      5. Tie-break by average execution time (faster wins).

    Parameters
    ----------
    candidates : list[list[str]]
        ``(n_questions, n_candidates)`` — SQL strings.
    db_paths : list[str]
        Path to the SQLite database for each question.
    db_kwargs : dict, optional
        Extra keyword arguments forwarded to ``sqlite3.connect``
        (e.g. ``timeout=30``).

    Returns
    -------
    list[str]
        Winning SQL for each question.  Falls back to the first candidate
        when all candidates fail.
    """
    kwargs = {"timeout": 10.0, **(db_kwargs or {})}
    voted: list[str] = []

    for sqls, db_path in zip(candidates, db_paths):
        groups: dict[str, list[tuple[str, float]]] = {}

        for sql in sqls:
            if not sql:
                continue
            try:
                start = time.time()
                conn = sqlite3.connect(str(db_path), **kwargs)
                conn.text_factory = str
                cur = conn.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                conn.close()
                elapsed = time.time() - start

                key = str(sorted(rows))
                groups.setdefault(key, []).append((sql, elapsed))
            except Exception:
                continue

        if not groups:
            voted.append(sqls[0] if sqls else "")
            continue

        # Most members first, then fastest average execution time
        def sort_key(item):
            _key, members = item
            avg_time = sum(m[1] for m in members) / len(members)
            return (-len(members), avg_time)

        winner = sorted(groups.items(), key=sort_key)[0]
        voted.append(winner[1][0][0])  # first SQL of the winning group

    return voted

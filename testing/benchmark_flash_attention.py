"""Compatibility entry point for the Stateful Operator versus FLA benchmark.

The former implementation compared softmax attention with FlashAttention.  That
is not a semantic comparison for the linear-attention examples in this
repository.  The canonical benchmark is now ``benchmark_fla_stateful.py`` and
uses the installed FLA 0.5.2 release by default. Pass ``--fla-source`` to
select a local checkout explicitly.

Examples::

    CUDA_VISIBLE_DEVICES=0 uv run python testing/benchmark_flash_attention.py \
        --families linear,simple-gla,gla --seqlens 64 2048 \
        --batches 1 --heads 1 8 --contract all \
        --output-dir benchmark_results/stateful_fla

The wrapper is retained so existing benchmark commands select the corrected
baseline instead of silently measuring the unrelated FlashAttention package.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.benchmark_fla_stateful import main  # noqa: E402


if __name__ == "__main__":
    main()

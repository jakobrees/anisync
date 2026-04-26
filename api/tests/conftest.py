"""
Test bootstrap.

The recommender pipeline transitively imports heavy ML packages (torch,
sentence-transformers). Unit tests that target the *pure-Python* parts of
the pipeline (e.g. compute_vote_summary, _safe_embedding, _safe_score) do
not actually exercise the model, so we stub these modules out before any
`app.*` import is collected. If the real packages are installed (e.g. on a
developer laptop running the full webapp), we keep the real ones intact.
"""

from __future__ import annotations

import sys
import types


def _ensure_stub(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


# torch stub
try:  # pragma: no cover - exercised only on machines with torch installed
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = _ensure_stub("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Backends:
        class mps:
            @staticmethod
            def is_available() -> bool:
                return False

    torch_stub.cuda = _Cuda()
    torch_stub.backends = _Backends()


# sentence_transformers stub
try:  # pragma: no cover
    import sentence_transformers  # noqa: F401
except ModuleNotFoundError:
    st_stub = _ensure_stub("sentence_transformers")

    class _StubSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "sentence_transformers is stubbed for unit tests. "
                "Tests that need a real model must mark themselves with "
                "@pytest.mark.requires_model and be skipped in this env."
            )

        def encode(self, *args, **kwargs):  # pragma: no cover
            raise RuntimeError("Stubbed model: encode should not be called.")

    st_stub.SentenceTransformer = _StubSentenceTransformer

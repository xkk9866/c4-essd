"""Dataset builders producing a unified window-level table (meta columns + physiological features).

Unified schema (one row = one analysis window):
    dataset, subject, session (unique per subject x task instance), condition (str), design (1 stress/load-inducing,
    0 calm/baseline, -1 not design-controlled), win_start, q_<name> (self-report scores, NaN if missing), <features>.
"""
META_COLS = ["dataset", "subject", "session", "condition", "design", "win_start"]

import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def raw_path(env_var, *relative):
    """Locate a raw dataset that we are not allowed to redistribute.

    None of the three corpora may be shipped with this code, so their location is a local choice:
    set ``env_var`` to point at your own copy, otherwise we look under ``data/raw/`` in the repository.
    Keeping this out of the module bodies is also what stops an absolute path on one machine from
    travelling with the source.
    """
    return os.environ.get(env_var) or os.path.join(_REPO, "data", "raw", *relative)

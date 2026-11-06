"""
Contract metadata helpers (lot size resolution).

Historical lot sizes for SENSEX options have changed. Prefer values present
in the bhavcopy itself. If absent, skip the trade rather than invent a lot size.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Known historical lot sizes (documentation only — never silently applied without verification)
# Users / future sources can extend this table; engine still requires explicit value.
KNOWN_LOT_HINTS = {
    # period_start, period_end, lot_size  (illustrative; not auto-applied)
}


def resolve_lot_size(
    row: pd.Series,
    fallback: Optional[int] = None,
) -> Optional[int]:
    """
    Resolve lot size for a contract row.

    Priority:
      1. lot_size column present and valid (>0)
      2. explicit fallback argument (must be provided by caller knowingly)
      3. None → caller must skip trade
    """
    if "lot_size" in row.index:
        val = row["lot_size"]
        if pd.notna(val):
            try:
                ls = int(float(val))
                if ls > 0:
                    return ls
            except (TypeError, ValueError):
                pass
    if fallback is not None and fallback > 0:
        return int(fallback)
    return None

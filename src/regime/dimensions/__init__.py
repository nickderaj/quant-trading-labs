"""Import dimension modules to populate the regime indicator registry.

Ported verbatim from ``ultron_finance.regime.dimensions``.
"""

from regime.dimensions import (
    carry,
    macro,
    mean_reversion,
    term_structure,
    trend,
    volatility,
)

__all__ = ["carry", "macro", "mean_reversion", "term_structure", "trend", "volatility"]

"""Shared publication-figure settings."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
from matplotlib.figure import Figure

# Typical IEEE/Elsevier two-column text width.  Heights are intentionally compact
# so captions and surrounding text still fit comfortably on the page.
DOUBLE_COLUMN_WIDTH = 7.16
DOUBLE_COLUMN_FIGSIZE = (DOUBLE_COLUMN_WIDTH, 3.55)
DOUBLE_COLUMN_TALL_FIGSIZE = (DOUBLE_COLUMN_WIDTH, 4.05)
DOUBLE_COLUMN_TWO_PANEL_FIGSIZE = (DOUBLE_COLUMN_WIDTH, 3.15)
DOUBLE_COLUMN_THREE_PANEL_FIGSIZE = (DOUBLE_COLUMN_WIDTH, 3.05)
SINGLE_COLUMN_FIGSIZE = (3.50, 2.65)


def configure_publication_style() -> None:
    """Apply a restrained journal-style Matplotlib configuration."""
    mpl.rcParams.update(
        {
            # Prefer Times-like serif fonts but keep portable fallbacks.
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "figure.titlesize": 9.0,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.3,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            # Keep text as editable text in SVG and use TrueType in PDF.
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def save_publication_figure(
    fig: Figure,
    path: str | Path,
    *,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 600,
) -> dict[str, Path]:
    """Save a figure as PDF, SVG, and high-resolution PNG.

    ``path`` may include an extension; it is treated as the common filename stem.
    Existing parent directories are created automatically.
    """
    raw_path = Path(path)
    base = raw_path.with_suffix("") if raw_path.suffix else raw_path
    base.parent.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for fmt in formats:
        fmt_norm = str(fmt).lower().lstrip(".")
        output = base.with_suffix(f".{fmt_norm}")
        kwargs = {
            "format": fmt_norm,
            "bbox_inches": "tight",
            "pad_inches": 0.03,
        }
        if fmt_norm == "png":
            kwargs["dpi"] = int(dpi)
        fig.savefig(output, **kwargs)
        outputs[fmt_norm] = output
    return outputs


# Importing this module is enough to make all plot scripts use the same style.
configure_publication_style()

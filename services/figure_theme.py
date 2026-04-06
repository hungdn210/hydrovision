"""
figure_theme.py
~~~~~~~~~~~~~~~
Shared dark-mode colour constants and Plotly layout helpers for all
HydroVision figures.

Import from here instead of redeclaring locals in every service::

    from .figure_theme import (
        DARK_BG, TEXT, SOFT, GRID, BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW,
        SUBTLE_TEXT, GRID_LIGHT, GRID_STRONG,
        dark_layout, axis_style, hover_style,
        legend_h, legend_v, title_cfg,
        MARGIN_STD, MARGIN_SUBPLOT, MARGIN_MAP,
        method_note_annotation, forecast_divider_shape, forecast_divider_annotation,
    )
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ── Palette ───────────────────────────────────────────────────────────────────

# Backgrounds
DARK_BG     = '#07111f'            # paper_bgcolor / plot_bgcolor

# Text
TEXT        = '#e5eefc'            # primary labels, titles
SUBTLE_TEXT = '#b6c2d9'            # secondary annotations, slider ticks
SOFT        = '#9db0d1'            # axis tick fonts, subtle markers

# Grid / borders
GRID        = 'rgba(148,163,184,0.12)'   # standard grid lines
GRID_LIGHT  = 'rgba(148,163,184,0.08)'  # lighter variant (dense subplots)
GRID_STRONG = 'rgba(148,163,184,0.16)'  # heavier variant (sparse charts)

# Data colours
BLUE        = '#38bdf8'            # primary series / GEV fit
ORANGE      = '#fb923c'            # secondary series / Gumbel / drought alert
GREEN       = '#34d399'            # normal / positive / empirical points
RED         = '#f87171'            # flood risk / critical alert
PURPLE      = '#a78bfa'            # third series / wavelet power
YELLOW      = '#fbbf24'            # warning / accent highlight


# ── Typography constants ──────────────────────────────────────────────────────

TITLE_SIZE       = 14   # figure title
SUBTITLE_SIZE    = 11   # subplot titles / secondary headings
TICK_SIZE        = 10   # axis tick labels
ANNOT_SIZE       = 10   # in-chart annotation text
METHOD_NOTE_SIZE = 10   # provenance / method-note footer text


# ── Standard margins ──────────────────────────────────────────────────────────

# For single-panel standard charts
MARGIN_STD     = dict(l=70, r=30, t=60, b=50)

# For multi-panel (make_subplots) charts — extra top room for legend + title
MARGIN_SUBPLOT = dict(l=70, r=30, t=80, b=50)

# For map figures (Scattermapbox / Scatter geo) — no axis gutters needed
MARGIN_MAP     = dict(l=0, r=0, t=44, b=0)


# ── Layout helper ─────────────────────────────────────────────────────────────

def dark_layout(
    *,
    title: str = '',
    title_size: int = TITLE_SIZE,
    height: int = 560,
    margin: Optional[Dict[str, int]] = None,
    show_legend: bool = True,
    hovermode: str = 'x unified',
    **extra: Any,
) -> Dict[str, Any]:
    """
    Return a ``dict`` for ``fig.update_layout(**dark_layout(...))``.

    All parameters are keyword-only. Extra kwargs are merged in last and
    take precedence, so callers can override any key::

        fig.update_layout(**dark_layout(
            title='Extreme Value Analysis',
            height=620,
            hovermode='closest',
            showlegend=False,
        ))
    """
    m = margin if margin is not None else MARGIN_STD
    base: Dict[str, Any] = dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color=TEXT, size=12),
        height=height,
        margin=m,
        hovermode=hovermode,
        hoverlabel=dict(
            bgcolor='rgba(7,17,31,0.92)',
            bordercolor='rgba(148,163,184,0.28)',
            font=dict(color=TEXT, size=12, family='Inter, sans-serif'),
        ),
    )
    if title:
        base['title'] = dict(
            text=title,
            font=dict(size=title_size, color=TEXT),
            x=0.5,
            xanchor='center',
        )
    if show_legend:
        base['legend'] = legend_h()
    base.update(extra)
    return base


# ── Axis helper ───────────────────────────────────────────────────────────────

def axis_style(grid: str = GRID) -> Dict[str, Any]:
    """
    Return a ``dict`` for ``fig.update_xaxes(**axis_style())`` etc.

    Pass a custom ``grid`` colour string to match the local density::

        fig.update_yaxes(**axis_style(grid=GRID_LIGHT))
    """
    return dict(
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=grid,
        tickfont=dict(size=TICK_SIZE, color=SOFT),
    )


# ── Hover label helper ────────────────────────────────────────────────────────

def hover_style() -> Dict[str, Any]:
    """
    Return a ``dict`` for ``fig.update_layout(hoverlabel=hover_style())``.

    Keeps hover labels consistent with the dark background across all charts.
    """
    return dict(
        bgcolor='rgba(7,17,31,0.92)',
        bordercolor='rgba(148,163,184,0.28)',
        font=dict(color=TEXT, size=12, family='Inter, sans-serif'),
    )


# ── Legend helpers ────────────────────────────────────────────────────────────

def legend_h(*, y: float = 1.02) -> Dict[str, Any]:
    """
    Horizontal legend placed just above the plot area.

    This is the default legend style for all time-series and bar charts.
    Keeps the chart body clean while making series immediately identifiable.
    """
    return dict(
        orientation='h',
        yanchor='bottom',
        y=y,
        xanchor='left',
        x=0,
        font=dict(size=11),
        bgcolor='rgba(0,0,0,0)',
    )


def legend_v(*, x: float = 0.01, y: float = 0.99) -> Dict[str, Any]:
    """
    Vertical legend with dark pill background.

    Use this for map figures, network plots, and dense charts where a
    floating in-chart legend is needed (e.g. Scattermapbox, Scatter-geo).
    """
    return dict(
        orientation='v',
        yanchor='top',
        y=y,
        xanchor='left',
        x=x,
        font=dict(size=11),
        bgcolor='rgba(7,17,31,0.82)',
        bordercolor='rgba(148,163,184,0.15)',
        borderwidth=1,
    )


# ── Title helper ──────────────────────────────────────────────────────────────

def title_cfg(text: str, *, size: int = TITLE_SIZE) -> Dict[str, Any]:
    """
    Return the Plotly ``title`` dict for ``fig.update_layout(title=title_cfg(...))``.

    Centres the title with the standard font size and colour::

        fig.update_layout(title=title_cfg('My Chart'))
    """
    return dict(
        text=text,
        font=dict(size=size, color=TEXT),
        x=0.5,
        xanchor='center',
    )


# ── Subplot title font helper ─────────────────────────────────────────────────

def style_subplot_titles(fig, *, size: int = SUBTITLE_SIZE, color: str = SOFT) -> None:
    """
    Apply consistent font to all subplot annotation titles in-place.

    Call after ``make_subplots`` and after all traces are added::

        style_subplot_titles(fig)
    """
    for ann in fig.layout.annotations:
        ann.font.size  = size
        ann.font.color = color


# ── Provenance / method-note annotation ──────────────────────────────────────

def method_note_annotation(text: str, *, y: float = -0.13) -> Dict[str, Any]:
    """
    Return an annotation dict that stamps a method/provenance note below the
    chart at ``y`` (in paper coordinates, negative = below the plot area).

    Usage::

        fig.update_layout(annotations=[method_note_annotation(
            'GEV fit (Coles 2001) · 1000-resample bootstrap CI'
        )])

    Or append to an existing annotations list::

        fig.add_annotation(**method_note_annotation('...'))
    """
    return dict(
        text=text,
        xref='paper',
        yref='paper',
        x=0.0,
        y=y,
        xanchor='left',
        yanchor='top',
        showarrow=False,
        font=dict(size=METHOD_NOTE_SIZE, color=SUBTLE_TEXT),
    )


# ── Forecast divider helpers ──────────────────────────────────────────────────

def forecast_divider_shape(x_val: Any) -> Dict[str, Any]:
    """
    Return a vertical dotted line shape marking the start of a forecast window.

    ``x_val`` can be a date, integer (year), or any Plotly x-axis value.
    """
    return dict(
        type='line',
        x0=x_val, x1=x_val,
        y0=0, y1=1, yref='paper',
        line=dict(color='rgba(148,163,184,0.4)', dash='dot', width=1.5),
    )


def forecast_divider_annotation(x_val: Any, *, label: str = '↑ Forecast') -> Dict[str, Any]:
    """
    Return the annotation that labels the forecast divider line.
    """
    return dict(
        x=x_val,
        y=1.04,
        yref='paper',
        text=label,
        showarrow=False,
        xanchor='center',
        font=dict(color=SUBTLE_TEXT, size=ANNOT_SIZE),
    )

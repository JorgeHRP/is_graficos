"""
Cardan BI Dashboard API — versão modular
==========================================

Em vez de exigir sempre o JSON inteiro da tool `contar_cartao`, este
serviço recebe uma LISTA DE COMPONENTES — e desenha só o que vier.
Isso permite que a IA envie desde uma imagem com um único gráfico de
linha (ex.: "cartões dos últimos 3 meses") até um conjunto maior com
vários blocos, dependendo do que o treinamento decidir mandar.

Tipos de componente suportados (catálogo fixo e limitado):

  kpi_row      -> 1 a 4 números grandes lado a lado
  line_chart   -> evolução ao longo do tempo (ex.: cartões por mês)
  bar_compare  -> comparação de poucas categorias (barras verticais)
  ranked_list  -> ranking com mais itens (barras horizontais + valor)
  donut        -> proporção entre 2-3 partes (ex.: abertos x finalizados)
  alert_list   -> lista de cartões críticos/antigos

Executar localmente:
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /render            -> devolve image/png (bytes)
    GET  /components/examples -> exemplos de payload por tipo de componente
    GET  /health             -> healthcheck simples
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Cardan BI Dashboard API", version="3.0.0")

# ---------------------------------------------------------------------------
# Temas
# ---------------------------------------------------------------------------
THEMES = {
    "dark": dict(
        bg="#0d0f14", card="#171a21", card_edge="#252933", track="#232732",
        text="#f5f6f8", subtext="#8a909c", grid="#252933",
        accent="#5b8def", accent2="#2ecf9c", warn="#ff6b6b", amber="#ffb84d",
        purple="#a487ff",
    ),
    "light": dict(
        bg="#f4f6fa", card="#ffffff", card_edge="#e6e9f0", track="#eef1f6",
        text="#1c2230", subtext="#6b7284", grid="#e6e9f0",
        accent="#3f6df0", accent2="#16a67e", warn="#e5484d", amber="#dd8b1d",
        purple="#7c5cff",
    ),
}
COLOR_KEYS = ["accent", "accent2", "amber", "purple", "warn"]


def _apply_rcparams(T: dict):
    plt.rcParams.update({
        "figure.facecolor": T["bg"],
        "savefig.facecolor": T["bg"],
        "text.color": T["text"],
        "font.family": "DejaVu Sans",
        "font.size": 11,
    })


def _truncate(s: str, n: int = 30) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Schemas dos componentes (Pydantic) — isto É o contrato que a IA vai seguir
# ---------------------------------------------------------------------------
class KpiItem(BaseModel):
    label: str
    value: Union[str, int, float]
    sub: Optional[str] = None
    color: Optional[Literal["accent", "accent2", "amber", "purple", "warn"]] = None


class KpiRow(BaseModel):
    type: Literal["kpi_row"]
    items: List[KpiItem] = Field(..., min_length=1, max_length=4)


class LineChart(BaseModel):
    type: Literal["line_chart"]
    title: Optional[str] = None
    data: Dict[str, float] = Field(..., description="Ex.: {'Maio': 124, 'Junho': 69}")
    color: Optional[str] = None
    unit_suffix: Optional[str] = ""

    @field_validator("data")
    @classmethod
    def _not_empty(cls, v):
        if not v:
            raise ValueError("data não pode ser vazio")
        return v


class BarCompare(BaseModel):
    type: Literal["bar_compare"]
    title: Optional[str] = None
    data: Dict[str, float] = Field(..., description="Ex.: {'Barcelos': 110, 'Famalicão': 110}")
    color: Optional[str] = None

    @field_validator("data")
    @classmethod
    def _not_empty(cls, v):
        if not v:
            raise ValueError("data não pode ser vazio")
        return v


class RankedList(BaseModel):
    type: Literal["ranked_list"]
    title: Optional[str] = None
    data: Dict[str, float]
    max_items: int = 7

    @field_validator("data")
    @classmethod
    def _not_empty(cls, v):
        if not v:
            raise ValueError("data não pode ser vazio")
        return v


class Donut(BaseModel):
    type: Literal["donut"]
    title: Optional[str] = None
    labels: List[str] = Field(..., min_length=2, max_length=3)
    values: List[float] = Field(..., min_length=2, max_length=3)
    center_label: Optional[str] = None

    @field_validator("values")
    @classmethod
    def _same_len(cls, v, info):
        labels = info.data.get("labels")
        if labels is not None and len(v) != len(labels):
            raise ValueError("labels e values devem ter o mesmo tamanho")
        return v


class AlertItem(BaseModel):
    cliente: str
    assunto: Optional[str] = None
    unidade: Optional[str] = None
    agente: Optional[str] = None
    dias: int


class AlertList(BaseModel):
    type: Literal["alert_list"]
    title: Optional[str] = "Cartões abertos há mais tempo"
    items: List[AlertItem]
    max_items: int = 5


Component = Union[KpiRow, LineChart, BarCompare, RankedList, Donut, AlertList]


class RenderRequest(BaseModel):
    titulo: Optional[str] = None
    subtitulo: Optional[str] = None
    theme: Literal["dark", "light"] = "dark"
    footer: bool = False
    components: List[Component] = Field(..., min_length=1, max_length=10)


# ---------------------------------------------------------------------------
# Layout engine — largura fixa, altura calculada bloco a bloco
# ---------------------------------------------------------------------------
FIG_W = 6.2
MARGIN_L = 0.34
MARGIN_R = 0.34
MARGIN_T = 0.28
MARGIN_B = 0.22
CONTENT_W = FIG_W - MARGIN_L - MARGIN_R

H_HEADER = 0.95
LIST_TITLE_H = 0.46
LIST_ROW_H = 0.445
ALERT_ROW_H = 0.60
H_FOOTER = 0.28


def _rounded_card(ax, x, y, w, h, facecolor, edgecolor=None, radius=0.045, lw=1.0, zorder=1):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          linewidth=lw, edgecolor=edgecolor or "none",
                          facecolor=facecolor, transform=ax.transAxes,
                          clip_on=False, zorder=zorder, mutation_aspect=1)
    ax.add_patch(box)
    return box


def _safe_radius(w: float, h: float, frac: float = 0.42) -> float:
    """FancyBboxPatch gera um caminho autointersectante (visual em "laço") quando
    rounding_size passa de metade do lado menor da caixa. Isto trava o raio
    sempre abaixo desse limite."""
    return max(min(w, h) * frac, 0.0)


def _circle_wh(block_h_in: float, diameter_in: float) -> tuple[float, float]:
    """Largura/altura (em fração de eixo, 0-1) para um círculo com diâmetro real
    de `diameter_in` polegadas. Necessário porque cada eixo aqui tem largura
    física CONTENT_W e altura física `block_h_in` diferentes — um raio igual em
    x/y nessas frações sairia como elipse achatada, não círculo."""
    return diameter_in / CONTENT_W, diameter_in / block_h_in


# ---- header / footer --------------------------------------------------
def _draw_header(ax, T, titulo, subtitulo):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # A axes do header é bem mais larga que alta (aspect físico != 1), então um
    # círculo desenhado com raio igual em x/y (em coordenadas de eixo) sai como
    # uma elipse achatada — aqui compensamos o aspect para o badge sair redondo.
    aspect = CONTENT_W / H_HEADER
    badge_h = 0.34
    badge_w = badge_h / aspect
    badge_cx, badge_cy = badge_w / 2 + 0.02, 0.76

    ax.add_patch(Ellipse((badge_cx, badge_cy), badge_w, badge_h, facecolor=T["accent"],
                          transform=ax.transAxes, clip_on=False, zorder=2))
    ax.text(badge_cx, badge_cy, "CB", fontsize=11, fontweight="bold", color="#ffffff",
            ha="center", va="center", transform=ax.transAxes)

    text_x = badge_w + 0.08
    ax.text(text_x, 0.82, titulo or "Cardan", fontsize=15, fontweight="bold",
            color=T["text"], ha="left", va="center", transform=ax.transAxes)
    if subtitulo:
        ax.text(text_x, 0.58, subtitulo, fontsize=9.6, color=T["subtext"],
                ha="left", va="center", transform=ax.transAxes)
    ax.text(0.0, 0.06, f"Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}",
            fontsize=8, color=T["subtext"], ha="left", va="center",
            transform=ax.transAxes)
    ax.plot([0, 1], [0.0, 0.0], color=T["grid"], lw=1, transform=ax.transAxes,
            clip_on=False)


def _draw_footer(ax, T):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.plot([0, 1], [1, 1], color=T["grid"], lw=1, transform=ax.transAxes, clip_on=False)
    ax.text(0.5, 0.35, "Cardan · relatório automático", fontsize=8,
            color=T["subtext"], ha="center", va="center", transform=ax.transAxes)


# ---- kpi_row ------------------------------------------------------------
def _kpi_cell(ax, x, y, w, h, T, label, value, color, block_h, sub=None):
    _rounded_card(ax, x, y, w, h, T["card"], edgecolor=T["card_edge"], radius=0.07, lw=1.1)
    dot_w, dot_h = _circle_wh(block_h, 0.12)
    ax.add_patch(Ellipse((x + w * 0.10 + dot_w / 2, y + h - h * 0.16), dot_w, dot_h,
                          facecolor=color, transform=ax.transAxes, clip_on=False, zorder=3))
    pad_x = x + w * 0.10
    ax.text(pad_x, y + h - h * 0.24, str(label).upper(), fontsize=8.2, color=T["subtext"],
            ha="left", va="center", transform=ax.transAxes, fontweight="bold")
    ax.text(pad_x, y + h * 0.46, str(value), fontsize=18.5, fontweight="bold",
            color=T["text"], ha="left", va="center", transform=ax.transAxes)
    if sub:
        ax.text(pad_x, y + h * 0.16, sub, fontsize=8.4, color=color,
                ha="left", va="center", transform=ax.transAxes, fontweight="bold")


def _height_kpi_row(comp: KpiRow) -> float:
    n = len(comp.items)
    return 1.55 if n <= 2 else (1.85 if n == 4 else 1.45)


def _draw_kpi_row(ax, T, comp: KpiRow):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(comp.items)
    gap = 0.045
    colors_cycle = COLOR_KEYS
    block_h = _height_kpi_row(comp)

    if n == 1:
        it = comp.items[0]
        _kpi_cell(ax, 0, 0, 1, 1, T, it.label, it.value,
                  T[it.color or "accent"], block_h, it.sub)
    elif n in (2, 3):
        w = (1 - gap * (n - 1)) / n
        for i, it in enumerate(comp.items):
            _kpi_cell(ax, i * (w + gap), 0, w, 1, T, it.label, it.value,
                      T[it.color or colors_cycle[i % len(colors_cycle)]], block_h, it.sub)
    else:  # 4 -> grade 2x2
        w = (1 - gap) / 2
        h = (1 - gap) / 2
        positions = [(0, h + gap), (w + gap, h + gap), (0, 0), (w + gap, 0)]
        for i, it in enumerate(comp.items):
            x, y = positions[i]
            _kpi_cell(ax, x, y, w, h, T, it.label, it.value,
                      T[it.color or colors_cycle[i % len(colors_cycle)]], block_h, it.sub)


# ---- line_chart -----------------------------------------------------------
def _height_line_chart(comp: LineChart) -> float:
    return 2.15 + (0.35 if comp.title else 0)


def _draw_line_chart(fig, ax, T, comp: LineChart, bottom, block_h):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _rounded_card(ax, 0, 0, 1, 1, T["card"], edgecolor=T["card_edge"], radius=0.03, lw=1.1)
    top_pad = 0.0
    if comp.title:
        ax.text(0.055, 0.90, comp.title, fontsize=11.5, fontweight="bold",
                color=T["text"], ha="left", va="center", transform=ax.transAxes)
        top_pad = 0.14

    color = T[comp.color] if (comp.color and comp.color in T) else T["accent"]
    labels = list(comp.data.keys())
    values = list(comp.data.values())

    pos = ax.get_position()
    inner = fig.add_axes([
        pos.x0 + 0.07 / FIG_W * FIG_W * 0 + pos.width * 0.08,
        pos.y0 + pos.height * 0.16,
        pos.width * 0.86,
        pos.height * (0.62 - top_pad + 0.14),
    ])
    inner.set_facecolor("none")
    x = list(range(len(values)))
    inner.plot(x, values, color=color, linewidth=2.4, marker="o", markersize=6,
               markerfacecolor=color, markeredgecolor=T["card"], markeredgewidth=1.4,
               zorder=3)
    inner.fill_between(x, values, min(values) - (max(values) - min(values) or 1) * 0.15,
                        color=color, alpha=0.14, zorder=1)

    vmin, vmax = min(values), max(values)
    pad = (vmax - vmin) * 0.25 or max(vmax * 0.2, 1)
    inner.set_ylim(vmin - pad * 0.4, vmax + pad)
    inner.set_xlim(-0.4, len(values) - 0.6)

    for xi, v in zip(x, values):
        inner.text(xi, v + pad * 0.28, f"{v:.0f}{comp.unit_suffix or ''}", ha="center",
                    va="bottom", fontsize=9, color=T["text"], fontweight="bold")

    inner.set_xticks(x)
    inner.set_xticklabels([_truncate(l, 10) for l in labels], fontsize=9, color=T["subtext"])
    inner.set_yticks([])
    for s in inner.spines.values():
        s.set_visible(False)
    inner.spines["bottom"].set_visible(True)
    inner.spines["bottom"].set_color(T["grid"])
    inner.tick_params(axis="x", length=0)


# ---- bar_compare ------------------------------------------------------------
def _height_bar_compare(comp: BarCompare) -> float:
    return 2.15 + (0.35 if comp.title else 0)


def _draw_bar_compare(fig, ax, T, comp: BarCompare, bottom, block_h):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _rounded_card(ax, 0, 0, 1, 1, T["card"], edgecolor=T["card_edge"], radius=0.03, lw=1.1)
    top_pad = 0.0
    if comp.title:
        ax.text(0.055, 0.90, comp.title, fontsize=11.5, fontweight="bold",
                color=T["text"], ha="left", va="center", transform=ax.transAxes)
        top_pad = 0.14

    color = T[comp.color] if (comp.color and comp.color in T) else T["accent2"]
    labels = list(comp.data.keys())
    values = list(comp.data.values())

    pos = ax.get_position()
    inner = fig.add_axes([
        pos.x0 + pos.width * 0.08,
        pos.y0 + pos.height * 0.16,
        pos.width * 0.86,
        pos.height * (0.62 - top_pad + 0.14),
    ])
    inner.set_facecolor("none")
    x = list(range(len(values)))
    maxv = max(values) or 1
    bars = inner.bar(x, values, color=color, width=0.55, zorder=3,
                      edgecolor="none")
    for b in bars:
        b.set_clip_on(False)
    for xi, v in zip(x, values):
        inner.text(xi, v + maxv * 0.04, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=9.3, color=T["text"], fontweight="bold")

    inner.set_ylim(0, maxv * 1.25)
    inner.set_xticks(x)
    inner.set_xticklabels([_truncate(l, 12) for l in labels], fontsize=9, color=T["subtext"])
    inner.set_yticks([])
    for s in inner.spines.values():
        s.set_visible(False)
    inner.tick_params(axis="x", length=0)


# ---- ranked_list ------------------------------------------------------------
def _height_ranked_list(comp: RankedList) -> float:
    n = min(len(comp.data), comp.max_items) or 1
    return (LIST_TITLE_H if comp.title else 0.18) + n * LIST_ROW_H


def _draw_ranked_list(ax, T, comp: RankedList):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if comp.title:
        ax.text(0, 1.0, comp.title, fontsize=12, fontweight="bold", color=T["text"],
                ha="left", va="top", transform=ax.transAxes)

    items = sorted(comp.data.items(), key=lambda kv: kv[1], reverse=True)[:comp.max_items]
    if not items:
        ax.text(0, 0.55, "Sem dados.", fontsize=9.5, color=T["subtext"],
                transform=ax.transAxes)
        return

    n = len(items)
    title_frac = (LIST_TITLE_H if comp.title else 0.18) / ((LIST_TITLE_H if comp.title else 0.18) + n * LIST_ROW_H)
    top_y = 1.0 - title_frac
    row_h = top_y / n
    maxv = max(v for _, v in items) or 1

    for i, (label, value) in enumerate(items):
        y_top = top_y - i * row_h
        color = T[COLOR_KEYS[i % len(COLOR_KEYS)]]
        name_y = y_top - row_h * 0.30
        ax.text(0, name_y, _truncate(label, 32), fontsize=9.8, color=T["text"],
                ha="left", va="center", transform=ax.transAxes)
        ax.text(1, name_y, f"{value:.0f}", fontsize=9.8, color=T["text"],
                fontweight="bold", ha="right", va="center", transform=ax.transAxes)
        bar_y = y_top - row_h * 0.74
        bar_h = row_h * 0.20
        _rounded_card(ax, 0, bar_y, 1, bar_h, T["track"], radius=_safe_radius(1, bar_h, 0.5))
        frac = max(value / maxv, 0.035)
        _rounded_card(ax, 0, bar_y, frac, bar_h, color, radius=_safe_radius(frac, bar_h, 0.5))


# ---- donut ------------------------------------------------------------
def _height_donut(comp: Donut) -> float:
    return 2.05


def _draw_donut(fig, ax, T, comp: Donut, bottom, block_h):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _rounded_card(ax, 0, 0, 1, 1, T["card"], edgecolor=T["card_edge"], radius=0.035, lw=1.1)
    if comp.title:
        ax.text(0.06, 0.90, comp.title, fontsize=11.5, fontweight="bold",
                color=T["text"], ha="left", va="center", transform=ax.transAxes)

    values = comp.values
    total = sum(values) or 1
    palette = [T["warn"], T["accent2"], T["amber"]]

    pos = ax.get_position()
    side_in = min(pos.width * FIG_W, block_h) * 0.60
    donut_ax = fig.add_axes([
        pos.x0 + 0.05,
        pos.y0 + (pos.height - side_in / block_h) / 2 - 0.02,
        side_in / FIG_W,
        side_in / block_h,
    ])
    donut_ax.set_aspect("equal")
    donut_ax.axis("off")
    donut_ax.pie(values, colors=palette[:len(values)], startangle=90,
                 wedgeprops=dict(width=0.34, edgecolor=T["card"], linewidth=2))
    center_txt = comp.center_label or f"{values[0] / total * 100:.0f}%"
    donut_ax.text(0, 0.08, center_txt, fontsize=15, fontweight="bold",
                  color=T["text"], ha="center", va="center")

    lx = 0.50
    ly_start = 0.66
    dot_w, dot_h = _circle_wh(block_h, 0.09)
    for i, (lab, val) in enumerate(zip(comp.labels, values)):
        ly = ly_start - i * 0.17
        ax.add_patch(Ellipse((lx, ly), dot_w, dot_h, facecolor=palette[i % len(palette)],
                              transform=ax.transAxes, clip_on=False))
        ax.text(lx + 0.05, ly, _truncate(lab, 16), fontsize=9.5, color=T["text"],
                ha="left", va="center", transform=ax.transAxes)
        ax.text(0.97, ly, f"{val:.0f}", fontsize=9.5, color=T["text"], fontweight="bold",
                ha="right", va="center", transform=ax.transAxes)


# ---- alert_list ------------------------------------------------------------
def _height_alert_list(comp: AlertList) -> float:
    n = min(len(comp.items), comp.max_items) or 1
    return (LIST_TITLE_H if comp.title else 0.18) + n * ALERT_ROW_H


def _draw_alert_list(ax, T, comp: AlertList, block_h: float):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if comp.title:
        ax.text(0, 1.0, comp.title, fontsize=12, fontweight="bold",
                color=T["text"], ha="left", va="top", transform=ax.transAxes)

    rows = sorted(comp.items, key=lambda a: a.dias, reverse=True)[:comp.max_items]
    if not rows:
        ax.text(0, 0.55, "Nenhum alerta. 🎉", fontsize=10, color=T["accent2"],
                transform=ax.transAxes)
        return

    n = len(rows)
    title_frac = (LIST_TITLE_H if comp.title else 0.18) / ((LIST_TITLE_H if comp.title else 0.18) + n * ALERT_ROW_H)
    top_y = 1.0 - title_frac
    row_h = top_y / n
    dot_w, dot_h = _circle_wh(block_h, 0.07)

    for i, item in enumerate(rows):
        y_top = top_y - i * row_h
        dias = item.dias
        color = T["warn"] if dias >= 30 else (T["amber"] if dias >= 7 else T["accent2"])

        ax.add_patch(Ellipse((dot_w / 2, y_top - row_h * 0.28), dot_w, dot_h, facecolor=color,
                              transform=ax.transAxes, clip_on=False))
        ax.text(0.045, y_top - row_h * 0.22, item.cliente, fontsize=9.7,
                color=T["text"], fontweight="bold", ha="left", va="center",
                transform=ax.transAxes)
        detalhe = " · ".join(x for x in [item.assunto, item.unidade, item.agente] if x)
        ax.text(0.045, y_top - row_h * 0.58, _truncate(detalhe, 40), fontsize=8.3,
                color=T["subtext"], ha="left", va="center", transform=ax.transAxes)

        badge_w, badge_h = 0.17, row_h * 0.55
        _rounded_card(ax, 0.83, y_top - row_h * 0.78, badge_w, badge_h, color,
                      radius=_safe_radius(badge_w, badge_h, 0.5))
        ax.text(0.915, y_top - row_h * 0.50, f"{dias}d", fontsize=9, fontweight="bold",
                color="#ffffff", ha="center", va="center", transform=ax.transAxes)

        if i < n - 1:
            ax.plot([0, 1], [y_top - row_h, y_top - row_h], color=T["grid"], lw=0.7,
                    transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------
def render_png(req: RenderRequest) -> bytes:
    T = THEMES[req.theme]
    _apply_rcparams(T)

    blocks = []
    if req.titulo:
        blocks.append(("header", H_HEADER, None))
    for comp in req.components:
        if isinstance(comp, KpiRow):
            blocks.append(("kpi_row", _height_kpi_row(comp), comp))
        elif isinstance(comp, LineChart):
            blocks.append(("line_chart", _height_line_chart(comp), comp))
        elif isinstance(comp, BarCompare):
            blocks.append(("bar_compare", _height_bar_compare(comp), comp))
        elif isinstance(comp, RankedList):
            blocks.append(("ranked_list", _height_ranked_list(comp), comp))
        elif isinstance(comp, Donut):
            blocks.append(("donut", _height_donut(comp), comp))
        elif isinstance(comp, AlertList):
            blocks.append(("alert_list", _height_alert_list(comp), comp))
    if req.footer:
        blocks.append(("footer", H_FOOTER, None))

    total_h = MARGIN_T + MARGIN_B + sum(h for _, h, _ in blocks)
    fig = plt.figure(figsize=(FIG_W, total_h), dpi=200)
    fig.patch.set_facecolor(T["bg"])

    cursor_top = total_h - MARGIN_T
    for kind, h, comp in blocks:
        bottom = cursor_top - h
        rect = [MARGIN_L / FIG_W, bottom / total_h, CONTENT_W / FIG_W, h / total_h]
        ax = fig.add_axes(rect)

        if kind == "header":
            _draw_header(ax, T, req.titulo, req.subtitulo)
        elif kind == "kpi_row":
            _draw_kpi_row(ax, T, comp)
        elif kind == "line_chart":
            _draw_line_chart(fig, ax, T, comp, bottom, h)
        elif kind == "bar_compare":
            _draw_bar_compare(fig, ax, T, comp, bottom, h)
        elif kind == "ranked_list":
            _draw_ranked_list(ax, T, comp)
        elif kind == "donut":
            _draw_donut(fig, ax, T, comp, bottom, h)
        elif kind == "alert_list":
            _draw_alert_list(ax, T, comp, h)
        elif kind == "footer":
            _draw_footer(ax, T)

        cursor_top = bottom

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=T["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/render")
def render(req: RenderRequest):
    try:
        png_bytes = render_png(req)
    except Exception as e:
        raise HTTPException(400, f"Erro ao gerar imagem: {e}")
    return Response(content=png_bytes, media_type="image/png")


@app.get("/components/examples")
def component_examples():
    """Exemplos de payload prontos — úteis para escrever o treinamento e o curl."""
    return JSONResponse({
        "exemplo_kpi_unico": {
            "titulo": "Cardan Barcelos",
            "theme": "dark",
            "components": [
                {"type": "kpi_row", "items": [
                    {"label": "Cartões no mês passado", "value": 56, "color": "accent"}
                ]}
            ],
        },
        "exemplo_line_chart_simples": {
            "titulo": "Evolução mensal",
            "theme": "dark",
            "components": [
                {"type": "line_chart", "title": "Cartões por mês",
                 "data": {"Maio": 124, "Junho": 69, "Julho": 56}}
            ],
        },
        "exemplo_kpis_mais_grafico": {
            "titulo": "Cardan Barcelos",
            "subtitulo": "Últimos 7 dias",
            "theme": "dark",
            "components": [
                {"type": "kpi_row", "items": [
                    {"label": "Total", "value": 10, "color": "accent"},
                    {"label": "Abertos", "value": 2, "color": "warn"},
                ]},
                {"type": "bar_compare", "title": "Por unidade",
                 "data": {"Barcelos": 2, "Famalicão": 2, "Guimarães": 4, "Penafiel": 2}},
            ],
        },
        "exemplo_donut": {
            "titulo": "Situação dos cartões",
            "theme": "light",
            "components": [
                {"type": "donut", "title": "Abertos x Finalizados",
                 "labels": ["Abertos", "Finalizados"], "values": [139, 301]}
            ],
        },
        "exemplo_ranked_list": {
            "titulo": "Cardan · Por tipo de assunto",
            "theme": "dark",
            "components": [
                {"type": "ranked_list", "title": "Cartões por tipo de assunto",
                 "data": {"Mecânica (Marcações)": 96, "Mecânica (Outros)": 66,
                          "Vendas Novos": 52, "Vendas Usados": 46, "Peças": 44}}
            ],
        },
        "exemplo_alertas": {
            "titulo": "Atenção",
            "theme": "dark",
            "components": [
                {"type": "alert_list", "title": "Cartões abertos há mais tempo",
                 "items": [
                     {"cliente": "Bruno Machado", "assunto": "Mecânica (Outros)",
                      "unidade": "Barcelos", "agente": "Não atribuído", "dias": 200}
                 ]}
            ],
        },
    })

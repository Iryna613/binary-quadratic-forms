from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

import plotly.graph_objects as go

from .quadratic_form import QuadraticForm
from .topograph import LaxVector, Topograph, TopographEdge, add, generate_topograph, sub

SuperbaseKey = Tuple[LaxVector, LaxVector, LaxVector]


@dataclass(frozen=True)
class RegionSlot:
    vector: LaxVector
    center: Tuple[float, float]


@dataclass(frozen=True)
class EdgeSlot:
    edge: TopographEdge
    p0: Tuple[float, float]
    p1: Tuple[float, float]


@dataclass(frozen=True)
class TopographTemplate:
    depth: int
    superbase_positions: Dict[SuperbaseKey, Tuple[float, float]]
    region_slots: Dict[LaxVector, RegionSlot]
    edge_slots: Dict[TopographEdge, EdgeSlot]

def _slot(u: LaxVector, v: LaxVector, p0: Tuple[float, float], p1: Tuple[float, float]) -> EdgeSlot:
    return EdgeSlot(edge=TopographEdge(u, v), p0=p0, p1=p1)


FIXED_TEMPLATE_MAX_DEPTH = 7


def _fixed_classic_superbase_positions(
    edge_to_superbases: Dict[TopographEdge, Tuple[SuperbaseKey, SuperbaseKey]]
) -> Dict[SuperbaseKey, Tuple[float, float]]:
    """
    Fixed radial interval layout up to depth 7.

    The geometry is independent of the quadratic form Q.
    It is generated once from the canonical topograph structure.

    Main idea:
      - root the topograph at the central edge {(1,0),(0,1)};
      - draw one rooted tree in the upper half-plane;
      - draw the other rooted tree in the lower half-plane;
      - give every subtree its own angular interval.

    This avoids the collision problem of the previous greedy 8-direction layout.
    """
    from math import cos, sin, pi

    adjacency = _build_adjacency(edge_to_superbases)
    if not adjacency:
        return {}

    central = _pick_central_edge(edge_to_superbases)
    if central is None:
        return {}

    _, upper_root, lower_root = central

    # The superbase containing (1,1) should be above the central edge.
    if LaxVector(1, 1) in lower_root and LaxVector(1, 1) not in upper_root:
        upper_root, lower_root = lower_root, upper_root

    def ordered_children(node: SuperbaseKey, parent: Optional[SuperbaseKey]) -> List[SuperbaseKey]:
        return sorted(
            (n for n in adjacency[node] if n != parent),
            key=lambda n: tuple(v.as_tuple() for v in n),
        )

    leaf_cache: Dict[Tuple[SuperbaseKey, Optional[SuperbaseKey]], int] = {}

    def leaf_count(node: SuperbaseKey, parent: Optional[SuperbaseKey]) -> int:
        key = (node, parent)
        if key in leaf_cache:
            return leaf_cache[key]

        children = ordered_children(node, parent)
        if not children:
            leaf_cache[key] = 1
        else:
            leaf_cache[key] = sum(leaf_count(ch, node) for ch in children)

        return leaf_cache[key]

    positions: Dict[SuperbaseKey, Tuple[float, float]] = {}

    root_gap = 0.95
    radial_gap = 0.72

    # Central edge endpoints.
    positions[upper_root] = (0.0, root_gap / 2.0)
    positions[lower_root] = (0.0, -root_gap / 2.0)

    def polar(radius: float, angle_deg: float) -> Tuple[float, float]:
        a = angle_deg * pi / 180.0
        return radius * cos(a), radius * sin(a)

    def place_subtree(
        node: SuperbaseKey,
        parent: SuperbaseKey,
        depth: int,
        angle_left: float,
        angle_right: float,
        min_radius: float,
    ) -> None:
        children = ordered_children(node, parent)
        if not children:
            return

        weights = [leaf_count(ch, node) for ch in children]
        total = float(sum(weights))

        cursor = angle_left

        for child, weight in zip(children, weights):
            width = (angle_right - angle_left) * (weight / total)
            child_left = cursor
            child_right = cursor + width
            child_angle = (child_left + child_right) / 2.0

            # Radius grows with depth. This is the key fix:
            # do NOT shrink steps at large depth.
            radius = min_radius + depth * radial_gap
            x, y = polar(radius, child_angle)

            positions[child] = (x, y)

            place_subtree(
                node=child,
                parent=node,
                depth=depth + 1,
                angle_left=child_left,
                angle_right=child_right,
                min_radius=min_radius,
            )

            cursor += width

    # Upper half gets angular interval [25°, 155°].
    # Lower half gets angular interval [205°, 335°].
    # This keeps the figure compact but prevents branch collisions.
    place_subtree(
        node=upper_root,
        parent=lower_root,
        depth=1,
        angle_left=25.0,
        angle_right=155.0,
        min_radius=root_gap,
    )

    place_subtree(
        node=lower_root,
        parent=upper_root,
        depth=1,
        angle_left=205.0,
        angle_right=335.0,
        min_radius=root_gap,
    )

    return positions

def _manual_classic_template(depth: int) -> TopographTemplate:
    """
    Fixed template renderer up to depth 7.

    Important:
    - the template is generated from a canonical topograph;
    - the geometry does not depend on Q;
    - for every form, the same region vectors occupy the same positions.
    """
    depth = max(1, min(depth, FIXED_TEMPLATE_MAX_DEPTH))

    canonical_form = QuadraticForm(1, 0, 1)
    canonical = generate_topograph(canonical_form, depth=depth)

    edge_to_superbases = _build_primal_edges(canonical)
    positions = _fixed_classic_superbase_positions(edge_to_superbases)

    edge_slots: Dict[TopographEdge, EdgeSlot] = {}
    for edge, (a, b) in edge_to_superbases.items():
        if a in positions and b in positions:
            edge_slots[edge] = EdgeSlot(
                edge=edge,
                p0=positions[a],
                p1=positions[b],
            )

    region_centers = _region_positions(
        canonical.regions.keys(),
        edge_to_superbases,
        positions,
    )

    # Manual central corrections so the central picture looks like the reference.
    central_overrides: Dict[LaxVector, Tuple[float, float]] = {
        LaxVector(1, 0): (-0.38, 0.03),
        LaxVector(0, 1): (0.38, 0.03),
        LaxVector(1, 1): (0.0, 1.08),
        LaxVector(-1, 1): (0.0, -1.08),
    }
    region_centers.update(central_overrides)

    region_slots = {
        vector: RegionSlot(vector=vector, center=center)
        for vector, center in region_centers.items()
    }

    return TopographTemplate(
        depth=depth,
        superbase_positions=positions,
        region_slots=region_slots,
        edge_slots=edge_slots,
    )
def _superbase_key(a: LaxVector, b: LaxVector, c: LaxVector) -> SuperbaseKey:
    return tuple(sorted((a, b, c)))  # type: ignore[return-value]


def _build_primal_edges(topograph: Topograph) -> Dict[TopographEdge, Tuple[SuperbaseKey, SuperbaseKey]]:
    """
    A topograph edge between regions u,v corresponds to the segment connecting
    superbases {u,v,u+v} and {u,v,u-v} in the dual graph.
    """
    data: Dict[TopographEdge, Tuple[SuperbaseKey, SuperbaseKey]] = {}
    for edge in topograph.edges:
        u, v = edge.u, edge.v
        data[edge] = (_superbase_key(u, v, add(u, v)), _superbase_key(u, v, sub(u, v)))
    return data


def _build_adjacency(edge_to_superbases: Dict[TopographEdge, Tuple[SuperbaseKey, SuperbaseKey]]):
    adjacency: DefaultDict[SuperbaseKey, List[SuperbaseKey]] = defaultdict(list)
    for a, b in edge_to_superbases.values():
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def _pick_central_edge(
    edge_to_superbases: Dict[TopographEdge, Tuple[SuperbaseKey, SuperbaseKey]]
) -> Optional[Tuple[TopographEdge, SuperbaseKey, SuperbaseKey]]:
    if not edge_to_superbases:
        return None

    try:
        canonical = TopographEdge(LaxVector(1, 0), LaxVector(0, 1))
    except ValueError:
        canonical = None

    if canonical is not None and canonical in edge_to_superbases:
        a, b = edge_to_superbases[canonical]
        return canonical, a, b

    def edge_score(item: Tuple[TopographEdge, Tuple[SuperbaseKey, SuperbaseKey]]) -> Tuple[int, Tuple[int, int], Tuple[int, int]]:
        edge, _ = item
        norm = abs(edge.u.m) + abs(edge.u.n) + abs(edge.v.m) + abs(edge.v.n)
        return norm, edge.u.as_tuple(), edge.v.as_tuple()

    edge, (a, b) = sorted(edge_to_superbases.items(), key=edge_score)[0]
    return edge, a, b


def _region_positions(
    regions: Iterable[LaxVector],
    edge_to_superbases: Dict[TopographEdge, Tuple[SuperbaseKey, SuperbaseKey]],
    positions: Dict[SuperbaseKey, Tuple[float, float]],
) -> Dict[LaxVector, Tuple[float, float]]:
    """
    Fixed region label positions.

    For each region/lax vector, we average all visible superbase vertices
    that contain this vector. Then we apply a small radial outward shift.

    This behaves better for depth 7 than choosing the closest incident edge.
    """
    region_nodes: DefaultDict[LaxVector, List[SuperbaseKey]] = defaultdict(list)

    for a, b in edge_to_superbases.values():
        if a in positions:
            for vec in a:
                region_nodes[vec].append(a)

        if b in positions:
            for vec in b:
                region_nodes[vec].append(b)

    result: Dict[LaxVector, Tuple[float, float]] = {}

    for vector in regions:
        nodes = [n for n in region_nodes.get(vector, []) if n in positions]
        if not nodes:
            continue

        x = sum(positions[n][0] for n in nodes) / len(nodes)
        y = sum(positions[n][1] for n in nodes) / len(nodes)

        # Move labels slightly outward from the center.
        r = sqrt(x * x + y * y)
        if r > 1e-12:
            x += 0.18 * x / r
            y += 0.18 * y / r

        result[vector] = (x, y)

    return result

@lru_cache(maxsize=16)
def _build_template(display_depth: int) -> TopographTemplate:
    if display_depth <= 3:
        return _manual_compact_template_3()
    return _manual_compact_template_4()

def _label_colour(value: int, colour_by_sign: bool) -> str:
    if not colour_by_sign:
        return "#00ffff"
    if value > 0:
        return "#00ffff"
    if value < 0:
        return "#ff00ff"
    return "#00ff00"


def _edge_xy(edge_slots: Iterable[EdgeSlot], visible_edges: Set[TopographEdge]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for slot in sorted(edge_slots, key=lambda s: (s.edge.u.as_tuple(), s.edge.v.as_tuple())):
        if slot.edge not in visible_edges:
            continue
        xs.extend([slot.p0[0], slot.p1[0], None])
        ys.extend([slot.p0[1], slot.p1[1], None])
    return xs, ys


def make_classic_conway_figure(
    topograph: Topograph,
    representation_number: Optional[int] = None,
    show_vectors: bool = False,
    show_superbase_vertices: bool = False,
    colour_values_by_sign: bool = False,
    layout_mode: str = "tree",
    display_depth: int = 3,
) -> go.Figure:
    """
    Fixed-template Conway topograph rendering.

    The geometry depends only on fragment depth, not on coefficients of Q.
    """
    _ = layout_mode  # compatibility with existing UI/API

    #display_depth = max(1, min(display_depth, FIXED_TEMPLATE_MAX_DEPTH))
    template = _build_template(display_depth)
    edge_slots = template.edge_slots

    template_edges = set(edge_slots.keys())
    river_edges = (
        set(topograph.river_edges()) & template_edges
        if topograph.form.classify() == "indefinite"
        else set()
    )
    normal_edges = template_edges - river_edges

    normal_x, normal_y = _edge_xy(edge_slots.values(), normal_edges)
    river_x, river_y = _edge_xy(edge_slots.values(), river_edges)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=normal_x,
            y=normal_y,
            mode="lines",
            line=dict(color="#e8e8e8", width=1.9),
            hoverinfo="skip",
            name="hrany topografu",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=river_x,
            y=river_y,
            mode="lines",
            line=dict(color="#ff3b30", width=3.2),
            hoverinfo="skip",
            name="river",
        )
    )

    if show_superbase_vertices:
        vx = [pos[0] for pos in template.superbase_positions.values()]
        vy = [pos[1] for pos in template.superbase_positions.values()]
        hover = [
            "<b>superbáza</b><br>"
            + "<br>".join(
                f"{v.as_tuple()}: Q={topograph.regions.get(v, '∅')}" for v in node
            )
            for node in template.superbase_positions
        ]
        fig.add_trace(
            go.Scatter(
                x=vx,
                y=vy,
                mode="markers",
                marker=dict(size=5, color="#444444"),
                hoverinfo="text",
                hovertext=hover,
                name="superbázy",
            )
        )

    label_positions = {
        vector: slot.center
        for vector, slot in template.region_slots.items()
        if vector in topograph.regions
    }

    marker_size = 16
    font_size = 12 if show_vectors else 14

    well = topograph.find_well()
    well_vector = well.vector if well else None

    for vector, value in sorted(
        topograph.regions.items(), key=lambda item: (abs(item[0].m) + abs(item[0].n), item[0].m, item[0].n)
    ):
        if vector not in label_positions:
            continue
        x, y = label_positions[vector]
        label = f"{value}" if not show_vectors else f"({vector.m},{vector.n})<br><span style='font-size:8px'>{value}</span>"
        color = _label_colour(value, colour_values_by_sign)

        marker_line_color = "#00ffff"
        marker_fill = "rgba(0,0,0,0.0)"
        marker_width = 0.0
        marker_size_local = 1
        if vector == well_vector or (representation_number is not None and value == representation_number):
            marker_fill = "rgba(255,221,87,0.32)" if vector == well_vector else "rgba(105,255,135,0.28)"
            marker_line_color = "rgba(255,255,255,0.95)"
            marker_width = 1.0
            marker_size_local = max(16, marker_size - 2)

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=marker_size_local,
                    color=marker_fill,
                    line=dict(color=marker_line_color, width=marker_width),
                ),
                text=[label],
                textposition="middle center",
                textfont=dict(color=color, size=font_size),
                hoverinfo="text",
                hovertext=[
                    f"<b>región</b><br>lax vektor: {vector.as_tuple()}<br>"
                    f"Farey label: {vector.slope_label()}<br>Q(m,n): {value}"
                ],
                showlegend=False,
            )
        )

    if template.superbase_positions:
        arrow_len = 0.75
        stem_y0 = -arrow_len * 0.35
        stem_y1 = arrow_len * 0.35
        wing = arrow_len * 0.22
        fig.add_trace(
            go.Scatter(
                x=[0.0, 0.0, None, 0.0, -wing, None, 0.0, wing],
                y=[stem_y0, stem_y1, None, stem_y1, stem_y1 - wing, None, stem_y1, stem_y1 - wing],
                mode="lines",
                line=dict(color="#e8e8e8", width=1.6),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    form = topograph.form
    fig.update_layout(
        title=(
            f"Conwayho topograf: Q=[{form.A},{form.B},{form.C}], "
            f"D={form.discriminant()}, compute depth={topograph.depth}, display depth={template.depth}"
        ),
        template=None,
        height=850,
        hovermode="closest",
        dragmode="pan",
        margin=dict(l=10, r=10, t=70, b=10),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#f2f2f2"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        range=[-6.2, 6.2],
    )

    fig.update_yaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        range=[-6.0, 6.0],
        scaleanchor="x",
        scaleratio=1,
    )
    return fig

def _manual_compact_template_3() -> TopographTemplate:
    V = LaxVector

    # region vectors
    e = V(1, 0)
    f = V(0, 1)
    p = V(1, 1)
    n = V(-1, 1)

    ul = V(2, 1)
    ur = V(1, 2)
    ull = V(3, 1)
    urr = V(1, 3)
    ult = V(3, 2)
    urt = V(2, 3)

    ll = V(-2, 1)
    lr = V(-1, 2)
    lll = V(-3, 1)
    lrr = V(-1, 3)
    llb = V(-3, 2)
    lrb = V(-2, 3)

    # superbase vertices
    CT = (0.0, 0.62)
    CB = (0.0, -0.62)

    UL = (-0.78, 1.34)
    UR = (0.78, 1.34)

    ULL = (-1.78, 1.34)
    ULT = (-0.78, 2.14)
    URR = (1.78, 1.34)
    URT = (0.78, 2.14)

    LL = (-0.78, -1.34)
    LR = (0.78, -1.34)

    LLL = (-1.78, -1.34)
    LLB = (-0.78, -2.14)
    LRR = (1.78, -1.34)
    LRB = (0.78, -2.14)

    superbase_positions: Dict[SuperbaseKey, Tuple[float, float]] = {
        _superbase_key(e, f, p): CT,
        _superbase_key(e, f, n): CB,

        _superbase_key(e, p, ul): UL,
        _superbase_key(f, p, ur): UR,

        _superbase_key(e, ul, ull): ULL,
        _superbase_key(p, ul, ult): ULT,

        _superbase_key(f, ur, urr): URR,
        _superbase_key(p, ur, urt): URT,

        _superbase_key(e, n, ll): LL,
        _superbase_key(f, n, lr): LR,

        _superbase_key(e, ll, lll): LLL,
        _superbase_key(n, ll, llb): LLB,

        _superbase_key(f, lr, lrr): LRR,
        _superbase_key(n, lr, lrb): LRB,
    }

    edges = [
        _slot(e, f, CB, CT),

        _slot(e, p, CT, UL),
        _slot(f, p, CT, UR),

        _slot(e, ul, UL, ULL),
        _slot(p, ul, UL, ULT),

        _slot(f, ur, UR, URR),
        _slot(p, ur, UR, URT),

        _slot(e, n, CB, LL),
        _slot(f, n, CB, LR),

        _slot(e, ll, LL, LLL),
        _slot(n, ll, LL, LLB),

        _slot(f, lr, LR, LRR),
        _slot(n, lr, LR, LRB),
    ]

    edge_slots = {slot.edge: slot for slot in edges}

    region_slots: Dict[LaxVector, RegionSlot] = {
        e: RegionSlot(e, (-0.34, 0.02)),
        f: RegionSlot(f, (0.34, 0.02)),
        p: RegionSlot(p, (0.0, 1.10)),

        ul: RegionSlot(ul, (-1.18, 1.70)),
        ur: RegionSlot(ur, (1.18, 1.70)),

        ull: RegionSlot(ull, (-2.34, 1.36)),
        urr: RegionSlot(urr, (2.34, 1.36)),

        ult: RegionSlot(ult, (-1.18, 2.52)),
        urt: RegionSlot(urt, (1.18, 2.52)),

        n: RegionSlot(n, (0.0, -1.10)),

        ll: RegionSlot(ll, (-1.18, -1.70)),
        lr: RegionSlot(lr, (1.18, -1.70)),

        lll: RegionSlot(lll, (-2.34, -1.36)),
        lrr: RegionSlot(lrr, (2.34, -1.36)),

        llb: RegionSlot(llb, (-1.18, -2.52)),
        lrb: RegionSlot(lrb, (1.18, -2.52)),
    }

    return TopographTemplate(
        depth=3,
        superbase_positions=superbase_positions,
        region_slots=region_slots,
        edge_slots=edge_slots,
    )

def _manual_compact_template_4() -> TopographTemplate:
    """
    Manually tuned compact Conway-like template of display depth 4.

    This is a presentation template:
      - fixed geometry;
      - independent of Q;
      - values and river are filled from the actual computed topograph.
    """
    V = LaxVector

    def same_lax(a: LaxVector, b: LaxVector) -> bool:
        return a == b or (a.m == -b.m and a.n == -b.n)

    def other_region(u: LaxVector, v: LaxVector, known: LaxVector) -> LaxVector:
        """
        For an edge between regions u and v, the two adjacent superbases are
        {u,v,u+v} and {u,v,u-v}. If one third region is known, return the other.
        """
        s = add(u, v)
        d = sub(u, v)

        if same_lax(s, known):
            return d
        if same_lax(d, known):
            return s

        # Fallback should normally not be reached for valid topograph edges.
        return s

    # ------------------------------------------------------------------
    # Core region vectors
    # ------------------------------------------------------------------
    e = V(1, 0)
    f = V(0, 1)

    p = V(1, 1)
    n = V(-1, 1)

    # Upper layer
    ul = V(2, 1)
    ur = V(1, 2)

    ull = V(3, 1)
    ult = V(3, 2)
    urr = V(1, 3)
    urt = V(2, 3)

    # Lower layer
    ll = V(-2, 1)
    lr = V(-1, 2)

    lll = V(-3, 1)
    llb = V(-3, 2)
    lrr = V(-1, 3)
    lrb = V(-2, 3)

    # ------------------------------------------------------------------
    # New outer region vectors for display depth 4
    # ------------------------------------------------------------------
    ull_a = other_region(e, ull, ul)
    ull_b = other_region(ul, ull, e)

    ult_a = other_region(p, ult, ul)
    ult_b = other_region(ul, ult, p)

    urr_a = other_region(f, urr, ur)
    urr_b = other_region(ur, urr, f)

    urt_a = other_region(p, urt, ur)
    urt_b = other_region(ur, urt, p)

    lll_a = other_region(e, lll, ll)
    lll_b = other_region(ll, lll, e)

    llb_a = other_region(n, llb, ll)
    llb_b = other_region(ll, llb, n)

    lrr_a = other_region(f, lrr, lr)
    lrr_b = other_region(lr, lrr, f)

    lrb_a = other_region(n, lrb, lr)
    lrb_b = other_region(lr, lrb, n)

    # ------------------------------------------------------------------
    # Fixed superbase vertex coordinates
    # ------------------------------------------------------------------
    CT = (0.0, 0.62)
    CB = (0.0, -0.62)

    UL = (-0.78, 1.34)
    UR = (0.78, 1.34)

    ULL = (-1.78, 1.34)
    ULT = (-0.78, 2.14)

    URR = (1.78, 1.34)
    URT = (0.78, 2.14)

    LL = (-0.78, -1.34)
    LR = (0.78, -1.34)

    LLL = (-1.78, -1.34)
    LLB = (-0.78, -2.14)

    LRR = (1.78, -1.34)
    LRB = (0.78, -2.14)

    # Outer endpoints
    ULL_A = (-2.70, 2.02)
    ULL_B = (-2.70, 0.66)

    ULT_A = (-1.46, 2.90)
    ULT_B = (-0.10, 2.90)

    URR_A = (2.70, 2.02)
    URR_B = (2.70, 0.66)

    URT_A = (0.10, 2.90)
    URT_B = (1.46, 2.90)

    LLL_A = (-2.70, -0.66)
    LLL_B = (-2.70, -2.02)

    LLB_A = (-1.46, -2.90)
    LLB_B = (-0.10, -2.90)

    LRR_A = (2.70, -0.66)
    LRR_B = (2.70, -2.02)

    LRB_A = (0.10, -2.90)
    LRB_B = (1.46, -2.90)

    superbase_positions: Dict[SuperbaseKey, Tuple[float, float]] = {
        _superbase_key(e, f, p): CT,
        _superbase_key(e, f, n): CB,

        _superbase_key(e, p, ul): UL,
        _superbase_key(f, p, ur): UR,

        _superbase_key(e, ul, ull): ULL,
        _superbase_key(p, ul, ult): ULT,

        _superbase_key(f, ur, urr): URR,
        _superbase_key(p, ur, urt): URT,

        _superbase_key(e, n, ll): LL,
        _superbase_key(f, n, lr): LR,

        _superbase_key(e, ll, lll): LLL,
        _superbase_key(n, ll, llb): LLB,

        _superbase_key(f, lr, lrr): LRR,
        _superbase_key(n, lr, lrb): LRB,

        # Outer upper
        _superbase_key(e, ull, ull_a): ULL_A,
        _superbase_key(ul, ull, ull_b): ULL_B,

        _superbase_key(p, ult, ult_a): ULT_A,
        _superbase_key(ul, ult, ult_b): ULT_B,

        _superbase_key(f, urr, urr_a): URR_A,
        _superbase_key(ur, urr, urr_b): URR_B,

        _superbase_key(p, urt, urt_a): URT_A,
        _superbase_key(ur, urt, urt_b): URT_B,

        # Outer lower
        _superbase_key(e, lll, lll_a): LLL_A,
        _superbase_key(ll, lll, lll_b): LLL_B,

        _superbase_key(n, llb, llb_a): LLB_A,
        _superbase_key(ll, llb, llb_b): LLB_B,

        _superbase_key(f, lrr, lrr_a): LRR_A,
        _superbase_key(lr, lrr, lrr_b): LRR_B,

        _superbase_key(n, lrb, lrb_a): LRB_A,
        _superbase_key(lr, lrb, lrb_b): LRB_B,
    }

    # ------------------------------------------------------------------
    # Fixed edge skeleton
    # ------------------------------------------------------------------
    edges: List[EdgeSlot] = [
        # Central
        _slot(e, f, CB, CT),

        # Upper central
        _slot(e, p, CT, UL),
        _slot(f, p, CT, UR),

        # Upper left
        _slot(e, ul, UL, ULL),
        _slot(p, ul, UL, ULT),

        # Upper right
        _slot(f, ur, UR, URR),
        _slot(p, ur, UR, URT),

        # Lower central
        _slot(e, n, CB, LL),
        _slot(f, n, CB, LR),

        # Lower left
        _slot(e, ll, LL, LLL),
        _slot(n, ll, LL, LLB),

        # Lower right
        _slot(f, lr, LR, LRR),
        _slot(n, lr, LR, LRB),

        # Outer upper left
        _slot(e, ull, ULL, ULL_A),
        _slot(ul, ull, ULL, ULL_B),

        _slot(p, ult, ULT, ULT_A),
        _slot(ul, ult, ULT, ULT_B),

        # Outer upper right
        _slot(f, urr, URR, URR_A),
        _slot(ur, urr, URR, URR_B),

        _slot(p, urt, URT, URT_A),
        _slot(ur, urt, URT, URT_B),

        # Outer lower left
        _slot(e, lll, LLL, LLL_A),
        _slot(ll, lll, LLL, LLL_B),

        _slot(n, llb, LLB, LLB_A),
        _slot(ll, llb, LLB, LLB_B),

        # Outer lower right
        _slot(f, lrr, LRR, LRR_A),
        _slot(lr, lrr, LRR, LRR_B),

        _slot(n, lrb, LRB, LRB_A),
        _slot(lr, lrb, LRB, LRB_B),
    ]

    edge_slots: Dict[TopographEdge, EdgeSlot] = {
        slot.edge: slot for slot in edges
    }

    # ------------------------------------------------------------------
    # Fixed region label positions
    # ------------------------------------------------------------------
    region_slots: Dict[LaxVector, RegionSlot] = {
        e: RegionSlot(e, (-0.34, 0.03)),
        f: RegionSlot(f, (0.34, 0.03)),

        p: RegionSlot(p, (0.0, 1.10)),
        n: RegionSlot(n, (0.0, -1.10)),

        ul: RegionSlot(ul, (-1.18, 1.67)),
        ur: RegionSlot(ur, (1.18, 1.67)),

        ull: RegionSlot(ull, (-2.18, 1.34)),
        ult: RegionSlot(ult, (-0.78, 2.52)),

        urr: RegionSlot(urr, (2.18, 1.34)),
        urt: RegionSlot(urt, (0.78, 2.52)),

        ll: RegionSlot(ll, (-1.18, -1.67)),
        lr: RegionSlot(lr, (1.18, -1.67)),

        lll: RegionSlot(lll, (-2.18, -1.34)),
        llb: RegionSlot(llb, (-0.78, -2.52)),

        lrr: RegionSlot(lrr, (2.18, -1.34)),
        lrb: RegionSlot(lrb, (0.78, -2.52)),

        # Outer upper left
        ull_a: RegionSlot(ull_a, (-2.92, 2.08)),
        ull_b: RegionSlot(ull_b, (-2.92, 0.60)),

        ult_a: RegionSlot(ult_a, (-1.58, 3.12)),
        ult_b: RegionSlot(ult_b, (-0.02, 3.12)),

        # Outer upper right
        urr_a: RegionSlot(urr_a, (2.92, 2.08)),
        urr_b: RegionSlot(urr_b, (2.92, 0.60)),

        urt_a: RegionSlot(urt_a, (0.02, 3.12)),
        urt_b: RegionSlot(urt_b, (1.58, 3.12)),

        # Outer lower left
        lll_a: RegionSlot(lll_a, (-2.92, -0.60)),
        lll_b: RegionSlot(lll_b, (-2.92, -2.08)),

        llb_a: RegionSlot(llb_a, (-1.58, -3.12)),
        llb_b: RegionSlot(llb_b, (-0.02, -3.12)),

        # Outer lower right
        lrr_a: RegionSlot(lrr_a, (2.92, -0.60)),
        lrr_b: RegionSlot(lrr_b, (2.92, -2.08)),

        lrb_a: RegionSlot(lrb_a, (0.02, -3.12)),
        lrb_b: RegionSlot(lrb_b, (1.58, -3.12)),
    }

    return TopographTemplate(
        depth=4,
        superbase_positions=superbase_positions,
        region_slots=region_slots,
        edge_slots=edge_slots,
    )
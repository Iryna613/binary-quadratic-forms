from __future__ import annotations

import json
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from topograph_analyzer.classic_view import make_classic_conway_figure
from topograph_analyzer.gauss import ReductionResult, reduce_positive_definite
from topograph_analyzer.quadratic_form import QuadraticForm
from topograph_analyzer.topograph import TopographicReductionResult, topographic_reduction

st.set_page_config(page_title="Conwayho topograf", page_icon="🧮", layout="wide")

BENCHMARK_FORMS: List[Tuple[str, int, int, int]] = [
    ("x²+y²", 1, 0, 1),
    ("x²+xy+y²", 1, 1, 1),
    ("2x²+xy+2y²", 2, 1, 2),
    ("2x²+3xy+5y²", 2, 3, 5),
    ("3x²+xy+3y²", 3, 1, 3),
    ("4x²+7xy+9y²", 4, 7, 9),
    ("5x²+3xy+2y²", 5, 3, 2),
    ("3x²+5xy+7y²", 3, 5, 7),
]

EXAMPLES: Dict[str, Tuple[int, int, int, int]] = {
    "Q(x,y)=x²-2y², D=8 — river": (1, 0, -2, 5),
    "Q(x,y)=4x²+7xy+9y², D=-95": (4, 7, 9, 5),
    "Q(x,y)=x²+xy+y², D=-3": (1, 1, 1, 5),
    "Q(x,y)=x²+2xy+y², D=0": (1, 2, 1, 4),
    "Custom": (1, 0, -2, 5),
}


@st.cache_data(show_spinner=False)
def compute(A: int, B: int, C: int, depth: int, rep_enabled: bool, rep_n: int, rep_bound: int) -> Dict[str, Any]:
    form = QuadraticForm(A, B, C)

    t0 = time.perf_counter()
    topo = topographic_reduction(form, depth=depth)
    topo_time_ms = (time.perf_counter() - t0) * 1000

    gauss: Optional[ReductionResult] = None
    gauss_error: Optional[str] = None
    gauss_time_ms: Optional[float] = None
    if form.classify() == "positive definite":
        try:
            t0 = time.perf_counter()
            gauss = reduce_positive_definite(form)
            gauss_time_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            gauss_error = str(exc)

    reps = form.find_representations(rep_n, rep_bound) if rep_enabled else None
    return {
        "form": form,
        "topo": topo,
        "gauss": gauss,
        "gauss_error": gauss_error,
        "representations": reps,
        "gauss_time_ms": gauss_time_ms,
        "topo_time_ms": topo_time_ms,
    }


def fig_download_buttons(fig: go.Figure, stem: str, key: str) -> None:
    """Render HTML + PDF download buttons side by side for a Plotly figure."""
    fig_white = go.Figure(fig)
    fig_white.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="black")
    c1, c2 = st.columns(2)
    c1.download_button(
        "Stiahnuť ako HTML",
        data=fig.to_html(include_plotlyjs="cdn", full_html=True),
        file_name=f"{stem}.html",
        mime="text/html",
        key=f"{key}_html",
    )
    c2.download_button(
        "Stiahnuť ako PNG",
        data=pio.to_image(fig_white, format="png", width=1200, height=fig.layout.height or 500, scale=2),
        file_name=f"{stem}.png",
        mime="image/png",
        key=f"{key}_pdf",
    )


def matrix_to_list(m):
    return [[m[0][0], m[0][1]], [m[1][0], m[1][1]]]


def analysis_json(form: QuadraticForm, topo: TopographicReductionResult, gauss: Optional[ReductionResult], reps, rep_n, rep_bound) -> Dict[str, Any]:
    t = topo.topograph
    data: Dict[str, Any] = {
        "input_form": list(form.as_tuple()),
        "discriminant": form.discriminant(),
        "classification": form.classify(),
        "primitive": form.is_primitive(),
        "topograph": {
            "depth": t.depth,
            "regions": len(t.regions),
            "edges": len(t.edges),
            "diamonds": len(t.diamonds),
            "river_edges": len(t.river_edges()) if form.classify() == "indefinite" else 0,
            "checks": t.validate(),
        },
    }
    if gauss:
        data["gauss_reduction"] = {
            "reduced_form": list(gauss.reduced_form.as_tuple()),
            "transformation_matrix": matrix_to_list(gauss.transformation_matrix),
            "steps": [
                {
                    "before": list(s.form_before),
                    "operation": s.operation,
                    "matrix": matrix_to_list(s.matrix),
                    "after": list(s.form_after),
                }
                for s in gauss.steps
            ],
        }
    if topo.well:
        data["well"] = {"vector": list(topo.well.vector.as_tuple()), "value": topo.well.value}
    if topo.selected:
        data["topographic_reduction"] = {
            "reduced_form": list(topo.selected.form.as_tuple()),
            "basis_matrix": matrix_to_list(topo.selected.basis_matrix),
        }
    if gauss and topo.selected:
        data["comparison"] = {"same_reduced_form": gauss.reduced_form.as_tuple() == topo.selected.form.as_tuple()}
    if reps is not None:
        data["representations"] = {"number": rep_n, "bound": rep_bound, "solutions": [list(r) for r in reps]}
    return data


def _time_calls(fn, repeats: int) -> List[float]:
    return [(lambda t0: (fn(), (time.perf_counter() - t0) * 1000)[1])(time.perf_counter()) for _ in range(repeats)]


def run_form_benchmark(depth: int = 5, repeats: int = 100) -> pd.DataFrame:
    rows = []
    for label, a, b, c in BENCHMARK_FORMS:
        f = QuadraticForm(a, b, c)
        if f.classify() != "positive definite":
            continue
        g_times = _time_calls(lambda f=f: reduce_positive_definite(f), repeats)
        t_times = _time_calls(lambda f=f, d=depth: topographic_reduction(f, depth=d), repeats)
        rows.append({
            "Forma": label,
            "Gauss mean (ms)": round(statistics.mean(g_times), 5),
            "Gauss std (ms)": round(statistics.stdev(g_times), 5),
            "Topograf mean (ms)": round(statistics.mean(t_times), 5),
            "Topograf std (ms)": round(statistics.stdev(t_times), 5),
            "Pomer": round(statistics.mean(t_times) / statistics.mean(g_times), 1) if statistics.mean(g_times) > 0 else 0,
            "Gauss kroky": len(reduce_positive_definite(f).steps),
            "Topograf regióny": len(topographic_reduction(f, depth=depth).topograph.regions),
        })
    return pd.DataFrame(rows)


def run_depth_benchmark(form: QuadraticForm, depths: List[int], repeats: int = 100) -> pd.DataFrame:
    rows = []
    for d in depths:
        t_times = _time_calls(lambda d=d: topographic_reduction(form, depth=d), repeats)
        topo = topographic_reduction(form, depth=d)
        rows.append({
            "Hĺbka": d,
            "Topograf mean (ms)": round(statistics.mean(t_times), 5),
            "Topograf std (ms)": round(statistics.stdev(t_times), 5),
            "Regióny": len(topo.topograph.regions),
            "Hrany": len(topo.topograph.edges),
        })
    return pd.DataFrame(rows)


def regions_df(topo: TopographicReductionResult) -> pd.DataFrame:
    t = topo.topograph
    rows = []
    well_v = topo.well.vector if topo.well else None
    for v, value in sorted(t.regions.items(), key=lambda item: (item[1], item[0].m, item[0].n)):
        rows.append({"vektor": v.as_tuple(), "Farey": v.slope_label(), "Q(m,n)": value, "well": v == well_v})
    return pd.DataFrame(rows)


def main() -> None:
    st.title("Interaktívny Conwayho topograf")
    st.caption(
        "Klasický regiónový pohľad: hodnoty sú umiestnené v regiónoch a river je zobrazená ako hranica medzi regiónmi opačného znamienka."
    )

    with st.sidebar:
        st.header("Parametre")
        ex = st.selectbox("Príklad", list(EXAMPLES.keys()), index=0)
        dA, dB, dC, dd = EXAMPLES[ex]
        if st.button("Načítať príklad"):
            st.session_state["A_input"] = dA
            st.session_state["B_input"] = dB
            st.session_state["C_input"] = dC
            st.session_state["depth_input"] = dd
        A = st.number_input("A", value=dA, step=1, format="%d", key="A_input")
        B = st.number_input("B", value=dB, step=1, format="%d", key="B_input")
        C = st.number_input("C", value=dC, step=1, format="%d", key="C_input")
        compute_depth = st.slider("Hĺbka výpočtu", 1, 7, dd, key="depth_input")
        display_depth = st.slider("Hĺbka zobrazenia", 3, 4, 3)

        st.divider()
        st.header("Zobrazenie")
        st.caption("Geometria topografu je fixný kanonický template pre zvolenú hĺbku.")
        show_vectors = st.checkbox("Do regiónov pridať aj lax vektory", value=False)
        show_superbases = st.checkbox("Zobraziť pomocné vrcholy / superbázy", value=False)
        colour_values = st.checkbox("Zafarbiť hodnoty podľa znamienka", value=False)

        st.divider()
        st.header("Reprezentácie")
        rep_enabled = st.checkbox("Hľadať Q(x,y)=N", value=False)
        rep_n = st.number_input("N", value=5, step=1, format="%d", disabled=not rep_enabled)
        rep_bound = st.slider("Bound", 1, 50, 10, disabled=not rep_enabled)

    try:
        with st.spinner("Generujem topograf..."):
            result = compute(
                int(A),
                int(B),
                int(C),
                int(compute_depth),
                bool(rep_enabled),
                int(rep_n),
                int(rep_bound),
            )
    except Exception as exc:
        st.error(f"Výpočet zlyhal: {exc}")
        return

    form: QuadraticForm = result["form"]
    topo: TopographicReductionResult = result["topo"]
    gauss: Optional[ReductionResult] = result["gauss"]
    gauss_error: Optional[str] = result.get("gauss_error")
    reps = result["representations"]
    gauss_time_ms: Optional[float] = result.get("gauss_time_ms")
    topo_time_ms: float = result["topo_time_ms"]

    st.subheader(f"Q(x,y) = {form.A}x² + {form.B}xy + {form.C}y²")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("D", form.discriminant())
    c2.metric("Typ", form.classify())
    c3.metric("Primitive", "áno" if form.is_primitive() else "nie")
    c4.metric("Regióny", len(topo.topograph.regions))
    c5.metric("Hrany", len(topo.topograph.edges))
    c6.metric("River", len(topo.topograph.river_edges()) if form.classify() == "indefinite" else 0)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Topograf", "Redukcia", "Kontroly", "Regióny", "JSON export", "Benchmark"])

    with tab1:
        fig = make_classic_conway_figure(
            topo.topograph,
            representation_number=int(rep_n) if rep_enabled else None,
            show_vectors=show_vectors,
            show_superbase_vertices=show_superbases,
            colour_values_by_sign=colour_values,
            display_depth=int(display_depth),
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
        fig_download_buttons(fig, f"classic_topograph_Q_{form.A}_{form.B}_{form.C}", "topo_fig")
        if form.classify() == "indefinite":
            st.info("Červené hrany sú river: hranice medzi regiónmi s kladnou a zápornou hodnotou.")
        elif form.classify() == "positive definite":
            st.info("Pri pozitívne definitnej forme sa sleduje well. River nie je redukčný objekt tejto triedy foriem.")

    with tab2:
        left, right = st.columns(2)
        with left:
            st.markdown("### Gaussova redukcia")
            if gauss:
                st.success(f"Redukovaná forma: {gauss.reduced_form.as_tuple()}")
                if gauss.steps:
                    st.dataframe(pd.DataFrame([s.__dict__ for s in gauss.steps]), use_container_width=True)
                else:
                    st.write("Forma je už redukovaná.")
            elif gauss_error:
                st.warning(f"Gaussova redukcia zlyhala: {gauss_error}")
            else:
                st.write("Gaussova redukcia je použitá iba pre pozitívne definitné formy.")
        with right:
            st.markdown("### Topografická redukcia")
            if topo.selected:
                st.success(f"Kandidát: {topo.selected.form.as_tuple()}")
                st.write(f"Bázová matica: `{topo.selected.basis_matrix}`")
            elif form.classify() == "positive definite":
                st.warning("Kandidát sa v tomto fragmente nenašiel. Zvýš depth.")
            else:
                st.write("Pre neurčité formy sa sleduje river, nie well/kandidát pozitívnej redukcie.")
        if gauss and topo.selected:
            same = gauss.reduced_form.as_tuple() == topo.selected.form.as_tuple()
            if same:
                st.success("Gaussova a topografická redukcia sa zhodujú.")
            else:
                st.error("Výsledky sú ekvivalentné, líšia sa iba znamienkom koeficientu.")
        if topo.well:
            st.write(f"Well: vektor `{topo.well.vector.as_tuple()}`, hodnota `{topo.well.value}`")

        if gauss_time_ms is not None:
            st.divider()
            st.markdown("### Porovnanie rýchlosti")
            st.caption("Časy sú merané pri prvom výpočte (výsledky sú potom kešované).")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Gaussova redukcia", f"{gauss_time_ms:.4f} ms", help="Čas behu algoritmu Gaussovej redukcie")
            sc2.metric("Topografická redukcia", f"{topo_time_ms:.4f} ms", help=f"Čas behu topografickej redukcie (depth={int(compute_depth)})")
            ratio = topo_time_ms / gauss_time_ms if gauss_time_ms > 0 else float("inf")
            if ratio >= 1:
                sc3.metric("Pomer (topograf / Gauss)", f"{ratio:.1f}×", help="Koľkokrát je topografická redukcia pomalšia ako Gaussova")
            else:
                sc3.metric("Pomer (Gauss / topograf)", f"{1/ratio:.1f}×", help="Koľkokrát je Gaussova redukcia pomalšia ako topografická")

            speed_data = pd.DataFrame(
                {
                    "Algoritmus": ["Gaussova redukcia", f"Topografická redukcia (depth={int(compute_depth)})"],
                    "Čas (ms)": [round(gauss_time_ms, 4), round(topo_time_ms, 4)],
                    "Kroky / uzly": [len(gauss.steps), len(topo.topograph.regions)],
                }
            )
            st.dataframe(speed_data, use_container_width=True, hide_index=True)

            fig_bar = go.Figure(go.Bar(
                x=["Gaussova redukcia", f"Topografická redukcia\n(depth={int(compute_depth)})"],
                y=[gauss_time_ms, topo_time_ms],
                marker_color=["#2196F3", "#FF5722"],
                text=[f"{gauss_time_ms:.4f} ms", f"{topo_time_ms:.4f} ms"],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title=f"Porovnanie rýchlosti — Q({form.A},{form.B},{form.C})",
                yaxis_title="Čas (ms)",
                yaxis_type="log",
                template="simple_white",
                showlegend=False,
                height=380,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            fig_download_buttons(fig_bar, f"speed_comparison_Q_{form.A}_{form.B}_{form.C}", "speed_bar")

    with tab3:
        checks = topo.topograph.validate()
        st.dataframe(pd.DataFrame([{"kontrola": k, "výsledok": "OK" if v else "FAIL"} for k, v in checks.items()]), use_container_width=True, hide_index=True)
        st.write(f"Diamond rule valid: {sum(d.valid for d in topo.topograph.diamonds)} / {len(topo.topograph.diamonds)}")

    with tab4:
        st.dataframe(regions_df(topo), use_container_width=True, hide_index=True)
        if rep_enabled:
            st.markdown(f"### Riešenia Q(x,y)={int(rep_n)}")
            if reps:
                st.dataframe(pd.DataFrame([{"x": x, "y": y} for x, y in reps]), use_container_width=True, hide_index=True)
            else:
                st.warning("V zadanom ohraničení sa nenašla reprezentácia.")

    with tab5:
        data = analysis_json(form, topo, gauss, reps, int(rep_n) if rep_enabled else None, int(rep_bound))
        txt = json.dumps(data, ensure_ascii=False, indent=2)
        st.download_button("Stiahnuť JSON", txt, file_name=f"analysis_Q_{form.A}_{form.B}_{form.C}.json", mime="application/json")
        st.json(data)

    with tab6:
        st.markdown("### Benchmark rýchlosti algoritmov redukcie")
        st.caption(
            "Každý algoritmus sa spustí 100-krát a vypočíta sa priemer ± štandardná odchýlka. "
            "Výsledky sú vhodné pre použitie v bakalárskej práci."
        )

        if form.classify() != "positive definite":
            st.info("Benchmark porovnáva Gaussovu a topografickú redukciu — dostupný iba pre pozitívne definitné formy.")
        else:
            b_repeats = st.slider("Počet opakovaní (repeats)", 20, 500, 100, step=20, key="bench_repeats")
            b_depth = st.slider("Hĺbka topografu (benchmark)", 1, 7, 5, key="bench_depth")

            if st.button("Spustiť benchmark", type="primary"):
                with st.spinner(f"Meriam časy ({b_repeats} opakovaní)…"):
                    st.session_state["bench_form_df"] = run_form_benchmark(depth=b_depth, repeats=b_repeats)
                    st.session_state["bench_depth_df"] = run_depth_benchmark(form, list(range(1, 8)), repeats=b_repeats)
                    st.session_state["bench_params"] = (b_depth, b_repeats, form.as_tuple())

            if "bench_form_df" in st.session_state:
                bd, br, bf = st.session_state["bench_params"]
                st.caption(f"Posledný beh: {br} opakovaní, depth={bd}, aktuálna forma = {bf}")

                # ── Graf 1: skupinový bar chart naprieč formami ──────────────
                st.markdown("#### Porovnanie naprieč formami (depth={})".format(bd))
                df_forms: pd.DataFrame = st.session_state["bench_form_df"]

                fig_forms = go.Figure()
                fig_forms.add_trace(go.Bar(
                    name="Gaussova redukcia",
                    x=df_forms["Forma"],
                    y=df_forms["Gauss mean (ms)"],
                    error_y={"type": "data", "array": df_forms["Gauss std (ms)"].tolist()},
                    marker_color="#2196F3",
                ))
                fig_forms.add_trace(go.Bar(
                    name=f"Topografická redukcia (depth={bd})",
                    x=df_forms["Forma"],
                    y=df_forms["Topograf mean (ms)"],
                    error_y={"type": "data", "array": df_forms["Topograf std (ms)"].tolist()},
                    marker_color="#FF5722",
                ))
                fig_forms.update_layout(
                    barmode="group",
                    title=f"Priemerný čas redukcie (n={br})",
                    xaxis_title="Kvadratická forma",
                    yaxis_title="Čas (ms)",
                    yaxis_type="log",
                    template="simple_white",
                    legend={"orientation": "h", "y": -0.25},
                    height=450,
                )
                st.plotly_chart(fig_forms, use_container_width=True)
                fig_download_buttons(fig_forms, "benchmark_forms", "dl_forms")

                # ── Graf 2: škálovanie s hĺbkou ──────────────────────────────
                st.markdown(f"#### Škálovanie topografickej redukcie s hĺbkou — Q{bf}")
                df_depth: pd.DataFrame = st.session_state["bench_depth_df"]

                gauss_ref = df_forms.loc[df_forms["Forma"] == f"{bf[0]}x²+{bf[1]}xy+{bf[2]}y²", "Gauss mean (ms)"]
                fig_depth = go.Figure()
                fig_depth.add_trace(go.Scatter(
                    x=df_depth["Hĺbka"],
                    y=df_depth["Topograf mean (ms)"],
                    error_y={"type": "data", "array": df_depth["Topograf std (ms)"].tolist()},
                    mode="lines+markers",
                    name="Topografická redukcia",
                    marker_color="#FF5722",
                    line={"width": 2},
                ))
                if gauss_time_ms is not None:
                    fig_depth.add_hline(
                        y=gauss_time_ms,
                        line_dash="dash",
                        line_color="#2196F3",
                        annotation_text=f"Gaussova redukcia ({gauss_time_ms:.4f} ms)",
                        annotation_position="top left",
                    )
                fig_depth.update_layout(
                    title=f"Čas topografickej redukcie vs. hĺbka (n={br})",
                    xaxis_title="Hĺbka (depth)",
                    yaxis_title="Čas (ms)",
                    xaxis={"tickmode": "linear", "dtick": 1},
                    template="simple_white",
                    height=420,
                )
                st.plotly_chart(fig_depth, use_container_width=True)
                fig_download_buttons(fig_depth, "benchmark_depth_scaling", "dl_depth")

                # ── Štatistická tabuľka ───────────────────────────────────────
                st.markdown("#### Štatistická tabuľka")
                st.dataframe(
                    df_forms[["Forma", "Gauss mean (ms)", "Gauss std (ms)", "Topograf mean (ms)", "Topograf std (ms)", "Pomer", "Gauss kroky", "Topograf regióny"]],
                    use_container_width=True,
                    hide_index=True,
                )
                csv = df_forms.to_csv(index=False).encode()
                st.download_button("Stiahnuť tabuľku ako CSV", csv, file_name="benchmark_stats.csv", mime="text/csv", key="dl_csv")

                st.markdown("#### Škálovanie s hĺbkou — tabuľka")
                st.dataframe(df_depth, use_container_width=True, hide_index=True)
                csv2 = df_depth.to_csv(index=False).encode()
                st.download_button("Stiahnuť CSV", csv2, file_name="benchmark_depth.csv", mime="text/csv", key="dl_csv2")


if __name__ == "__main__":
    main()

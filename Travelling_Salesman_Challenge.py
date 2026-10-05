"""
Travelling_Salesman_Challenge.py - Streamlit Web Application
Advanced Operations Research (AOR) TSP Challenge.

Modes:
1. Official Faculty Benchmark Mode:
   - Evaluates all 10 TSPLIB benchmark instances with fixed faculty protocol.
   - Initial solution: Nearest Neighbor (City 1).
   - Local search: Variable Neighborhood Descent (2-opt, Relocate, Swap) with O(1) incremental evaluation.
   - Compares with verified TSPLIB optimal values.
   - Strict tour validity checks (no duplicates, all cities visited, cyclic closure).
   - Generates downloadable CSV report for faculty submission.
2. Interactive Experimental Studio:
   - Single-instance testing and custom .tsp file upload.
   - Toggle individual operators (2-opt, Relocate, Swap), VND, or Multi-start Restarts.
   - Interactive Plotly route map visualization.
   - Convergence trajectory plots.
3. Methodology & Mathematical Documentation:
   - Full AOR course theory, mathematical formulations, and incremental delta equations.
"""

import os
import glob
import time
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import altair as alt

from tsp_solver import (
    TSPLIBParser,
    compute_distance_matrix,
    validate_tour,
    calculate_tour_distance,
    nearest_neighbor_tour,
    best_nearest_neighbor_tour,
    local_search_2opt,
    local_search_relocate,
    local_search_swap,
    variable_neighborhood_descent,
    multi_start_local_search,
    solve_faculty_benchmark,
    VERIFIED_OPTIMAL_SOLUTIONS,
)

# Set page configuration
st.set_page_config(
    page_title="TSP Optimization Studio - AOR Challenge",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Determine base directory robustly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Custom CSS for polished styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)


def plot_tour_plotly(coords, tour, title="TSP Tour Visualization"):
    """Creates an interactive Plotly route map."""
    n = len(tour)
    closed_tour = list(tour) + [tour[0]]
    x_coords = [coords[i][0] for i in closed_tour]
    y_coords = [coords[i][1] for i in closed_tour]

    fig = go.Figure()

    # Route edges
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='lines',
        line=dict(color='#1E88E5', width=2),
        name='Tour Route',
        hoverinfo='none'
    ))

    # Cities (scatter markers)
    all_x = [coords[i][0] for i in range(len(coords))]
    all_y = [coords[i][1] for i in range(len(coords))]
    hover_text = [f"City {i+1} (Index {i})<br>X: {coords[i][0]:.1f}, Y: {coords[i][1]:.1f}" for i in range(len(coords))]

    fig.add_trace(go.Scatter(
        x=all_x,
        y=all_y,
        mode='markers',
        marker=dict(size=6, color='#D32F2F', symbol='circle'),
        text=hover_text,
        hoverinfo='text',
        name='Cities'
    ))

    # Highlight start city (City 1 / tour[0])
    start_city = tour[0]
    fig.add_trace(go.Scatter(
        x=[coords[start_city][0]],
        y=[coords[start_city][1]],
        mode='markers',
        marker=dict(size=14, color='#2E7D32', symbol='star', line=dict(width=2, color='white')),
        text=[f"Start City: {start_city+1}"],
        hoverinfo='text',
        name='Start City'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="X Coordinate", showgrid=True, zeroline=False),
        yaxis=dict(title="Y Coordinate", showgrid=True, zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=550
    )
    return fig


def plot_convergence_chart(history, initial_dist, title="Convergence History"):
    """Plots distance reduction curve using Altair."""
    df = pd.DataFrame({
        "Iteration": list(range(len(history))),
        "Distance (km)": history
    })
    chart = alt.Chart(df).mark_line(point=True, color="#D32F2F").encode(
        x=alt.X("Iteration:Q", title="Search Improvement Step"),
        y=alt.Y("Distance (km):Q", scale=alt.Scale(zero=False), title="Tour Length (km)"),
        tooltip=["Iteration", "Distance (km)"]
    ).properties(
        title=title,
        height=280
    ).interactive()
    return chart


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------

st.title("🗺️ TSP Optimization Studio & Benchmark Runner")
st.caption("Advanced Operations Research (AOR) Challenge • Python & Streamlit Application")

# Sidebar navigation / Global info
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("Custom Optimization Engine: Active")
    st.info("Optimization Libraries: None (Pure Custom Implementation)")
    st.markdown("---")
    st.subheader("📚 Course Coverage")
    st.markdown("""
    - **Nearest Neighbor** (Construction)
    - **2-Opt Edge Reversal** ($O(1)$ Delta)
    - **Relocate / Insertion** ($O(1)$ Delta)
    - **Swap / Exchange** ($O(1)$ Delta)
    - **Variable Neighborhood Descent**
    - **Random Restarts / Multi-start**
    - **Strict Tour Validity Invariants**
    """)
    st.markdown("---")
    st.caption("Streamlit Community Cloud Ready • TSPLIB EUC_2D Metric")

tab_benchmark, tab_studio, tab_docs = st.tabs([
    "🏆 Official Faculty Benchmark (10 Instances)",
    "🔬 Interactive Experimental Studio",
    "📖 Methodology & Mathematical Documentation"
])

# =============================================================================
# TAB 1: OFFICIAL FACULTY BENCHMARK MODE
# =============================================================================
with tab_benchmark:
    st.header("🏆 Official Faculty Benchmark Evaluation")
    st.markdown("""
    This mode enforces the **official faculty submission protocol**:
    1. **Initial Solution**: Constructed using **Nearest Neighbor starting at City 1** (TSPLIB Node 1).
    2. **Optimization**: Improved via **Variable Neighborhood Descent (2-opt, Relocate, Swap)** with $O(1)$ incremental delta evaluation.
    3. **Distance Reporting**: Reported in **km** using the official TSPLIB EUC_2D integer metric: $\\text{int}(\\text{round}(\\sqrt{\\Delta x^2 + \\Delta y^2}))$.
    4. **Validity Verification**: Every tour strictly verifies **zero duplicates, full $0..N-1$ city coverage, and cyclic return-to-start closure**.
    5. **Optimality Gap**: Compared against official verified TSPLIB literature optima.
    """)

    available_files = sorted(glob.glob(os.path.join(BASE_DIR, "*.tsp")))
    file_basenames = [os.path.basename(f) for f in available_files]
    st.write(f"**Found {len(available_files)} benchmark files:** `{', '.join([f.replace('.tsp', '') for f in file_basenames])}`")

    col_btn, col_time = st.columns([2, 3])
    with col_btn:
        run_all_btn = st.button("🚀 Run All 10 TSPLIB Benchmarks", type="primary", use_container_width=True)
    with col_time:
        per_instance_timeout = st.slider("Max Runtime Per Instance (seconds)", min_value=3, max_value=30, value=8, step=1)

    if run_all_btn or "benchmark_results" in st.session_state:
        if run_all_btn:
            results_list = []
            progress_bar = st.progress(0, text="Initializing benchmark run...")
            status_text = st.empty()

            for idx, file_path in enumerate(available_files):
                inst_name = os.path.basename(file_path)
                status_text.text(f"Solving {inst_name} ({idx+1}/{len(available_files)})...")
                progress_bar.progress(idx / len(available_files), text=f"Solving {inst_name}...")

                parser = TSPLIBParser().parse_file(file_path)
                res = solve_faculty_benchmark(parser.coordinates, inst_name, time_limit_sec=float(per_instance_timeout))
                results_list.append(res)

            progress_bar.progress(1.0, text="Benchmark completed!")
            status_text.empty()
            st.session_state["benchmark_results"] = results_list

        results = st.session_state["benchmark_results"]

        # Convert to display DataFrame
        df_rows = []
        for r in results:
            df_rows.append({
                "Instance": r["instance"],
                "Cities (N)": r["dimension"],
                "Initial NN (km)": f"{r['initial_distance_km']:,}",
                "Final Improved (km)": f"{r['final_distance_km']:,}",
                "Verified Optimal (km)": f"{r['optimal_reference_km']:,}" if r['optimal_reference_km'] else "N/A",
                "Improvement (km)": f"{r['improvement_km']:,}",
                "Improvement (%)": f"{r['improvement_pct']:.2f}%",
                "Optimality Gap (%)": f"+{r['gap_pct']:.2f}%" if r['gap_pct'] is not None else "N/A",
                "Runtime (s)": f"{r['runtime_sec']:.2f}s",
                "Tour Validity": "✅ PASSED" if r["validity"] == "PASSED" else "❌ FAILED"
            })

        df_display = pd.DataFrame(df_rows)

        # KPI metric cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        avg_imp = np.mean([r["improvement_pct"] for r in results])
        avg_gap = np.mean([r["gap_pct"] for r in results if r["gap_pct"] is not None])
        total_time = sum([r["runtime_sec"] for r in results])

        with kpi1:
            st.metric("Total Instances Solved", len(results))
        with kpi2:
            st.metric("Avg Distance Improvement", f"{avg_imp:.2f}%")
        with kpi3:
            st.metric("Avg Optimality Gap", f"+{avg_gap:.2f}%")
        with kpi4:
            st.metric("Total Benchmark Time", f"{total_time:.2f} s")

        st.subheader("📋 Verified Benchmark Summary Table")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Faculty CSV Export
        csv_buffer = io.StringIO()
        df_export = pd.DataFrame([{
            "Instance": r["instance"],
            "Cities_N": r["dimension"],
            "Initial_NN_km": r["initial_distance_km"],
            "Final_Improved_km": r["final_distance_km"],
            "Verified_Optimal_km": r["optimal_reference_km"],
            "Improvement_km": r["improvement_km"],
            "Improvement_pct": r["improvement_pct"],
            "Optimality_Gap_pct": r["gap_pct"],
            "Runtime_seconds": r["runtime_sec"],
            "Validity": r["validity"]
        } for r in results])
        df_export.to_csv(csv_buffer, index=False)

        st.download_button(
            label="📥 Download Faculty Benchmark Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name="tsp_faculty_benchmark_report.csv",
            mime="text/csv",
            type="primary"
        )

        # Comparison Bar Chart
        st.subheader("📊 Initial NN vs. Final Improved vs. Verified Optimal")
        chart_data = []
        for r in results:
            chart_data.append({"Instance": r["instance"].replace(".tsp", ""), "Type": "1. Initial NN", "Distance (km)": r["initial_distance_km"]})
            chart_data.append({"Instance": r["instance"].replace(".tsp", ""), "Type": "2. Final Improved", "Distance (km)": r["final_distance_km"]})
            if r["optimal_reference_km"]:
                chart_data.append({"Instance": r["instance"].replace(".tsp", ""), "Type": "3. Verified Optimal", "Distance (km)": r["optimal_reference_km"]})

        df_chart = pd.DataFrame(chart_data)
        bar_chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X("Type:N", title=None),
            y=alt.Y("Distance (km):Q", title="Distance (km)"),
            color=alt.Color("Type:N", scale=alt.Scale(range=["#FFA726", "#1E88E5", "#43A047"])),
            column=alt.Column("Instance:N", title="Benchmark Instance")
        ).properties(height=300)
        st.altair_chart(bar_chart, use_container_width=True)

# =============================================================================
# TAB 2: INTERACTIVE EXPERIMENTAL STUDIO
# =============================================================================
with tab_studio:
    st.header("🔬 Interactive Experimental Studio")
    st.markdown("Explore individual instances, test specific neighborhood search operators, run multi-start restarts, and visualize route maps.")

    col_setup1, col_setup2, col_setup3 = st.columns([2, 2, 2])

    with col_setup1:
        input_source = st.radio("Instance Source", ["Preloaded Benchmarks", "Upload Custom .tsp"], horizontal=True)
        if input_source == "Preloaded Benchmarks":
            selected_basename = st.selectbox("Select TSPLIB File", file_basenames, index=0)
            selected_file = os.path.join(BASE_DIR, selected_basename)
            parser = TSPLIBParser().parse_file(selected_file)
            inst_display_name = selected_basename
        else:
            uploaded_file = st.file_uploader("Upload .tsp File", type=["tsp", "txt"])
            if uploaded_file is not None:
                content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                parser = TSPLIBParser().parse_string(content)
                inst_display_name = uploaded_file.name
            else:
                st.info("Please upload a .tsp file to begin.")
                parser = None
                inst_display_name = ""

    with col_setup2:
        start_city_mode = st.selectbox(
            "Nearest Neighbor Starting City",
            ["City 1 (TSPLIB Node 1, Default)", "Best of All Cities (Multi-Start NN)", "Custom City ID"]
        )
        custom_city_val = 1
        if start_city_mode == "Custom City ID" and parser and parser.dimension > 0:
            custom_city_val = st.number_input("Custom City ID", min_value=1, max_value=parser.dimension, value=1)

    with col_setup3:
        operator_choice = st.selectbox(
            "Optimization Search Operator",
            [
                "Variable Neighborhood Descent (2-opt + Relocate + Swap)",
                "Pure 2-Opt",
                "Pure Relocate (Insertion)",
                "Pure Swap (Exchange)",
                "Multi-Start Restarts (Escaping Local Minima)"
            ]
        )
        if operator_choice == "Multi-Start Restarts (Escaping Local Minima)":
            num_restarts = st.slider("Number of Restarts", min_value=2, max_value=15, value=5)
        else:
            num_restarts = 1

    if parser and len(parser.coordinates) > 0:
        st.caption(f"**Loaded Instance**: `{inst_display_name}` • **Dimension**: `{parser.dimension}` cities • **Edge Weight**: `{parser.edge_weight_type}`")

        opt_col1, opt_col2 = st.columns([1, 3])
        with opt_col1:
            max_iter = st.number_input("Max Iterations", min_value=10, max_value=2000, value=500, step=50)
            time_limit = st.slider("Timeout (seconds)", min_value=2, max_value=60, value=15, step=1)
            run_single_btn = st.button("⚡ Run Optimization", type="primary", use_container_width=True)

        if run_single_btn:
            coords = parser.coordinates
            dist_matrix = compute_distance_matrix(coords)
            n_cities = len(coords)

            with st.spinner("Generating starting tour via Nearest Neighbor..."):
                t_start = time.perf_counter()

                # Phase 1: Construction
                if start_city_mode == "Best of All Cities (Multi-Start NN)":
                    init_tour, init_dist, best_start = best_nearest_neighbor_tour(dist_matrix)
                elif start_city_mode == "Custom City ID":
                    init_tour, init_dist = nearest_neighbor_tour(dist_matrix, start_city=int(custom_city_val - 1))
                else:
                    init_tour, init_dist = nearest_neighbor_tour(dist_matrix, start_city=0)

                validate_tour(init_tour, n_cities)

            with st.spinner(f"Running {operator_choice}..."):
                # Phase 2: Improvement
                if operator_choice == "Pure 2-Opt":
                    res = local_search_2opt(init_tour, dist_matrix, max_iterations=int(max_iter), time_limit_sec=float(time_limit))
                elif operator_choice == "Pure Relocate (Insertion)":
                    res = local_search_relocate(init_tour, dist_matrix, max_iterations=int(max_iter), time_limit_sec=float(time_limit))
                elif operator_choice == "Pure Swap (Exchange)":
                    res = local_search_swap(init_tour, dist_matrix, max_iterations=int(max_iter), time_limit_sec=float(time_limit))
                elif operator_choice == "Multi-Start Restarts (Escaping Local Minima)":
                    res = multi_start_local_search(dist_matrix, num_starts=int(num_restarts), method="vnd", time_limit_sec=float(time_limit))
                else:
                    res = variable_neighborhood_descent(init_tour, dist_matrix, max_passes=int(max_iter), time_limit_sec=float(time_limit))

                final_tour = res["tour"]
                final_dist = res["distance"]
                validate_tour(final_tour, n_cities)
                total_runtime = time.perf_counter() - t_start

            # Metrics
            base_name = inst_display_name.replace(".tsp", "").lower()
            opt_ref = VERIFIED_OPTIMAL_SOLUTIONS.get(base_name, None)
            imp_km = init_dist - final_dist
            imp_pct = (imp_km / init_dist * 100.0) if init_dist > 0 else 0.0
            gap_pct = ((final_dist - opt_ref) / opt_ref * 100.0) if opt_ref else None

            st.success("Optimization completed with all tour validity invariants verified!")

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Initial NN Distance", f"{init_dist:,} km")
            with m2:
                st.metric("Final Tour Distance", f"{final_dist:,} km", delta=f"-{imp_km:,} km", delta_color="inverse")
            with m3:
                st.metric("Improvement", f"{imp_pct:.2f}%")
            with m4:
                st.metric("Optimality Gap", f"+{gap_pct:.2f}%" if gap_pct else "N/A (Custom)")
            with m5:
                st.metric("Runtime", f"{total_runtime:.2f} s")

            # Route Visualizations
            tab_view1, tab_view2, tab_view3 = st.tabs(["🗺️ Final Tour Map", "🔄 Initial vs Final Comparison", "📈 Convergence Curve"])

            with tab_view1:
                fig_final = plot_tour_plotly(coords, final_tour, title=f"Optimized Tour: {inst_display_name} ({final_dist:,} km)")
                st.plotly_chart(fig_final, use_container_width=True)

            with tab_view2:
                c_map1, c_map2 = st.columns(2)
                with c_map1:
                    fig_init = plot_tour_plotly(coords, init_tour, title=f"Initial Nearest Neighbor ({init_dist:,} km)")
                    st.plotly_chart(fig_init, use_container_width=True)
                with c_map2:
                    fig_opt = plot_tour_plotly(coords, final_tour, title=f"Final Improved ({final_dist:,} km)")
                    st.plotly_chart(fig_opt, use_container_width=True)

            with tab_view3:
                if "history" in res and len(res["history"]) > 1:
                    conv_chart = plot_convergence_chart(res["history"], init_dist, title=f"Distance Drop Over Steps ({operator_choice})")
                    st.altair_chart(conv_chart, use_container_width=True)
                else:
                    st.info("Tour reached local minimum in a single pass.")

            # Tour sequence inspector
            with st.expander("🔍 Inspect Full Tour Sequence & Coordinates"):
                tour_1based = [c + 1 for c in final_tour]
                st.write(f"**Tour Array (1-indexed City IDs)** ({len(tour_1based)} nodes):")
                st.code(str(tour_1based[:50]) + ("..." if len(tour_1based) > 50 else ""))

# =============================================================================
# TAB 3: METHODOLOGY & MATHEMATICAL DOCUMENTATION
# =============================================================================
with tab_docs:
    st.header("📖 Advanced Operations Research (AOR) - Technical Methodology")
    st.markdown("""
    ### 1. Problem Formulation & Distance Metric
    The Traveling Salesman Problem (TSP) seeks a Hamiltonian cycle of minimum total length visiting each city exactly once and returning to the starting point.

    The coordinates in the 10 benchmark instances follow the **TSPLIB EUC_2D** standard. Given city $i = (x_i, y_i)$ and city $j = (x_j, y_j)$, distance in **km** is calculated via:
    $$\\Delta x = x_i - x_j, \\quad \\Delta y = y_i - y_j$$
    $$d(i, j) = \\text{int}\\left(\\text{round}\\left(\\sqrt{\\Delta x^2 + \\Delta y^2}\\right)\\right)$$

    The total cyclical distance of a tour $\\pi = (c_0, c_1, \\dots, c_{N-1})$ is:
    $$D(\\pi) = \\sum_{k=0}^{N-2} d(c_k, c_{k+1}) + d(c_{N-1}, c_0)$$

    ---

    ### 2. Phase 1: Construction Heuristic (Nearest Neighbor)
    Nearest Neighbor (NN) builds an initial feasible tour greedily:
    1. Begin at starting node $c_0$ (City 1 for the official faculty submission).
    2. At step $k$, select unvisited city $v$ that minimizes $d(c_{k-1}, v)$.
    3. Repeat until all $N$ cities are visited, then connect back to $c_0$.
    *Time Complexity*: $O(N^2)$.

    ---

    ### 3. Phase 2: Neighborhood Search Operators & $\\mathcal{O}(1)$ Incremental Delta Evaluation
    Rather than recalculating the entire tour distance in $O(N)$ after every candidate move, our solver computes the exact change in distance $\\Delta d$ in **$\\mathcal{O}(1)$ time**:

    #### A. 2-Opt Operator
    Reverses the subsegment between positions $i$ and $j$ ($i < j$).
    - Cuts edges: $(c_{i-1}, c_i)$ and $(c_j, c_{j+1 \\pmod N})$.
    - Adds edges: $(c_{i-1}, c_j)$ and $(c_i, c_{j+1 \\pmod N})$.
    $$\\Delta d_{2\\text{-opt}} = d(c_{i-1}, c_j) + d(c_i, c_{j+1}) - [d(c_{i-1}, c_i) + d(c_j, c_{j+1})]$$
    If $\\Delta d < 0$, the move is accepted and the subsegment reversed in-place.

    #### B. Relocate (Insertion) Operator
    Removes city $u = c_i$ from its position and inserts it between $c_{j-1}$ and $c_j$.
    $$\\Delta d_{\\text{rem}} = d(c_{i-1}, c_{i+1}) - [d(c_{i-1}, u) + d(u, c_{i+1})]$$
    $$\\Delta d_{\\text{ins}} = d(c_{j-1}, u) + d(u, c_j) - d(c_{j-1}, c_j)$$
    $$\\Delta d_{\\text{relocate}} = \\Delta d_{\\text{rem}} + \\Delta d_{\\text{ins}}$$

    #### C. Swap (Exchange) Operator
    Exchanges the positions of two distinct cities $u = c_i$ and $v = c_j$.
    Accounts for both non-adjacent and adjacent node edge cases in $O(1)$ time.

    ---

    ### 4. Metaheuristic Frameworks (Escaping Local Minima)
    - **Variable Neighborhood Descent (VND)**:
      Combines multiple neighborhood structures ($N_1 = \\text{Relocate}$, $N_2 = \\text{Swap}$, $N_3 = \\text{2-opt}$).
      Whenever an improvement is found in neighborhood $N_k$, the search loops back to $N_1$. It terminates only when a solution is a local minimum across **all three** neighborhood topologies.
    - **Multi-Start / Random Restarts**:
      Generates multiple diversified starting tours (sampling different starting cities and randomized nearest neighbors) and runs local search to discover distinct local basins of attraction.

    ---

    ### 5. Verified TSPLIB Reference Optimal Values
    | Instance | Dimension ($N$) | Verified Optimal (km) | Reference |
    |---|---|---|---|
    | `a280` | 280 | **2,579** | TSPLIB95 / Concorde Benchmark |
    | `d198` | 198 | **15,780** | TSPLIB95 / Concorde Benchmark |
    | `d493` | 493 | **35,002** | TSPLIB95 / Concorde Benchmark |
    | `fl417` | 417 | **11,861** | TSPLIB95 / Concorde Benchmark |
    | `lin318` | 318 | **42,029** | TSPLIB95 / Concorde Benchmark |
    | `pcb442` | 442 | **50,778** | TSPLIB95 / Concorde Benchmark |
    | `pr152` | 152 | **73,682** | TSPLIB95 / Concorde Benchmark |
    | `pr226` | 226 | **80,369** | TSPLIB95 / Concorde Benchmark |
    | `pr439` | 439 | **107,217** | TSPLIB95 / Concorde Benchmark |
    | `ts225` | 225 | **126,643** | TSPLIB95 / Concorde Benchmark |
    """)
"""
tsp_solver.py - Pure Python & NumPy Optimization Engine for TSP
Custom-coded for Advanced Operations Research (AOR) Challenge.

Features:
- TSPLIB EUC_2D parser and exact integer distance matrix computation.
- Nearest Neighbor starting solution construction (single start & best of all).
- O(1) incremental delta evaluation for local search operators:
    * 2-opt (subsequence reversal)
    * Relocate (single-city insertion)
    * Swap (two-city position exchange)
- Variable Neighborhood Descent (VND) combining Relocate, Swap, and 2-opt.
- Multi-start / Random Restarts for escaping local minima.
- Strict tour validation invariants (length, uniqueness, full coverage, cyclic closure).
- Stopping criteria: natural local minimum, iteration ceiling, stagnation limit, time limit.
- Verified official TSPLIB benchmark reference values.

NO EXTERNAL OPTIMIZATION LIBRARIES (No OR-Tools, No SciPy optimize, etc.).
"""

import math
import time
import random
from typing import List, Tuple, Dict, Optional, Any
import numpy as np


# Verified official TSPLIB optimal values (Reinelt 1991, Concorde TSP Benchmark)
VERIFIED_OPTIMAL_SOLUTIONS: Dict[str, int] = {
    "a280": 2579,
    "d198": 15780,
    "d493": 35002,
    "fl417": 11861,
    "lin318": 42029,
    "pcb442": 50778,
    "pr152": 73682,
    "pr226": 80369,
    "pr439": 107217,
    "ts225": 126643,
}


class TSPLIBParser:
    """Parses standard TSPLIB .tsp files (EUC_2D)."""

    def __init__(self):
        self.name: str = ""
        self.comment: str = ""
        self.dimension: int = 0
        self.edge_weight_type: str = "EUC_2D"
        self.coordinates: List[Tuple[float, float]] = []
        self.node_ids: List[int] = []

    def parse_string(self, content: str) -> "TSPLIBParser":
        lines = content.strip().splitlines()
        in_coord_section = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if in_coord_section:
                if line == "EOF" or line.startswith("EOF"):
                    break
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        self.node_ids.append(node_id)
                        self.coordinates.append((x, y))
                    except ValueError:
                        continue
                if self.dimension > 0 and len(self.coordinates) >= self.dimension:
                    break
            else:
                if ":" in line:
                    key, val = [p.strip() for p in line.split(":", 1)]
                else:
                    parts = line.split(None, 1)
                    key = parts[0]
                    val = parts[1] if len(parts) > 1 else ""

                upper_key = key.upper()
                if upper_key == "NAME":
                    self.name = val
                elif upper_key == "COMMENT":
                    self.comment = val
                elif upper_key == "DIMENSION":
                    self.dimension = int(val)
                elif upper_key == "EDGE_WEIGHT_TYPE":
                    self.edge_weight_type = val.upper()
                elif upper_key == "NODE_COORD_SECTION":
                    in_coord_section = True

        if self.dimension == 0:
            self.dimension = len(self.coordinates)

        return self

    def parse_file(self, filepath: str) -> "TSPLIBParser":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return self.parse_string(f.read())


def compute_distance_matrix(coords: List[Tuple[float, float]]) -> np.ndarray:
    """
    Computes standard TSPLIB EUC_2D integer distance matrix:
    d(i, j) = int(round(sqrt((xi - xj)^2 + (yi - yj)^2)))
    Distance represents kilometers (km) for the challenge.
    """
    arr = np.array(coords, dtype=np.float64)
    # Pairwise differences
    diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
    euclidean_dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # TSPLIB standard rounding to nearest integer
    int_dists = np.rint(euclidean_dists).astype(np.int64)
    return int_dists


def validate_tour(tour: List[int], num_cities: int) -> bool:
    """
    Strict validation of tour invariants:
    1. Length equals num_cities.
    2. No duplicates.
    3. Every city from 0 to num_cities - 1 is present.
    Raises ValueError with detailed error message if any invariant is violated.
    """
    if not isinstance(tour, (list, tuple)):
        raise ValueError(f"Tour must be a list or tuple, got {type(tour)}")

    if len(tour) != num_cities:
        raise ValueError(f"Tour length invariant violated: expected {num_cities}, got {len(tour)}")

    unique_cities = set(tour)
    if len(unique_cities) != num_cities:
        duplicates = len(tour) - len(unique_cities)
        raise ValueError(f"Tour duplicate invariant violated: found {duplicates} duplicate city visits")

    expected_cities = set(range(num_cities))
    missing_cities = expected_cities - unique_cities
    if missing_cities:
        raise ValueError(f"Tour completeness invariant violated: missing cities {list(missing_cities)[:5]}")

    return True


def calculate_tour_distance(tour: List[int], dist_matrix: np.ndarray) -> int:
    """
    Computes total cyclical tour distance in km:
    sum(d(tour[k], tour[k+1])) + d(tour[N-1], tour[0])
    Strictly enforces return-to-start cycle closure.
    """
    n = len(tour)
    if n == 0:
        return 0
    total = 0
    for i in range(n - 1):
        total += int(dist_matrix[tour[i], tour[i + 1]])
    # Return-to-start closure edge
    total += int(dist_matrix[tour[n - 1], tour[0]])
    return total


def nearest_neighbor_tour(
    dist_matrix: np.ndarray,
    start_city: int = 0
) -> Tuple[List[int], int]:
    """
    Phase 1: Construction Heuristic.
    Builds a starting tour using the greedy Nearest Neighbor method.
    start_city: 0-indexed city ID (default 0 = TSPLIB City 1).
    Returns (tour, distance_km).
    """
    num_cities = dist_matrix.shape[0]
    if start_city < 0 or start_city >= num_cities:
        raise ValueError(f"Start city {start_city} out of range [0, {num_cities - 1}]")

    visited = [False] * num_cities
    tour = [start_city]
    visited[start_city] = True

    current = start_city
    for _ in range(num_cities - 1):
        best_next = -1
        best_dist = float("inf")
        # Find closest unvisited neighbor
        for candidate in range(num_cities):
            if not visited[candidate]:
                d = dist_matrix[current, candidate]
                if d < best_dist:
                    best_dist = d
                    best_next = candidate

        visited[best_next] = True
        tour.append(best_next)
        current = best_next

    validate_tour(tour, num_cities)
    total_dist = calculate_tour_distance(tour, dist_matrix)
    return tour, total_dist


def best_nearest_neighbor_tour(dist_matrix: np.ndarray) -> Tuple[List[int], int, int]:
    """
    Evaluates Nearest Neighbor starting from every city (0 to N-1)
    and returns the best starting tour found: (best_tour, best_dist_km, best_start_city).
    """
    num_cities = dist_matrix.shape[0]
    best_dist = float("inf")
    best_tour: List[int] = []
    best_start = 0

    for start in range(num_cities):
        tour, dist = nearest_neighbor_tour(dist_matrix, start_city=start)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour
            best_start = start

    return best_tour, int(best_dist), best_start


# ==============================================================================
# Phase 2: Local Search Improvement Operators with O(1) Incremental Evaluation
# ==============================================================================

def local_search_2opt(
    tour: List[int],
    dist_matrix: np.ndarray,
    max_iterations: int = 500,
    patience: int = 50,
    time_limit_sec: float = 30.0,
    first_improvement: bool = True
) -> Dict[str, Any]:
    """
    2-Opt Neighborhood Search with O(1) Incremental Delta Evaluation.
    Reverses subsegment tour[i:j+1] if delta < 0.
    Broken edges: (tour[i-1], tour[i]) and (tour[j], tour[(j+1)%N])
    Added edges:  (tour[i-1], tour[j]) and (tour[i], tour[(j+1)%N])
    Delta = d(p, v) + d(u, s) - [d(p, u) + d(v, s)]
    """
    n = len(tour)
    validate_tour(tour, n)
    current_tour = list(tour)
    current_dist = calculate_tour_distance(current_tour, dist_matrix)

    history = [current_dist]
    start_time = time.perf_counter()
    moves_accepted = 0
    stagnant_passes = 0
    iteration = 0
    stopped_reason = "local_minimum"

    while iteration < max_iterations:
        iteration += 1
        elapsed = time.perf_counter() - start_time
        if elapsed >= time_limit_sec:
            stopped_reason = "time_limit"
            break

        improved_in_pass = False
        best_delta = 0
        best_move: Optional[Tuple[int, int]] = None

        for i in range(1, n - 1):
            p = current_tour[i - 1]
            u = current_tour[i]
            d_p_u = dist_matrix[p, u]

            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue  # Inverting whole tour does nothing

                s = current_tour[(j + 1) % n]
                v = current_tour[j]

                # O(1) Incremental delta calculation
                delta = (
                    dist_matrix[p, v]
                    + dist_matrix[u, s]
                    - d_p_u
                    - dist_matrix[v, s]
                )

                if delta < 0:
                    if first_improvement:
                        # First improvement: apply immediately
                        current_tour[i:j + 1] = reversed(current_tour[i:j + 1])
                        current_dist += int(delta)
                        moves_accepted += 1
                        improved_in_pass = True
                        history.append(current_dist)
                        break
                    else:
                        # Best improvement: track best
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, j)

            if first_improvement and improved_in_pass:
                break

        if not first_improvement and best_move is not None:
            bi, bj = best_move
            current_tour[bi:bj + 1] = reversed(current_tour[bi:bj + 1])
            current_dist += int(best_delta)
            moves_accepted += 1
            improved_in_pass = True
            history.append(current_dist)

        if improved_in_pass:
            stagnant_passes = 0
        else:
            stagnant_passes += 1
            if stagnant_passes >= 1:  # In local search, 1 full pass with no moves is local minimum
                stopped_reason = "local_minimum"
                break

        if stagnant_passes >= patience:
            stopped_reason = "patience_limit"
            break

    if iteration >= max_iterations and stopped_reason == "local_minimum":
        stopped_reason = "max_iterations"

    # Enforce strict validation invariant on output tour
    validate_tour(current_tour, n)
    actual_dist = calculate_tour_distance(current_tour, dist_matrix)
    runtime = time.perf_counter() - start_time

    return {
        "tour": current_tour,
        "distance": actual_dist,
        "iterations": iteration,
        "moves_accepted": moves_accepted,
        "runtime_sec": runtime,
        "stopped_reason": stopped_reason,
        "history": history,
    }


def local_search_relocate(
    tour: List[int],
    dist_matrix: np.ndarray,
    max_iterations: int = 500,
    time_limit_sec: float = 30.0,
    first_improvement: bool = True
) -> Dict[str, Any]:
    """
    Relocate (Insertion) Neighborhood Search with O(1) Incremental Delta Evaluation.
    Extracts city u = tour[i] and inserts it before city tour[j].
    Delta removal: d(p, s) - [d(p, u) + d(u, s)]
    Delta insertion: d(a, u) + d(u, b) - d(a, b)
    """
    n = len(tour)
    validate_tour(tour, n)
    current_tour = list(tour)
    current_dist = calculate_tour_distance(current_tour, dist_matrix)

    history = [current_dist]
    start_time = time.perf_counter()
    moves_accepted = 0
    iteration = 0
    stopped_reason = "local_minimum"

    while iteration < max_iterations:
        iteration += 1
        if time.perf_counter() - start_time >= time_limit_sec:
            stopped_reason = "time_limit"
            break

        improved_in_pass = False

        for i in range(n):
            u = current_tour[i]
            p = current_tour[(i - 1) % n]
            s = current_tour[(i + 1) % n]

            # Cost change from removing u
            rem_delta = dist_matrix[p, s] - dist_matrix[p, u] - dist_matrix[u, s]

            for j in range(n):
                # Cannot re-insert at adjacent position
                if j == i or j == (i + 1) % n:
                    continue

                a = current_tour[(j - 1) % n]
                b = current_tour[j]

                # Cost change from inserting u between a and b
                ins_delta = dist_matrix[a, u] + dist_matrix[u, b] - dist_matrix[a, b]
                total_delta = rem_delta + ins_delta

                if total_delta < 0:
                    # Apply relocation
                    node = current_tour.pop(i)
                    if j > i:
                        current_tour.insert(j - 1, node)
                    else:
                        current_tour.insert(j, node)

                    current_dist += int(total_delta)
                    moves_accepted += 1
                    improved_in_pass = True
                    history.append(current_dist)
                    break

            if improved_in_pass:
                break

        if not improved_in_pass:
            stopped_reason = "local_minimum"
            break

    validate_tour(current_tour, n)
    actual_dist = calculate_tour_distance(current_tour, dist_matrix)
    runtime = time.perf_counter() - start_time

    return {
        "tour": current_tour,
        "distance": actual_dist,
        "iterations": iteration,
        "moves_accepted": moves_accepted,
        "runtime_sec": runtime,
        "stopped_reason": stopped_reason,
        "history": history,
    }


def local_search_swap(
    tour: List[int],
    dist_matrix: np.ndarray,
    max_iterations: int = 500,
    time_limit_sec: float = 30.0,
    first_improvement: bool = True
) -> Dict[str, Any]:
    """
    Swap (Exchange) Neighborhood Search with O(1) Incremental Delta Evaluation.
    Exchanges tour[i] and tour[j]. Handles adjacent and non-adjacent cases.
    """
    n = len(tour)
    validate_tour(tour, n)
    current_tour = list(tour)
    current_dist = calculate_tour_distance(current_tour, dist_matrix)

    history = [current_dist]
    start_time = time.perf_counter()
    moves_accepted = 0
    iteration = 0
    stopped_reason = "local_minimum"

    while iteration < max_iterations:
        iteration += 1
        if time.perf_counter() - start_time >= time_limit_sec:
            stopped_reason = "time_limit"
            break

        improved_in_pass = False

        for i in range(n - 1):
            u = current_tour[i]
            p_i = current_tour[(i - 1) % n]
            s_i = current_tour[(i + 1) % n]

            for j in range(i + 1, n):
                v = current_tour[j]
                p_j = current_tour[(j - 1) % n]
                s_j = current_tour[(j + 1) % n]

                # Adjacent nodes case
                if j == (i + 1) % n:
                    old_cost = dist_matrix[p_i, u] + dist_matrix[u, v] + dist_matrix[v, s_j]
                    new_cost = dist_matrix[p_i, v] + dist_matrix[v, u] + dist_matrix[u, s_j]
                    delta = new_cost - old_cost
                elif i == 0 and j == n - 1:
                    old_cost = dist_matrix[p_j, v] + dist_matrix[v, u] + dist_matrix[u, s_i]
                    new_cost = dist_matrix[p_j, u] + dist_matrix[u, v] + dist_matrix[v, s_i]
                    delta = new_cost - old_cost
                else:
                    # Non-adjacent nodes
                    old_cost = (
                        dist_matrix[p_i, u] + dist_matrix[u, s_i]
                        + dist_matrix[p_j, v] + dist_matrix[v, s_j]
                    )
                    new_cost = (
                        dist_matrix[p_i, v] + dist_matrix[v, s_i]
                        + dist_matrix[p_j, u] + dist_matrix[u, s_j]
                    )
                    delta = new_cost - old_cost

                if delta < 0:
                    current_tour[i], current_tour[j] = current_tour[j], current_tour[i]
                    current_dist += int(delta)
                    moves_accepted += 1
                    improved_in_pass = True
                    history.append(current_dist)
                    break

            if improved_in_pass:
                break

        if not improved_in_pass:
            stopped_reason = "local_minimum"
            break

    validate_tour(current_tour, n)
    actual_dist = calculate_tour_distance(current_tour, dist_matrix)
    runtime = time.perf_counter() - start_time

    return {
        "tour": current_tour,
        "distance": actual_dist,
        "iterations": iteration,
        "moves_accepted": moves_accepted,
        "runtime_sec": runtime,
        "stopped_reason": stopped_reason,
        "history": history,
    }


def variable_neighborhood_descent(
    initial_tour: List[int],
    dist_matrix: np.ndarray,
    max_passes: int = 100,
    time_limit_sec: float = 30.0
) -> Dict[str, Any]:
    """
    Variable Neighborhood Descent (VND) combining Relocate, Swap, and 2-opt.
    Hierarchy:
      k = 1: Relocate
      k = 2: Swap
      k = 3: 2-opt
    If neighborhood k yields an improvement, reset k to 1.
    If no improvement, increment k.
    Terminates when k > 3 (local minimum across all 3 neighborhoods) or timeout.
    """
    n = len(initial_tour)
    validate_tour(initial_tour, n)
    current_tour = list(initial_tour)
    current_dist = calculate_tour_distance(current_tour, dist_matrix)

    history = [current_dist]
    start_time = time.perf_counter()
    k = 1
    total_moves = 0
    passes = 0
    stopped_reason = "local_minimum"

    while k <= 3 and passes < max_passes:
        passes += 1
        if time.perf_counter() - start_time >= time_limit_sec:
            stopped_reason = "time_limit"
            break

        remaining_time = max(0.5, time_limit_sec - (time.perf_counter() - start_time))

        if k == 1:
            res = local_search_relocate(current_tour, dist_matrix, max_iterations=50, time_limit_sec=remaining_time)
        elif k == 2:
            res = local_search_swap(current_tour, dist_matrix, max_iterations=50, time_limit_sec=remaining_time)
        else:
            res = local_search_2opt(current_tour, dist_matrix, max_iterations=200, time_limit_sec=remaining_time)

        if res["distance"] < current_dist:
            current_tour = res["tour"]
            current_dist = res["distance"]
            total_moves += res["moves_accepted"]
            history.extend(res["history"][1:])
            # Reset to first neighborhood on improvement
            k = 1
        else:
            # Advance to next neighborhood structure
            k += 1

    validate_tour(current_tour, n)
    runtime = time.perf_counter() - start_time

    return {
        "tour": current_tour,
        "distance": current_dist,
        "moves_accepted": total_moves,
        "runtime_sec": runtime,
        "stopped_reason": stopped_reason,
        "history": history,
    }


def multi_start_local_search(
    dist_matrix: np.ndarray,
    num_starts: int = 5,
    method: str = "2opt",  # "2opt" or "vnd"
    time_limit_sec: float = 30.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Random Restarts / Multi-Start Local Search to escape local minima.
    Generates multiple diversified starting tours (randomized nearest neighbor),
    applies local search or VND to each, and retains the globally best validated tour.
    """
    n = dist_matrix.shape[0]
    rng = random.Random(seed)
    start_time = time.perf_counter()

    best_dist = float("inf")
    best_tour: List[int] = []
    best_start_id = -1
    overall_history: List[int] = []

    # First run: canonical Nearest Neighbor from City 1 (index 0)
    canonical_tour, _ = nearest_neighbor_tour(dist_matrix, start_city=0)
    starting_tours = [canonical_tour]

    # Additional starts: Nearest Neighbor from diverse sampled cities
    sampled_starts = rng.sample(range(1, n), min(num_starts - 1, n - 1))
    for s_city in sampled_starts:
        st_tour, _ = nearest_neighbor_tour(dist_matrix, start_city=s_city)
        starting_tours.append(st_tour)

    for idx, start_tour in enumerate(starting_tours):
        elapsed = time.perf_counter() - start_time
        if elapsed >= time_limit_sec:
            break

        rem_time = max(1.0, (time_limit_sec - elapsed) / (len(starting_tours) - idx))

        if method == "vnd":
            res = variable_neighborhood_descent(start_tour, dist_matrix, time_limit_sec=rem_time)
        else:
            res = local_search_2opt(start_tour, dist_matrix, time_limit_sec=rem_time)

        if res["distance"] < best_dist:
            best_dist = res["distance"]
            best_tour = res["tour"]
            best_start_id = idx
            overall_history.append(best_dist)

    validate_tour(best_tour, n)
    runtime = time.perf_counter() - start_time

    return {
        "tour": best_tour,
        "distance": best_dist,
        "best_start_index": best_start_id,
        "starts_evaluated": len(starting_tours),
        "runtime_sec": runtime,
        "history": overall_history,
    }


def solve_faculty_benchmark(
    coords: List[Tuple[float, float]],
    instance_name: str,
    time_limit_sec: float = 15.0
) -> Dict[str, Any]:
    """
    Standard Faculty Benchmark Protocol:
    1. Distance Matrix: standard TSPLIB EUC_2D in km.
    2. Construction: Nearest Neighbor starting at City 1 (index 0).
    3. Improvement: Variable Neighborhood Descent (2-opt + Relocate + Swap).
    4. Strict invariant validation on all tours.
    5. Calculates optimality gap against verified TSPLIB reference.
    """
    dist_matrix = compute_distance_matrix(coords)
    n = len(coords)

    # 1. Starting Solution: Nearest Neighbor (City 1 = index 0)
    init_start_time = time.perf_counter()
    init_tour, init_dist = nearest_neighbor_tour(dist_matrix, start_city=0)
    init_time = time.perf_counter() - init_start_time
    validate_tour(init_tour, n)

    # 2. Local Search Improvement: Fast 2-opt + VND
    search_start_time = time.perf_counter()
    search_res = variable_neighborhood_descent(
        init_tour,
        dist_matrix,
        max_passes=100,
        time_limit_sec=time_limit_sec
    )
    final_tour = search_res["tour"]
    final_dist = search_res["distance"]
    search_time = time.perf_counter() - search_start_time
    validate_tour(final_tour, n)

    # 3. Reference and Optimality Gap
    base_name = instance_name.replace(".tsp", "").lower()
    optimal_ref = VERIFIED_OPTIMAL_SOLUTIONS.get(base_name, None)

    improvement_km = init_dist - final_dist
    improvement_pct = (improvement_km / init_dist * 100.0) if init_dist > 0 else 0.0

    if optimal_ref is not None and optimal_ref > 0:
        gap_pct = ((final_dist - optimal_ref) / optimal_ref) * 100.0
    else:
        gap_pct = None

    return {
        "instance": instance_name,
        "dimension": n,
        "initial_distance_km": init_dist,
        "final_distance_km": final_dist,
        "improvement_km": improvement_km,
        "improvement_pct": round(improvement_pct, 2),
        "optimal_reference_km": optimal_ref,
        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
        "runtime_sec": round(init_time + search_time, 3),
        "validity": "PASSED",
        "initial_tour": init_tour,
        "final_tour": final_tour,
        "convergence_history": search_res["history"],
    }

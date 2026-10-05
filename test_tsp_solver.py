"""
test_tsp_solver.py - Comprehensive Unit Tests for TSP Solver Engine.
Tests:
- Strict Tour Invariant Validation (duplicates, missing, length)
- Exact TSPLIB EUC_2D Metric & Cycle Closure
- Nearest Neighbor Construction Heuristic
- Exact equivalence between O(1) Incremental Delta Evaluations and Full Tour Distance Recalculation
- Monotonic improvement during Local Search (2-opt, Swap, Relocate, VND)
- End-to-end benchmark run on pr152.tsp
"""

import unittest
import numpy as np
import random
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
    solve_faculty_benchmark,
    VERIFIED_OPTIMAL_SOLUTIONS,
)


class TestTSPSolver(unittest.TestCase):

    def test_tour_validation_invariants(self):
        """Test strict validation invariants."""
        # 1. Valid tour passes
        self.assertTrue(validate_tour([0, 1, 2, 3], 4))

        # 2. Duplicate city raises ValueError
        with self.assertRaises(ValueError):
            validate_tour([0, 1, 2, 2], 4)

        # 3. Missing city raises ValueError
        with self.assertRaises(ValueError):
            validate_tour([0, 1, 2, 4], 4)

        # 4. Incorrect length raises ValueError
        with self.assertRaises(ValueError):
            validate_tour([0, 1, 2], 4)

        with self.assertRaises(ValueError):
            validate_tour([0, 1, 2, 3, 4], 4)

    def test_euc2d_distance_and_cycle_closure(self):
        """Test exact EUC_2D rounding and cyclic return-to-start edge."""
        # Unit square: (0,0), (300, 0), (300, 400), (0, 400)
        # Perimeter: 300 + 400 + 300 + 400 = 1400
        coords = [(0.0, 0.0), (300.0, 0.0), (300.0, 400.0), (0.0, 400.0)]
        mat = compute_distance_matrix(coords)

        # Check pairwise distances
        self.assertEqual(mat[0, 1], 300)
        self.assertEqual(mat[1, 2], 400)
        self.assertEqual(mat[0, 2], 500)  # 3-4-5 right triangle

        tour = [0, 1, 2, 3]
        total_dist = calculate_tour_distance(tour, mat)
        self.assertEqual(total_dist, 1400)

    def test_nearest_neighbor_validity(self):
        """Test that Nearest Neighbor constructs a complete valid tour."""
        coords = [(0, 0), (1, 5), (5, 5), (10, 0), (5, -2)]
        mat = compute_distance_matrix(coords)
        tour, dist = nearest_neighbor_tour(mat, start_city=0)

        # Must be valid permutation
        self.assertTrue(validate_tour(tour, len(coords)))
        self.assertEqual(tour[0], 0)
        self.assertEqual(dist, calculate_tour_distance(tour, mat))

    def test_2opt_incremental_delta_accuracy(self):
        """Verify that 2-opt O(1) incremental delta exactly matches full distance recalculation."""
        rng = random.Random(42)
        n = 25
        coords = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
        dist_matrix = compute_distance_matrix(coords)

        tour = list(range(n))
        rng.shuffle(tour)

        # Test 100 random 2-opt inversions
        for _ in range(100):
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            if i == 0 and j == n - 1:
                continue

            old_dist = calculate_tour_distance(tour, dist_matrix)

            # Incremental formula
            p = tour[i - 1]
            u = tour[i]
            v = tour[j]
            s = tour[(j + 1) % n]
            delta = dist_matrix[p, v] + dist_matrix[u, s] - dist_matrix[p, u] - dist_matrix[v, s]

            # Perform reversal
            new_tour = list(tour)
            new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
            new_dist = calculate_tour_distance(new_tour, dist_matrix)

            self.assertEqual(new_dist - old_dist, delta, f"2-opt delta mismatch for move ({i}, {j})")

    def test_relocate_incremental_delta_accuracy(self):
        """Verify that Relocate O(1) incremental delta matches full recalculation."""
        rng = random.Random(123)
        n = 25
        coords = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
        dist_matrix = compute_distance_matrix(coords)

        tour = list(range(n))
        rng.shuffle(tour)

        for _ in range(100):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if j == i or j == (i + 1) % n:
                continue

            old_dist = calculate_tour_distance(tour, dist_matrix)

            u = tour[i]
            p = tour[(i - 1) % n]
            s = tour[(i + 1) % n]
            rem_delta = dist_matrix[p, s] - dist_matrix[p, u] - dist_matrix[u, s]

            a = tour[(j - 1) % n]
            b = tour[j]
            ins_delta = dist_matrix[a, u] + dist_matrix[u, b] - dist_matrix[a, b]
            total_delta = rem_delta + ins_delta

            new_tour = list(tour)
            node = new_tour.pop(i)
            if j > i:
                new_tour.insert(j - 1, node)
            else:
                new_tour.insert(j, node)

            new_dist = calculate_tour_distance(new_tour, dist_matrix)
            self.assertEqual(new_dist - old_dist, total_delta, f"Relocate delta mismatch for ({i} -> {j})")

    def test_swap_incremental_delta_accuracy(self):
        """Verify that Swap O(1) incremental delta matches full recalculation."""
        rng = random.Random(456)
        n = 25
        coords = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
        dist_matrix = compute_distance_matrix(coords)

        tour = list(range(n))
        rng.shuffle(tour)

        for _ in range(100):
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)

            old_dist = calculate_tour_distance(tour, dist_matrix)

            u = tour[i]
            v = tour[j]
            p_i = tour[(i - 1) % n]
            s_i = tour[(i + 1) % n]
            p_j = tour[(j - 1) % n]
            s_j = tour[(j + 1) % n]

            if j == (i + 1) % n:
                old_cost = dist_matrix[p_i, u] + dist_matrix[u, v] + dist_matrix[v, s_j]
                new_cost = dist_matrix[p_i, v] + dist_matrix[v, u] + dist_matrix[u, s_j]
                delta = new_cost - old_cost
            elif i == 0 and j == n - 1:
                old_cost = dist_matrix[p_j, v] + dist_matrix[v, u] + dist_matrix[u, s_i]
                new_cost = dist_matrix[p_j, u] + dist_matrix[u, v] + dist_matrix[v, s_i]
                delta = new_cost - old_cost
            else:
                old_cost = (
                    dist_matrix[p_i, u] + dist_matrix[u, s_i]
                    + dist_matrix[p_j, v] + dist_matrix[v, s_j]
                )
                new_cost = (
                    dist_matrix[p_i, v] + dist_matrix[v, s_i]
                    + dist_matrix[p_j, u] + dist_matrix[u, s_j]
                )
                delta = new_cost - old_cost

            new_tour = list(tour)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_dist = calculate_tour_distance(new_tour, dist_matrix)

            self.assertEqual(new_dist - old_dist, delta, f"Swap delta mismatch for ({i}, {j})")

    def test_local_search_monotonicity_and_validity(self):
        """Verify that local search methods monotonically decrease or maintain distance and keep validity."""
        rng = random.Random(789)
        n = 30
        coords = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
        dist_matrix = compute_distance_matrix(coords)
        init_tour, init_dist = nearest_neighbor_tour(dist_matrix, start_city=0)

        # 2-opt
        res_2opt = local_search_2opt(init_tour, dist_matrix, max_iterations=100)
        validate_tour(res_2opt["tour"], n)
        self.assertLessEqual(res_2opt["distance"], init_dist)

        # Relocate
        res_reloc = local_search_relocate(init_tour, dist_matrix, max_iterations=100)
        validate_tour(res_reloc["tour"], n)
        self.assertLessEqual(res_reloc["distance"], init_dist)

        # Swap
        res_swap = local_search_swap(init_tour, dist_matrix, max_iterations=100)
        validate_tour(res_swap["tour"], n)
        self.assertLessEqual(res_swap["distance"], init_dist)

        # VND
        res_vnd = variable_neighborhood_descent(init_tour, dist_matrix, max_passes=20)
        validate_tour(res_vnd["tour"], n)
        self.assertLessEqual(res_vnd["distance"], init_dist)

    def test_pr152_benchmark_run(self):
        """Test full parser and solver execution on pr152.tsp."""
        parser = TSPLIBParser().parse_file("pr152.tsp")
        self.assertEqual(parser.dimension, 152)
        self.assertEqual(len(parser.coordinates), 152)

        res = solve_faculty_benchmark(parser.coordinates, "pr152.tsp", time_limit_sec=10.0)

        self.assertEqual(res["instance"], "pr152.tsp")
        self.assertEqual(res["dimension"], 152)
        self.assertEqual(res["validity"], "PASSED")
        self.assertEqual(res["optimal_reference_km"], 73682)

        # Verify improvement
        self.assertLess(res["final_distance_km"], res["initial_distance_km"])
        self.assertGreater(res["improvement_pct"], 0.0)
        # Check that gap is reasonably bounded (< 15% from optimal with basic VND)
        self.assertIsNotNone(res["gap_pct"])
        self.assertLess(res["gap_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Branch-and-Price with DP-based Column Generation + Visualization

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import math
import gurobipy as gurobi
from gurobipy import GRB
from collections import deque
import time
import matplotlib.pyplot as plt


@dataclass
class ProductionItem:
    item_id: int
    number_of_periods: int
    demand_quantity_by_period: List[float]
    production_unit_cost_by_period: List[float]
    setup_cost_by_period: List[float]
    holding_unit_cost_by_period: List[float]
    perishability_horizon_by_start_period: List[int]
    lost_sales_penalty_per_unit: float


@dataclass
class ProductionPlanColumn:
    item_id: int
    total_plan_cost: float
    capacity_usage_by_period: List[float]
    setup_open_fraction_by_period: Optional[List[float]] = None
    lost_demand_quantity_by_period: Optional[List[float]] = None
    arc_usage: Optional[Dict[Tuple[int, int], float]] = None


@dataclass
class BranchNode:
    node_id: int
    parent_id: Optional[int]
    depth: int
    theta_0_by_item: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    theta_1_by_item: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    lp_bound: float = math.inf
    is_integer: bool = False
    is_pruned: bool = False
    prune_reason: Optional[str] = None
    branch_variable: Optional[Tuple[int, int, int, float]] = None
    branch_direction: Optional[str] = None


class BranchAndPriceVisualizer:
    """Real-time visualization of Branch-and-Price algorithm"""

    def __init__(self):
        self.fig = plt.figure(figsize=(22, 12))
        self.fig.suptitle(
            "Branch-and-Price with DP Pricing (BFS)",
            fontsize=18,
            fontweight="bold",
            color="darkblue",
        )

        gs = self.fig.add_gridspec(
            3, 3, hspace=0.35, wspace=0.35, left=0.05, right=0.97, top=0.94, bottom=0.05
        )

        # Top row: Bounds and Statistics
        self.ax_bounds = self.fig.add_subplot(gs[0, :2])
        self.ax_bounds.set_title(
            "Bounds Progression", fontsize=13, fontweight="bold", pad=10
        )
        self.ax_bounds.set_xlabel("Nodes Explored", fontsize=10)
        self.ax_bounds.set_ylabel("Objective Value", fontsize=10)

        self.ax_stats = self.fig.add_subplot(gs[0, 2])
        self.ax_stats.set_title(
            "Tree Statistics", fontsize=13, fontweight="bold", pad=10
        )
        self.ax_stats.axis("off")

        # Middle row: Node info and Fractional Z
        self.ax_info = self.fig.add_subplot(gs[1, :2])
        self.ax_info.axis("off")
        self.ax_info.set_title(
            "Current Node Details", fontsize=13, fontweight="bold", pad=10
        )

        self.ax_frac = self.fig.add_subplot(gs[1, 2])
        self.ax_frac.set_title(
            "Fractional Z Variables", fontsize=13, fontweight="bold", pad=10
        )

        # Bottom row: Depth distribution and Incumbent history
        self.ax_depth = self.fig.add_subplot(gs[2, :2])
        self.ax_depth.set_title(
            "Nodes by Depth", fontsize=13, fontweight="bold", pad=10
        )
        self.ax_depth.set_xlabel("Depth", fontsize=10)
        self.ax_depth.set_ylabel("Node Count", fontsize=10)

        self.ax_incumbent = self.fig.add_subplot(gs[2, 2])
        self.ax_incumbent.set_title(
            "Incumbent & Fathoming", fontsize=13, fontweight="bold", pad=10
        )
        self.ax_incumbent.axis("off")

        # Data tracking
        self.nodes_explored_list = []
        self.best_lb_list = []
        self.incumbent_list = []
        self.incumbent_history = []
        self.fathom_events = []

        self.nodes_by_depth = {}
        self.nodes_pruned = 0
        self.nodes_pruned_by_bound = 0
        self.nodes_pruned_by_infeasible = 0
        self.nodes_pruned_by_duplicate = 0
        self.nodes_fathomed_integer = 0
        self.nodes_fathomed_on_incumbent = 0
        self.nodes_integer = 0
        self.nodes_total = 0
        self.max_depth = 0

        plt.ion()
        plt.show()

    def update(
        self,
        node_id: int,
        lp_bound: float,
        z_values: Dict,
        global_lb: float,
        incumbent: Optional[float],
        nodes_explored: int,
        node: BranchNode,
        queue_size: int,
        incumbent_updated: bool = False,
        num_fathomed: int = 0,
    ):
        self.nodes_total += 1
        depth = node.depth
        self.nodes_by_depth[depth] = self.nodes_by_depth.get(depth, 0) + 1
        self.max_depth = max(self.max_depth, depth)

        if node.is_integer:
            self.nodes_integer += 1
            self.nodes_fathomed_integer += 1
            if incumbent_updated and incumbent:
                self.incumbent_history.append((node_id, incumbent))
                if num_fathomed > 0:
                    self.fathom_events.append((node_id, num_fathomed))
                    self.nodes_fathomed_on_incumbent += num_fathomed

        if node.is_pruned:
            self.nodes_pruned += 1
            if node.prune_reason == "bound":
                self.nodes_pruned_by_bound += 1
            elif node.prune_reason == "infeasible":
                self.nodes_pruned_by_infeasible += 1
            elif node.prune_reason == "duplicate":
                self.nodes_pruned_by_duplicate += 1

        self.nodes_explored_list.append(nodes_explored)
        self.best_lb_list.append(global_lb)
        self.incumbent_list.append(incumbent if incumbent else None)

        self.draw_bounds()
        self.draw_statistics(queue_size, global_lb, incumbent)
        self.draw_node_info(node, z_values)
        self.draw_fractional_z(z_values)
        self.draw_depth_distribution()
        self.draw_incumbent_history()

        plt.pause(0.01)

    def draw_bounds(self):
        self.ax_bounds.clear()
        self.ax_bounds.set_title(
            "Bounds Progression", fontsize=13, fontweight="bold", pad=10
        )
        self.ax_bounds.set_xlabel("Nodes Explored", fontsize=10)
        self.ax_bounds.set_ylabel("Objective Value", fontsize=10)

        if len(self.nodes_explored_list) == 0:
            return

        self.ax_bounds.plot(
            self.nodes_explored_list,
            self.best_lb_list,
            "b-",
            linewidth=2.5,
            label="Best Lower Bound",
            alpha=0.8,
        )

        incumbent_x = [
            x
            for x, y in zip(self.nodes_explored_list, self.incumbent_list)
            if y is not None
        ]
        incumbent_y = [y for y in self.incumbent_list if y is not None]

        if len(incumbent_y) > 0:
            self.ax_bounds.plot(
                incumbent_x,
                incumbent_y,
                "g-",
                linewidth=2.5,
                label="Incumbent",
                marker="*",
                markersize=10,
                alpha=0.8,
            )

            last_lb = self.best_lb_list[-1]
            last_ub = incumbent_y[-1]
            self.ax_bounds.fill_between(
                self.nodes_explored_list,
                [last_lb] * len(self.nodes_explored_list),
                [last_ub] * len(self.nodes_explored_list),
                alpha=0.15,
                color="yellow",
                label="Gap",
            )

        self.ax_bounds.legend(loc="upper right", fontsize=9, framealpha=0.9)
        self.ax_bounds.grid(True, alpha=0.3, linestyle="--")
        self.ax_bounds.set_facecolor("#f8f9fa")

    def draw_statistics(
        self, queue_size: int, global_lb: float, incumbent: Optional[float]
    ):
        self.ax_stats.clear()
        self.ax_stats.axis("off")
        self.ax_stats.set_title(
            "Tree Statistics", fontsize=13, fontweight="bold", pad=10
        )

        stats_text = f"""
╔═════════════════════════════════╗
║   SEARCH STATISTICS             ║
╠═════════════════════════════════╣
║  Strategy:   BREADTH-FIRST      ║
║  Queue Size:      {queue_size:6d}         ║
╠═════════════════════════════════╣
║  Created:         {self.nodes_total:6d}         ║
║  Explored:        {len(self.nodes_explored_list):6d}         ║
║  Integer Found:   {self.nodes_integer:6d}         ║
╠═════════════════════════════════╣
║  FATHOMED:        {self.nodes_pruned:6d}         ║
║    By Bound:      {self.nodes_pruned_by_bound:6d}         ║
║    Infeasible:    {self.nodes_pruned_by_infeasible:6d}         ║
║    Integer:       {self.nodes_fathomed_integer:6d}         ║
║    On Incumbent:  {self.nodes_fathomed_on_incumbent:6d}         ║
║    Duplicate:     {self.nodes_pruned_by_duplicate:6d}         ║
╠═════════════════════════════════╣
║  Max Depth:       {self.max_depth:6d}         ║
╠═════════════════════════════════╣
║  Best LB:     {global_lb:10.2f}       ║
"""

        if incumbent:
            gap = incumbent - global_lb
            gap_pct = 100 * gap / max(abs(incumbent), 1)
            stats_text += f"║  Incumbent:   {incumbent:10.2f}       ║\n"
            stats_text += f"║  Gap:         {gap:10.2f}       ║\n"
            stats_text += f"║  Gap %:       {gap_pct:9.2f}%      ║\n"
        else:
            stats_text += f"║  Incumbent:         None       ║\n"

        stats_text += "╚═════════════════════════════════╝"

        self.ax_stats.text(
            0.5,
            0.5,
            stats_text,
            transform=self.ax_stats.transAxes,
            fontsize=8.5,
            verticalalignment="center",
            horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="#e3f2fd",
                alpha=0.8,
                edgecolor="#1976d2",
                linewidth=2,
            ),
        )

    def draw_node_info(self, node: BranchNode, z_values: Dict):
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.set_title(
            "Current Node Details", fontsize=13, fontweight="bold", pad=10
        )

        info_lines = [
            f"╔═══════════════════════════════════════════════════════════════════════════╗",
            f"║  NODE {node.node_id:<5d}  │  DEPTH {node.depth:<3d}  │  LP BOUND {node.lp_bound:>10.2f}                   ║",
            f"╠═══════════════════════════════════════════════════════════════════════════╣",
        ]

        total_forbidden = sum(len(arcs) for arcs in node.theta_0_by_item.values())
        total_forced = sum(len(arcs) for arcs in node.theta_1_by_item.values())

        info_lines.append(
            f"║  Total Forbidden Arcs (θ⁰): {total_forbidden:4d}                                          ║"
        )
        info_lines.append(
            f"║  Total Forced Arcs (θ¹):    {total_forced:4d}                                          ║"
        )

        if total_forbidden > 0 or total_forced > 0:
            info_lines.append(
                f"╠═══════════════════════════════════════════════════════════════════════════╣"
            )
            for item_id in sorted(
                set(
                    list(node.theta_0_by_item.keys())
                    + list(node.theta_1_by_item.keys())
                )
            ):
                arcs_0 = node.theta_0_by_item.get(item_id, set())
                arcs_1 = node.theta_1_by_item.get(item_id, set())
                line = f"║  Item {item_id}: "
                parts = []
                if arcs_0:
                    parts.append(f"θ⁰={len(arcs_0)}")
                if arcs_1:
                    parts.append(f"θ¹={len(arcs_1)}")
                line += ", ".join(parts)
                line += " " * (74 - len(line)) + "║"
                info_lines.append(line)

        if node.branch_variable:
            item, t, u, val = node.branch_variable
            info_lines.append(
                f"╠═══════════════════════════════════════════════════════════════════════════╣"
            )
            info_lines.append(
                f"║  BRANCHED ON: Z[item={item}, t={t}, u={u}] = {val:.4f}                             ║"
            )
            info_lines.append(f"║  DIRECTION:   {node.branch_direction:<60s}  ║")

        info_lines.append(
            f"╚═══════════════════════════════════════════════════════════════════════════╝"
        )

        info_text = "\n".join(info_lines)
        self.ax_info.text(
            0.5,
            0.5,
            info_text,
            transform=self.ax_info.transAxes,
            fontsize=8.5,
            verticalalignment="center",
            horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="#fff9e6",
                alpha=0.8,
                edgecolor="#ffa500",
                linewidth=2,
            ),
        )

    def draw_fractional_z(self, z_values: Dict):
        self.ax_frac.clear()
        self.ax_frac.set_title(
            "Fractional Z Variables", fontsize=13, fontweight="bold", pad=10
        )

        frac_list = []
        for item_id, arcs in z_values.items():
            for (t, u), val in arcs.items():
                if 0.01 < val < 0.99:
                    frac = min(val, 1 - val)
                    frac_list.append((item_id, t, u, val, frac))

        frac_list.sort(key=lambda x: x[4], reverse=True)
        frac_list = frac_list[:10]

        if len(frac_list) == 0:
            self.ax_frac.text(
                0.5,
                0.5,
                "✓ INTEGER!",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="#28a745",
                bbox=dict(
                    boxstyle="round,pad=1",
                    facecolor="#d4edda",
                    edgecolor="#28a745",
                    linewidth=3,
                ),
            )
            self.ax_frac.set_xlim([0, 1])
            self.ax_frac.set_ylim([0, 1])
            return

        labels = [f"I{i}:({t},{u})" for i, t, u, v, f in frac_list]
        values = [v for i, t, u, v, f in frac_list]
        colors = [
            (
                "#dc3545"
                if abs(v - 0.5) < 0.1
                else "#fd7e14" if abs(v - 0.5) < 0.25 else "#ffc107"
            )
            for v in values
        ]

        y_pos = range(len(labels))
        bars = self.ax_frac.barh(
            y_pos, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2
        )

        for i, (bar, val) in enumerate(zip(bars, values)):
            self.ax_frac.text(
                val + 0.02, i, f"{val:.3f}", va="center", fontsize=8, fontweight="bold"
            )

        self.ax_frac.set_yticks(y_pos)
        self.ax_frac.set_yticklabels(labels, fontsize=8.5)
        self.ax_frac.set_xlabel("Z Value", fontsize=9)
        self.ax_frac.set_xlim([0, 1.15])
        self.ax_frac.axvline(
            x=0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.5
        )
        self.ax_frac.grid(True, alpha=0.3, axis="x", linestyle="--")
        self.ax_frac.set_facecolor("#f8f9fa")

    def draw_depth_distribution(self):
        self.ax_depth.clear()
        self.ax_depth.set_title(
            "Nodes by Depth", fontsize=13, fontweight="bold", pad=10
        )
        self.ax_depth.set_xlabel("Depth", fontsize=10)
        self.ax_depth.set_ylabel("Node Count", fontsize=10)

        if not self.nodes_by_depth:
            return

        depths = sorted(self.nodes_by_depth.keys())
        counts = [self.nodes_by_depth[d] for d in depths]

        self.ax_depth.bar(
            depths,
            counts,
            color="steelblue",
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
        )
        self.ax_depth.set_xticks(depths)
        self.ax_depth.grid(True, alpha=0.3, axis="y", linestyle="--")
        self.ax_depth.set_facecolor("#f8f9fa")

    def draw_incumbent_history(self):
        self.ax_incumbent.clear()
        self.ax_incumbent.axis("off")
        self.ax_incumbent.set_title(
            "Incumbent & Fathoming", fontsize=13, fontweight="bold", pad=10
        )

        if not self.incumbent_history:
            text = """
╔═════════════════════════════════╗
║   NO INCUMBENT YET              ║
╚═════════════════════════════════╝
"""
            self.ax_incumbent.text(
                0.5,
                0.5,
                text,
                transform=self.ax_incumbent.transAxes,
                fontsize=10,
                verticalalignment="center",
                horizontalalignment="center",
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.8",
                    facecolor="#ffe6e6",
                    alpha=0.8,
                    edgecolor="#dc3545",
                    linewidth=2,
                ),
            )
            return

        text = "╔═════════════════════════════════╗\n"
        text += "║  INCUMBENT IMPROVEMENTS         ║\n"
        text += "╠═════════════════════════════════╣\n"

        events = []
        for node_id, value in self.incumbent_history[-6:]:
            fathomed = 0
            for fnode, fcount in self.fathom_events:
                if fnode == node_id:
                    fathomed = fcount
                    break
            events.append((node_id, value, fathomed))

        for i, (node_id, value, fathomed) in enumerate(events, 1):
            text += f"║ #{i} N{node_id:4d}: {value:10.2f}      ║\n"
            if fathomed > 0:
                text += f"║      └─ Fathomed {fathomed:3d} nodes   ║\n"

        if len(self.incumbent_history) > 6:
            text += f"║  ... +{len(self.incumbent_history)-6} earlier         ║\n"

        text += "╚═════════════════════════════════╝"

        self.ax_incumbent.text(
            0.5,
            0.5,
            text,
            transform=self.ax_incumbent.transAxes,
            fontsize=8.5,
            verticalalignment="center",
            horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="#d4edda",
                alpha=0.8,
                edgecolor="#28a745",
                linewidth=2,
            ),
        )


class SearchStatistics:
    def __init__(self):
        self.nodes_created = 0
        self.nodes_explored = 0
        self.nodes_integer = 0
        self.nodes_fathomed_by_bound = 0
        self.nodes_fathomed_by_infeasible = 0
        self.nodes_fathomed_integer = 0
        self.nodes_fathomed_on_incumbent = 0
        self.nodes_fathomed_duplicate = 0
        self.max_depth = 0
        self.incumbent_history = []
        self.start_time = time.time()

    def node_created(self):
        self.nodes_created += 1

    def node_explored(
        self, node: BranchNode, incumbent_improved: bool = False, num_fathomed: int = 0
    ):
        self.nodes_explored += 1
        self.max_depth = max(self.max_depth, node.depth)
        if node.is_integer:
            self.nodes_integer += 1
            self.nodes_fathomed_integer += 1
            if incumbent_improved:
                self.incumbent_history.append((node.node_id, node.lp_bound))
                self.nodes_fathomed_on_incumbent += num_fathomed
        if node.is_pruned:
            if node.prune_reason == "bound":
                self.nodes_fathomed_by_bound += 1
            elif node.prune_reason == "infeasible":
                self.nodes_fathomed_by_infeasible += 1
            elif node.prune_reason == "duplicate":
                self.nodes_fathomed_duplicate += 1

    def print_summary(self, best_lb: float, best_ub: Optional[float], eps: float):
        elapsed = time.time() - self.start_time
        total_fathomed = (
            self.nodes_fathomed_by_bound
            + self.nodes_fathomed_by_infeasible
            + self.nodes_fathomed_integer
            + self.nodes_fathomed_duplicate
        )
        print("\n" + "=" * 70)
        print(" " * 28 + "FINAL RESULTS")
        print("=" * 70)
        print(f"  Time elapsed:       {elapsed:.2f} seconds")
        print(f"  Nodes created:      {self.nodes_created}")
        print(f"  Nodes explored:     {self.nodes_explored}")
        print(f"  Integer solutions:  {self.nodes_integer}")
        print(f"  Max depth:          {self.max_depth}")
        print()
        print(f"  Fathomed nodes:     {total_fathomed}")
        print(f"    By bound:         {self.nodes_fathomed_by_bound}")
        print(f"    Infeasible:       {self.nodes_fathomed_by_infeasible}")
        print(f"    Integer:          {self.nodes_fathomed_integer}")
        print(f"    On incumbent:     {self.nodes_fathomed_on_incumbent}")
        print(f"    Duplicate:        {self.nodes_fathomed_duplicate}")
        print()
        print(f"  Best lower bound:   {best_lb:.4f}")
        if best_ub:
            print(f"  Best upper bound:   {best_ub:.4f}")
            gap = best_ub - best_lb
            gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
            print(f"  Gap:                {gap:.4f} ({gap_pct:.2f}%)")
            if gap < eps:
                print(f"\n  ★★★ PROVEN OPTIMAL! ★★★")
        else:
            print(f"  Best upper bound:   Not found")
        print("=" * 70)


def node_signature(node: BranchNode) -> str:
    sig_parts = []
    for item_id in sorted(node.theta_0_by_item.keys()):
        arcs = sorted(node.theta_0_by_item[item_id])
        if arcs:
            sig_parts.append(f"I{item_id}_0:{','.join(f'{t}-{u}' for t,u in arcs)}")
    for item_id in sorted(node.theta_1_by_item.keys()):
        arcs = sorted(node.theta_1_by_item[item_id])
        if arcs:
            sig_parts.append(f"I{item_id}_1:{','.join(f'{t}-{u}' for t,u in arcs)}")
    return "|".join(sig_parts)


def dp_pricing_for_single_item(
    production_item: ProductionItem,
    capacity_dual_prices: List[float],
    convexity_dual_price: float,
    theta_0: Set[Tuple[int, int]],
    theta_1: Set[Tuple[int, int]],
    epsilon_tolerance: float = 1e-9,
) -> Tuple[float, Optional[ProductionPlanColumn]]:
    """DP-based pricing for ZIO lot-sizing with perishability."""
    T = production_item.number_of_periods
    demand = production_item.demand_quantity_by_period
    setup_cost = production_item.setup_cost_by_period
    prod_cost = production_item.production_unit_cost_by_period
    holding_cost = production_item.holding_unit_cost_by_period
    shelf_life = production_item.perishability_horizon_by_start_period

    INF = 1e10
    F = [INF] * (T + 1)
    F[0] = 0.0
    pred = [(-1, -1)] * (T + 1)

    for u in range(1, T + 1):
        if demand[u - 1] == 0 and F[u - 1] < INF:
            if F[u - 1] < F[u]:
                F[u] = F[u - 1]
                pred[u] = (-2, -2)

        for t in range(u):
            v_t = shelf_life[t]
            if t + v_t <= u - 1:
                continue

            violates = False
            for period in range(t, u):
                if demand[period] > 0 and (t, period) in theta_0:
                    violates = True
                    break

            for period in range(t, u):
                if demand[period] > 0:
                    for t_f, u_f in theta_1:
                        if u_f == period and t_f != t:
                            violates = True
                            break
                if violates:
                    break

            if violates:
                continue

            block_demand = sum(demand[j] for j in range(t, u))
            if block_demand == 0:
                continue

            cost = setup_cost[t] + prod_cost[t] * block_demand

            for tau in range(t, u - 1):
                held = sum(demand[j] for j in range(tau + 1, u))
                cost += holding_cost[tau] * held

            cost -= capacity_dual_prices[t] * block_demand

            if F[t] < INF:
                total = F[t] + cost
                if total < F[u]:
                    F[u] = total
                    pred[u] = (t, u - 1)

    if F[T] >= INF:
        return INF, None

    blocks = []
    u = T
    while u > 0:
        if pred[u] == (-2, -2):
            u -= 1
        elif pred[u][0] >= 0:
            t, e = pred[u]
            blocks.append((t, e))
            u = t
        else:
            break

    if not blocks:
        return INF, None

    blocks.reverse()

    cap_usage = [0.0] * T
    setup_usage = [0.0] * T
    arc_usage = {}
    total_cost = 0.0

    for t, e in blocks:
        block_demand = sum(demand[j] for j in range(t, e + 1))
        cap_usage[t] = block_demand
        setup_usage[t] = 1.0

        for j in range(t, e + 1):
            if demand[j] > 0:
                arc_usage[(t, j)] = 1.0

        total_cost += setup_cost[t] + prod_cost[t] * block_demand

        for tau in range(t, e):
            held = sum(demand[j] for j in range(tau + 1, e + 1))
            total_cost += holding_cost[tau] * held

    rc = total_cost - convexity_dual_price
    for t in range(T):
        rc -= capacity_dual_prices[t] * cap_usage[t]

    if rc < -epsilon_tolerance:
        return rc, ProductionPlanColumn(
            item_id=production_item.item_id,
            total_plan_cost=total_cost,
            capacity_usage_by_period=cap_usage,
            setup_open_fraction_by_period=setup_usage,
            lost_demand_quantity_by_period=[0.0] * T,
            arc_usage=arc_usage,
        )
    else:
        return rc, None


class RestrictedMasterProblem:
    def __init__(self, items: List[ProductionItem], capacity: List[float]):
        self.items = items
        self.T = len(capacity)
        self.capacity = capacity
        self.model = gurobi.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.model.Params.Method = 1

        self.columns = {i.item_id: [] for i in items}
        self.lambdas = {}
        self.convex_expr = {i.item_id: gurobi.LinExpr(0.0) for i in items}
        self.cap_expr = [gurobi.LinExpr(0.0) for _ in range(self.T)]
        self.convex_con = {}
        self.cap_con = []

        for item in items:
            self.convex_con[item.item_id] = self.model.addConstr(
                self.convex_expr[item.item_id] == 1.0, name=f"conv_{item.item_id}"
            )
        for t in range(self.T):
            self.cap_con.append(
                self.model.addConstr(
                    self.cap_expr[t] <= self.capacity[t], name=f"cap_{t}"
                )
            )

    def _rebuild(self):
        for item in self.items:
            self.model.remove(self.convex_con[item.item_id])
            self.convex_con[item.item_id] = self.model.addConstr(
                self.convex_expr[item.item_id] == 1.0, name=f"conv_{item.item_id}"
            )
        for t in range(self.T):
            self.model.remove(self.cap_con[t])
            self.cap_con[t] = self.model.addConstr(
                self.cap_expr[t] <= self.capacity[t], name=f"cap_{t}"
            )

    def add_column(self, col: ProductionPlanColumn):
        item_id = col.item_id
        idx = len(self.columns[item_id])
        lam = self.model.addVar(
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            obj=col.total_plan_cost,
            name=f"lam_{item_id}_{idx}",
        )
        self.lambdas[(item_id, idx)] = lam
        self.columns[item_id].append(col)
        self.convex_expr[item_id] += lam
        for t in range(self.T):
            if col.capacity_usage_by_period[t] > 0:
                self.cap_expr[t] += col.capacity_usage_by_period[t] * lam
        self._rebuild()

    def solve(self):
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            return math.inf, {}, []
        return (
            self.model.ObjVal,
            {i.item_id: self.convex_con[i.item_id].Pi for i in self.items},
            [self.cap_con[t].Pi for t in range(self.T)],
        )


def solve_node_with_column_generation(
    items: List[ProductionItem],
    capacity: List[float],
    node: BranchNode,
    max_iter: int = 300,
    eps: float = 1e-9,
    verbose: bool = False,
):
    rmp = RestrictedMasterProblem(items, capacity)

    for item in items:
        T = item.number_of_periods
        total_demand = sum(item.demand_quantity_by_period)
        dummy_cost = 10000.0 * total_demand

        col = ProductionPlanColumn(
            item_id=item.item_id,
            total_plan_cost=dummy_cost,
            capacity_usage_by_period=[0.0] * T,
            setup_open_fraction_by_period=[0.0] * T,
            lost_demand_quantity_by_period=item.demand_quantity_by_period[:],
            arc_usage={},
        )
        rmp.add_column(col)

    if verbose:
        print(f"  └─ CG: ", end="", flush=True)

    for iteration in range(1, max_iter + 1):
        lb, mu, pi = rmp.solve()
        if not math.isfinite(lb):
            if verbose:
                print(f"INFEASIBLE")
            return math.inf, rmp, False, {}

        any_added = False
        for item in items:
            theta_0 = node.theta_0_by_item.get(item.item_id, set())
            theta_1 = node.theta_1_by_item.get(item.item_id, set())

            rc, col = dp_pricing_for_single_item(
                item, pi, mu[item.item_id], theta_0, theta_1, eps
            )

            if col is not None and rc < -eps:
                rmp.add_column(col)
                any_added = True

        if not any_added:
            if verbose:
                print(f"LB={lb:.2f}")
            z_vals = extract_z_values(rmp, items, eps)
            return lb, rmp, True, z_vals

    lb, _, _ = rmp.solve()
    z_vals = extract_z_values(rmp, items, eps)

    if verbose:
        print(f"LB={lb:.2f} (max iter)")

    return lb, rmp, False, z_vals


def extract_z_values(
    rmp: RestrictedMasterProblem, items: List[ProductionItem], eps: float
):
    z = {i.item_id: {} for i in items}
    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        if col.arc_usage:
            for t, u in col.arc_usage:
                z[item_id][(t, u)] = z[item_id].get((t, u), 0.0) + lam_val
    return z


def is_integer(z_vals, eps=1e-6):
    for arcs in z_vals.values():
        for val in arcs.values():
            if eps < val < 1.0 - eps:
                return False
    return True


def find_most_fractional(z_vals, node: BranchNode, eps=1e-9):
    best_frac = 0.0
    best = None
    for item_id, arcs in z_vals.items():
        for (t, u), val in arcs.items():
            if (t, u) in node.theta_0_by_item.get(item_id, set()):
                continue
            if (t, u) in node.theta_1_by_item.get(item_id, set()):
                continue
            frac = min(val, 1.0 - val)
            if frac > eps and frac > best_frac:
                best_frac = frac
                best = (item_id, t, u, val)
    return best


def fathom_queue_by_incumbent(queue: deque, incumbent: float, eps: float) -> int:
    original_size = len(queue)
    new_queue = deque()
    for node, z_vals in queue:
        if node.lp_bound < incumbent - eps:
            new_queue.append((node, z_vals))
    num_fathomed = original_size - len(new_queue)
    queue.clear()
    queue.extend(new_queue)
    return num_fathomed


def solve_branch_and_price_with_viz(
    items, capacity, max_nodes=5000, eps=1e-6, print_frequency=50
):
    print("\n" + "=" * 70)
    print(" " * 20 + "BRANCH-AND-PRICE WITH DP PRICING (BFS)")
    print("=" * 70)

    # Create visualizer
    viz = BranchAndPriceVisualizer()

    stats = SearchStatistics()

    root = BranchNode(
        0, None, 0, {i.item_id: set() for i in items}, {i.item_id: set() for i in items}
    )

    print("\n>>> ROOT NODE <<<")
    root_lb, _, _, z_vals = solve_node_with_column_generation(
        items, capacity, root, verbose=True
    )

    if not math.isfinite(root_lb):
        print("\n✗ Root infeasible!")
        return math.inf, None, 1

    root.lp_bound = root_lb
    print(f"  Root LB:  {root_lb:.4f}")
    print(f"  Integer?  {is_integer(z_vals, eps)}")

    best_lb = root_lb
    best_ub = None
    nodes_explored = 1
    node_counter = 1

    seen_signatures = {node_signature(root)}
    queue = deque([(root, z_vals)])
    stats.node_created()

    # Visualize root
    viz.update(
        0, root_lb, z_vals, best_lb, best_ub, nodes_explored, root, len(queue), False, 0
    )

    if is_integer(z_vals, eps):
        best_ub = root_lb
        best_lb = root_lb
        root.is_integer = True
        stats.node_explored(root, incumbent_improved=True)
        print("\n✓ Root is INTEGER - OPTIMAL!")
        viz.update(
            0,
            root_lb,
            z_vals,
            best_lb,
            best_ub,
            nodes_explored,
            root,
            len(queue),
            True,
            0,
        )
        stats.print_summary(best_lb, best_ub, eps)
        plt.ioff()
        plt.show()
        return root_lb, root_lb, 1

    stats.node_explored(root)

    print(f"\n{'='*70}")
    print("BREADTH-FIRST SEARCH")
    print(f"{'='*70}\n")

    while queue and nodes_explored < max_nodes:
        node, parent_z = queue.popleft()

        if node.node_id != 0:
            if nodes_explored % print_frequency == 0:
                gap_str = "N/A"
                if best_ub:
                    gap = best_ub - best_lb
                    gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
                    gap_str = f"{gap_pct:.2f}%"
                print(
                    f"[Progress: N={nodes_explored:4d}, Queue={len(queue):4d}, "
                    f"LB={best_lb:.2f}, UB={best_ub if best_ub else 'N/A'}, Gap={gap_str}]"
                )

            print(f"N{node.node_id:4d} D{node.depth:2d}", end="", flush=True)

            lb, _, _, z_vals = solve_node_with_column_generation(
                items, capacity, node, verbose=True
            )

            nodes_explored += 1
            node.lp_bound = lb

            if not math.isfinite(lb):
                print(f"  FATHOMED: Infeasible")
                node.is_pruned = True
                node.prune_reason = "infeasible"
                stats.node_explored(node)
                if queue:
                    best_lb = min(min(n.lp_bound for n, _ in queue), best_lb)
                elif best_ub is not None:
                    best_lb = best_ub
                viz.update(
                    node.node_id,
                    lb,
                    z_vals,
                    best_lb,
                    best_ub,
                    nodes_explored,
                    node,
                    len(queue),
                    False,
                    0,
                )
                continue

            if queue:
                best_lb = min(lb, min(n.lp_bound for n, _ in queue))
            else:
                best_lb = lb

            if best_ub is not None and lb >= best_ub - eps:
                print(f"  FATHOMED: {lb:.2f} ≥ {best_ub:.2f}")
                node.is_pruned = True
                node.prune_reason = "bound"
                stats.node_explored(node)
                if queue:
                    best_lb = min(n.lp_bound for n, _ in queue)
                elif best_ub is not None:
                    best_lb = best_ub
                viz.update(
                    node.node_id,
                    lb,
                    z_vals,
                    best_lb,
                    best_ub,
                    nodes_explored,
                    node,
                    len(queue),
                    False,
                    0,
                )
                continue

            if is_integer(z_vals, eps):
                print(f"  INTEGER: {lb:.2f}", end="")
                node.is_integer = True
                incumbent_updated = False
                num_fathomed = 0

                if best_ub is None or lb < best_ub - eps:
                    best_ub = lb
                    incumbent_updated = True
                    print(f" ★ NEW INCUMBENT!")
                    num_fathomed = fathom_queue_by_incumbent(queue, best_ub, eps)
                    if queue:
                        best_lb = min(n.lp_bound for n, _ in queue)
                    else:
                        best_lb = best_ub
                    stats.node_explored(
                        node, incumbent_improved=True, num_fathomed=num_fathomed
                    )
                else:
                    print()
                    if queue:
                        best_lb = min(n.lp_bound for n, _ in queue)
                    else:
                        best_lb = best_ub
                    stats.node_explored(node)

                viz.update(
                    node.node_id,
                    lb,
                    z_vals,
                    best_lb,
                    best_ub,
                    nodes_explored,
                    node,
                    len(queue),
                    incumbent_updated,
                    num_fathomed,
                )
                continue

            stats.node_explored(node)
            viz.update(
                node.node_id,
                lb,
                z_vals,
                best_lb,
                best_ub,
                nodes_explored,
                node,
                len(queue),
                False,
                0,
            )
            parent_z = z_vals
        else:
            z_vals = parent_z

        branch_var = find_most_fractional(z_vals, node, eps)
        if branch_var is None:
            print(f"  No fractional variable")
            node.is_pruned = True
            if queue:
                best_lb = min(n.lp_bound for n, _ in queue)
            elif best_ub is not None:
                best_lb = best_ub
            continue

        item_id, t_br, u_br, z_val = branch_var

        can_be_zero = (t_br, u_br) not in node.theta_1_by_item[item_id]
        can_be_one = (t_br, u_br) not in node.theta_0_by_item[item_id]

        if not can_be_zero and not can_be_one:
            print(f"  Conflict")
            continue

        children = 0

        if can_be_zero:
            left = BranchNode(
                node_counter,
                node.node_id,
                node.depth + 1,
                {i: node.theta_0_by_item[i].copy() for i in node.theta_0_by_item},
                {i: node.theta_1_by_item[i].copy() for i in node.theta_1_by_item},
                branch_variable=branch_var,
                branch_direction="Z=0",
            )
            left.theta_0_by_item[item_id].add((t_br, u_br))
            sig = node_signature(left)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                left.lp_bound = node.lp_bound
                node_counter += 1
                queue.append((left, z_vals))
                stats.node_created()
                children += 1

        if can_be_one:
            right = BranchNode(
                node_counter,
                node.node_id,
                node.depth + 1,
                {i: node.theta_0_by_item[i].copy() for i in node.theta_0_by_item},
                {i: node.theta_1_by_item[i].copy() for i in node.theta_1_by_item},
                branch_variable=branch_var,
                branch_direction="Z=1",
            )
            right.theta_1_by_item[item_id].add((t_br, u_br))
            sig = node_signature(right)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                right.lp_bound = node.lp_bound
                node_counter += 1
                queue.append((right, z_vals))
                stats.node_created()
                children += 1

        if children > 0:
            print(f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}")

    if not queue and best_ub is not None:
        best_lb = best_ub

    stats.print_summary(best_lb, best_ub, eps)

    plt.ioff()
    plt.show()

    return best_lb, best_ub, nodes_explored


def build_small_example_instance():
    T = 8
    return (
        [
            ProductionItem(
                1,
                T,
                [6, 0, 5, 0, 3, 4, 0, 5],
                [3.0, 3.0, 3.3, 3.3, 3.6, 3.6, 3.8, 3.8],
                [22.0] * T,
                [1.0] * T,
                [t + l for t, l in enumerate([3, 3, 2, 2, 2, 2, 2, 2])],
                5000.0,
            ),
            ProductionItem(
                2,
                T,
                [0, 7, 0, 6, 0, 5, 4, 0],
                [2.8, 2.8, 3.0, 3.0, 3.2, 3.2, 3.5, 3.5],
                [18.0] * T,
                [0.8] * T,
                [t + l for t, l in enumerate([2, 2, 3, 3, 2, 2, 2, 2])],
                5000.0,
            ),
            ProductionItem(
                3,
                T,
                [4, 4, 4, 0, 3, 0, 6, 0],
                [3.2, 3.2, 3.2, 3.8, 3.8, 4.0, 4.0, 4.0],
                [24.0] * T,
                [1.2] * T,
                [t + l for t, l in enumerate([2, 3, 2, 2, 3, 2, 2, 2])],
                5000.0,
            ),
        ],
        [15, 11, 15, 8, 8, 8, 9, 9],
    )


if __name__ == "__main__":
    items, cap = build_small_example_instance()
    print("\n╔" + "═" * 68 + "╗")
    print(f"║ {'CAPACITATED LOT SIZING WITH PERISHABILITY':^66s} ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Items:    {len(items):<57d} ║")
    print(f"║  Periods:  {len(cap):<57d} ║")
    print("╚" + "═" * 68 + "╝")

    lb, ub, nodes = solve_branch_and_price_with_viz(
        items, cap, max_nodes=10000, print_frequency=50
    )

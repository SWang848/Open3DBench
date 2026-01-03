from typing import List, Optional, Tuple, Dict, Any
import networkx as nx
import os
import sys
import random
import logging
import math
import time
import json
import copy
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import distance
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import PlaceDB
import Params


UPPER_DIE = 1
BOTTOM_DIE = 0

def graph_construction(db):
    G = nx.MultiGraph()
    # G = nx.Graph()
    
    # nodes
    node_attrs = {}
    mean_node_area = 0.
    num = 0
    for node_name in db.node_names:
        node = db.node_name2id_map[node_name.decode('utf-8')]
        if node < (db.num_physical_nodes - db.num_terminal_NIs):  # exclude IO ports
            G.add_node(node)
            node_area = db.node_size_x[node] * db.node_size_y[node]
            node_attrs[node] = {"is_macro": False, 
                                "area": node_area,    # scale the area of cells
                                "name": node_name.decode('utf-8'),
                                "partition": BOTTOM_DIE}
            
            mean_node_area += node_area
            num += 1
            
    mean_node_area = mean_node_area / num
    # detect macros
    for node_name in db.node_names:
        node = db.node_name2id_map[node_name.decode('utf-8')]
        if node < (db.num_physical_nodes - db.num_terminal_NIs):  # exclude IO ports
            node_area = db.node_size_x[node] * db.node_size_y[node]
            if (node_area > (mean_node_area * 10)) and (db.node_size_y[node] > (db.row_height * 2)):
                node_attrs[node]["is_macro"] = True
                
    nx.set_node_attributes(G, node_attrs)

    # edges
    edges = []
            
    for net_name in db.net_names:
        net = db.net_name2id_map[net_name.decode('utf-8')]
        pins = db.net2pin_map[net]
        connected_nodes = []
        
        for pin in pins:
            if db.pin2node_map[pin] < (db.num_physical_nodes - db.num_terminal_NIs):  # exclude IO ports
                connected_nodes.append(db.pin2node_map[pin])
        if len(pins) < 10:
            edges.extend(combinations(connected_nodes, r=2))

    for edge in edges:
        G.add_edge(edge[0], edge[1])
    return G
        
class HierarchyNode:
    """
    Represents one node in the hierarchy (a cluster or a macro)
    """
    def __init__(self, hierarchy_name: str, parent: 'HierarchyNode' = None):
        self.hierarchy_name: str = hierarchy_name
        self.parent: 'HierarchyNode' | None = parent
        self.children: List['HierarchyNode'] = []
        
        self.node_id: Optional[str] = None
        self.total_area: float = 0
    
    @property
    def is_leaf(self) -> bool:
        return not self.children
    
    def get_all_leaf_descendants(self) -> List['HierarchyNode']:
        if self.is_leaf:
            return [self]
        all_leaves = []
        for child in self.children:
            all_leaves.extend(child.get_all_leaf_descendants())
        
        return all_leaves

class HierarchyTree:
    """
    reads a .def file and a networkx graph to construct the tree from hierarchy information.
    """
    def __init__(self, placedb: PlaceDB, case_name: str) -> None:
        self.placedb: PlaceDB = placedb
        self.root: HierarchyNode = HierarchyNode(hierarchy_name="root")

        self.construct_hierarchy_tree()
        self._calculate_cluster_areas(self.root)
        # Plot the hierarchy tree structure
        self.save_ascii_tree(os.path.join(f"./hmsa_results/{case_name}", "hierarchy_tree.txt"))
        
    def construct_hierarchy_tree(self) -> None:        
        for node_name in self.placedb.node_names:
            node_id = self.placedb.node_name2id_map[node_name.decode('utf-8')]
            if 'macro' in node_name.decode('utf-8') and self.placedb.node_size_y[node_id] > self.placedb.row_height * 2:
                hierarchy_path = node_name.decode('utf-8').split("__")
                self._add_node_from_path(hierarchy_path, node_id)

    def _add_node_from_path(self, hierarchy_path: List[str], node_id: int) -> None:
        """
        It walks the tree and adds nodes
        """
        current_node = self.root
        current_path_str = ""
        
        for i, part_name in enumerate(hierarchy_path):
            if i == 0:
                hierarchy_path_name = part_name
            else:
                hierarchy_path_name = current_path_str + "__" + part_name
            current_path_str = hierarchy_path_name
                        
            found_child = None
            for child in current_node.children:
                if child.hierarchy_name == hierarchy_path_name:
                    found_child = child
                    break
            
            if found_child:
                current_node = found_child
            else:
                new_node = HierarchyNode(hierarchy_name=hierarchy_path_name, parent=current_node)
                current_node.children.append(new_node)
                current_node = new_node
        
        current_node.node_id = node_id
        current_node.total_area = self.placedb.node_size_x[node_id] * self.placedb.node_size_y[node_id]
    
    def _calculate_cluster_areas(self, node: HierarchyNode) -> None:
        if node.is_leaf:
            return node.total_area
        
        cluster_area = 0
        for child in node.children:
            cluster_area += self._calculate_cluster_areas(child)
        
        node.total_area = cluster_area
        return cluster_area
        
    def get_nodes_by_level(self) -> Dict[int, List[HierarchyNode]]:
        levels = {}
        def traverse(node: HierarchyNode, level: int):
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
            for child in node.children:
                traverse(child, level + 1)
        
        for child in self.root.children:
            traverse(child, 1)
        return levels
    
    def get_ascii_tree(self):
        lines = []
        self._build_ascii_recursive(self.root, prefix="", is_last=True, lines_list=lines)
        return "\n".join(lines)
    
    def _build_ascii_recursive(self, node: HierarchyNode, prefix: str, is_last: bool, lines_list: List[str]):
        """Recursive helper for building the ASCII tree string."""
        connector = "└── " if is_last else "├── "
        line = ""

        if node.is_leaf:
            line = (f"{prefix}{connector}{node.hierarchy_name} "
                    f"[Leaf, ID: {node.node_id}, Area: {node.total_area:.0f}]")
        else:
            if node.parent is not None: 
                line = (f"{prefix}{connector}{node.hierarchy_name} "
                        f"[Cluster, Area: {node.total_area:.0f}]")
            else:
                line = f"{node.hierarchy_name} [Root, Area: {node.total_area:.0f}]"
        
        lines_list.append(line)

        new_prefix = prefix + ("    " if is_last else "│   ")
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            self._build_ascii_recursive(child, new_prefix, (i == child_count - 1), lines_list)
            
    def save_ascii_tree(self, save_path: str):
        """
        Saves the human-readable ASCII tree to a .txt file.
        """
        try:
            tree_string = self.get_ascii_tree()
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(tree_string)
            print(f"\nASCII Tree successfully saved to '{save_path}'")
        except Exception as e:
            print(f"\nError saving ASCII tree to '{save_path}': {e}")    

class HMSA:
    def __init__(self, G:nx.MultiGraph, placedb: PlaceDB, case_name: str, grid_size: int=40) -> None:
        self.G: nx.MultiGraph = G
        self.placedb: PlaceDB = placedb
        self.hierarchy_tree: HierarchyTree = HierarchyTree(placedb, case_name)
        self.root = self.hierarchy_tree.root
        self.total_area = self.hierarchy_tree.root.total_area
        self.nodes_by_level = self.hierarchy_tree.get_nodes_by_level()
        self.max_depth = max(self.nodes_by_level.keys())
        self.current_area_balance = [0.0, 0.0]
        self.current_solution = [[], []] # [bottom_die_macros_ids, upper_die_macros_ids]
        self.pareto_archive_grid = {} # keys: (cell_x, cell_y), values: (cost, solution)
        self.grid_size = grid_size
        self._initial_partition()

        self.current_total_cutsize = self._calculate_initial_cutsize()
        self.current_imbalance = self._get_imbalance_metric()
        log_cut = math.log1p(self.current_total_cutsize)
        log_imbalance = math.log1p(self.current_imbalance)
        self.norm_bounds = {
            'log_cut': {'min': log_cut, 'max': log_cut},
            'log_imbalance': {'min': log_imbalance, 'max': log_imbalance}
        }
        self._update_pareto_archive((self.current_total_cutsize, self.current_imbalance), self.current_solution)
                            
    def _initial_partition(self):
        partitions = {}
        for node_id in self.G.nodes():
            node_area = self.placedb.node_size_x[node_id] * self.placedb.node_size_y[node_id]
            if self.G.nodes[node_id].get('is_macro'):
                if random.random() < 0.5:
                    part = UPPER_DIE
                    self.current_area_balance[UPPER_DIE] += node_area
                    partitions[node_id] = {'partition': part}
                    self.current_solution[UPPER_DIE].append(node_id)
                else:
                    part = BOTTOM_DIE
                    self.current_area_balance[BOTTOM_DIE] += node_area
                    partitions[node_id] = {'partition': part}
                    self.current_solution[BOTTOM_DIE].append(node_id)
            # node_area includes std cells, but the partition only counts macros
            else:
                self.current_area_balance[BOTTOM_DIE] += node_area

        nx.set_node_attributes(self.G, partitions)
        
    def _calculate_initial_cutsize(self) -> int:
        cut = 0
        for u, v in self.G.edges():
            if self.G.nodes[u]['partition'] != self.G.nodes[v]['partition']:
                cut += 1
        return cut
    
    def _get_imbalance_metric(self) -> float:
        return abs(self.current_area_balance[UPPER_DIE] - self.current_area_balance[BOTTOM_DIE])
    
    def calculate_deltas(self, node_to_move: HierarchyNode) -> Tuple[int, float]:
        """
        Calculates the change in cutsize and area when moving a node to the other die.
        """
        leaf_nodes = node_to_move.get_all_leaf_descendants()
        leaf_macro_ids = [leaf.node_id for leaf in leaf_nodes]
        leaf_macro_set = set(leaf_macro_ids)
        
        from_part = self.G.nodes[leaf_macro_ids[0]]['partition']
        to_part = 1 - from_part
        delta_area = node_to_move.total_area
        delta_cut = 0
        
        for macro_id in leaf_macro_ids:
            if macro_id in self.current_solution[to_part]:
                delta_area -= self.placedb.node_size_x[macro_id] * self.placedb.node_size_y[macro_id]
            else:
                for neighbor_id in self.G.neighbors(macro_id):
                    edge_count = self.G.number_of_edges(macro_id, neighbor_id)
                    if self.G.nodes[neighbor_id]['partition'] == to_part:
                        delta_cut -= edge_count
                    else:
                        if neighbor_id in leaf_macro_set:
                            continue
                        delta_cut += edge_count
        
        return delta_cut, delta_area
    
    def commit_move(self, node_to_move: HierarchyNode, delta_cut: int, delta_area: float) -> None:
        leaf_macro_ids = [leaf.node_id for leaf in node_to_move.get_all_leaf_descendants()]
        from_part = self.G.nodes[leaf_macro_ids[0]]['partition']
        to_part = 1 - from_part
        
        for macro_id in leaf_macro_ids:
            if macro_id in self.current_solution[to_part]:
                pass
            else:
                self.current_solution[from_part].remove(macro_id)
                self.current_solution[to_part].append(macro_id)
        new_partitions = {macro_id: {'partition': to_part} for macro_id in leaf_macro_ids}
        nx.set_node_attributes(self.G, new_partitions)
        
        self.current_total_cutsize += delta_cut
        self.current_area_balance[from_part] -= delta_area
        self.current_area_balance[to_part] += delta_area
        self.current_imbalance = self._get_imbalance_metric()
    
    def _get_normalized_cost(self, cut: int, imbalance: float) -> Tuple[float, float]:
        """
        normalizes the log costs.
        """
        log_cut = math.log1p(cut)
        log_imbalance = math.log1p(imbalance)
        
        cut_range = self.norm_bounds['log_cut']['max'] - self.norm_bounds['log_cut']['min']
        imbalance_range = self.norm_bounds['log_imbalance']['max'] - self.norm_bounds['log_imbalance']['min']
        
        normalized_cut = 0.5 if cut_range == 0 else (log_cut - self.norm_bounds['log_cut']['min']) / cut_range
        normalized_imbalance = 0.5 if imbalance_range == 0 else (log_imbalance - self.norm_bounds['log_imbalance']['min']) / imbalance_range
        return normalized_cut, normalized_imbalance
    
    def _update_norm_bounds(self, cut: int, imbalance: float) -> None:
        """
        updates the bounds for the logarithmic costs and remaps archive if bounds changed
        """
        log_cut = math.log1p(cut)
        log_imbalance = math.log1p(imbalance)
        
        old_bounds_min_cut = self.norm_bounds['log_cut']['min']
        old_bounds_max_cut = self.norm_bounds['log_cut']['max']
        old_bounds_min_imbalance = self.norm_bounds['log_imbalance']['min']
        old_bounds_max_imbalance = self.norm_bounds['log_imbalance']['max']
        
        self.norm_bounds['log_cut']['min'] = min(self.norm_bounds['log_cut']['min'], log_cut)
        self.norm_bounds['log_cut']['max'] = max(self.norm_bounds['log_cut']['max'], log_cut)
        self.norm_bounds['log_imbalance']['min'] = min(self.norm_bounds['log_imbalance']['min'], log_imbalance)
        self.norm_bounds['log_imbalance']['max'] = max(self.norm_bounds['log_imbalance']['max'], log_imbalance)
        
        # Only remap if bounds actually changed
        bounds_changed = (old_bounds_min_cut != self.norm_bounds['log_cut']['min'] or 
                         old_bounds_max_cut != self.norm_bounds['log_cut']['max'] or
                         old_bounds_min_imbalance != self.norm_bounds['log_imbalance']['min'] or
                         old_bounds_max_imbalance != self.norm_bounds['log_imbalance']['max'])
        
        if bounds_changed:
            self._remap_pareto_archive()
    
    def _update_pareto_archive(self, cost: Tuple[int, float], solution: List[List[int]]) -> None:
        log_cut = math.log1p(cost[0])
        log_imbalance = math.log1p(cost[1])
        
        min_log_cut = self.norm_bounds['log_cut']['min']
        max_log_cut = self.norm_bounds['log_cut']['max']
        min_log_imbalance = self.norm_bounds['log_imbalance']['min']
        max_log_imbalance = self.norm_bounds['log_imbalance']['max']
        
        cut_range = max_log_cut - min_log_cut
        imbalance_range = max_log_imbalance - min_log_imbalance
        
        cut_ratio = 0 if cut_range == 0 else (log_cut - min_log_cut) / (cut_range + 1e-9)
        imbalance_ratio = 0 if imbalance_range == 0 else (log_imbalance - min_log_imbalance) / (imbalance_range + 1e-9)
        
        cell_x = min(self.grid_size-1, int(cut_ratio * self.grid_size))
        cell_y = min(self.grid_size-1, int(imbalance_ratio * self.grid_size))
        cell = (cell_x, cell_y)
        if cell in self.pareto_archive_grid:
            existing_cost, _ = self.pareto_archive_grid[cell]
            if cost[0] <= existing_cost[0] and cost[1] <= existing_cost[1]:
                self.pareto_archive_grid[cell] = (cost, copy.deepcopy(solution))
        else:  
            self.pareto_archive_grid[cell] = (cost, copy.deepcopy(solution))
    
    def _remap_pareto_archive(self) -> None:
        old_archive_grid = list(self.pareto_archive_grid.values())
        self.pareto_archive_grid = {}
        for cost, solution in old_archive_grid:
            self._update_pareto_archive(cost, solution)
    
    def reset_state_from_solution(self, solution: List[List[int]]) -> None:
        """
        Re-initialize current state (partitions, balances, cutsize, imbalance) from a given solution
        while preserving the existing pareto archive and normalization bounds.
        """
        partitions: Dict[int, Dict[str, int]] = {}
        self.current_solution = [[], []]
        self.current_area_balance = [0.0, 0.0]
        
        bottom_ids: List[int] = solution[BOTTOM_DIE]
        upper_ids: List[int] = solution[UPPER_DIE]
        bottom_set = set(bottom_ids)
        upper_set = set(upper_ids)
        
        for node_id in self.G.nodes():
            node_area = self.placedb.node_size_x[node_id] * self.placedb.node_size_y[node_id]
            if self.G.nodes[node_id].get('is_macro'):
                if node_id in upper_set:
                    partitions[node_id] = {'partition': UPPER_DIE}
                    self.current_area_balance[UPPER_DIE] += node_area
                    self.current_solution[UPPER_DIE].append(node_id)
                else:
                    partitions[node_id] = {'partition': BOTTOM_DIE}
                    self.current_area_balance[BOTTOM_DIE] += node_area
                    self.current_solution[BOTTOM_DIE].append(node_id)
            else:
                # std cells count toward bottom die area balance in current formulation
                self.current_area_balance[BOTTOM_DIE] += node_area
        
        nx.set_node_attributes(self.G, partitions)
        
        # Update current metrics and archive without clearing existing solutions
        self.current_total_cutsize = self._calculate_initial_cutsize()
        self.current_imbalance = self._get_imbalance_metric()
        self._update_norm_bounds(self.current_total_cutsize, self.current_imbalance)
        self._update_pareto_archive((self.current_total_cutsize, self.current_imbalance), self.current_solution)
            
    def _snapshot_state(
        self,
        solution: List[List[int]],
        cut: int,
        imbalance: float,
        accepted: bool,
        temperature: float,
    ) -> Dict[str, Any]:
        return {
            "cost": [
                int(cut),
                float(imbalance),
            ],
            "solution": [
                copy.deepcopy(solution[BOTTOM_DIE]),
                copy.deepcopy(solution[UPPER_DIE]),
            ],
            "accepted": bool(accepted),
            "temperature": float(temperature),
        }
    
    def _build_candidate_solution(
        self,
        node_to_move: HierarchyNode,
        from_part: int,
        to_part: int,
    ) -> List[List[int]]:
        candidate_solution = [
            copy.deepcopy(self.current_solution[BOTTOM_DIE]),
            copy.deepcopy(self.current_solution[UPPER_DIE]),
        ]
        leaf_macro_ids = [leaf.node_id for leaf in node_to_move.get_all_leaf_descendants()]
        for macro_id in leaf_macro_ids:
            if macro_id in candidate_solution[from_part]:
                candidate_solution[from_part].remove(macro_id)
            if macro_id not in candidate_solution[to_part]:
                candidate_solution[to_part].append(macro_id)
        return candidate_solution
    
    def run_annealing(self, T_max=500, T_min=0.1, gamma=0.95, steps_per_T=100, history: Optional[List[Dict[str, Any]]] = None) -> None:
        T = T_max
        available_levels = sorted(self.nodes_by_level.keys())
    
        if history is not None:
            history.append(
                self._snapshot_state(
                    solution=self.current_solution,
                    cut=self.current_total_cutsize,
                    imbalance=self.current_imbalance,
                    accepted=True,
                    temperature=T,
                )
            )

        while T > T_min:
            for _ in range(steps_per_T):
                temp_ratio = (T - T_min) / (T_max - T_min)
                
                weights = []
                baseline_weight = 0.1
                
                for level in available_levels:
                    # Avoid division by zero if max_depth == 1
                    if self.max_depth > 1:
                        normal_depth = (level - 1) / (self.max_depth - 1) # 1 is finest move, 0 is coarsest move
                    else:
                        normal_depth = 0.5

                    coarse_bias = (1.0 - normal_depth) * temp_ratio
                    fine_bias = normal_depth * (1.0 - temp_ratio)
                    
                    final_weight = baseline_weight + coarse_bias + fine_bias
                    weights.append(final_weight)
                
                chosen_level = random.choices(available_levels, weights=weights, k=1)[0]
                node_to_move = random.choice(self.nodes_by_level[chosen_level])
                
                delta_cut, delta_area = self.calculate_deltas(node_to_move)
                
                new_cut = self.current_total_cutsize + delta_cut
                from_part = self.G.nodes[node_to_move.get_all_leaf_descendants()[0].node_id]['partition']
                to_part = 1 - from_part
                
                # Calculate new imbalance without modifying state
                temp_area_balance = self.current_area_balance.copy()
                temp_area_balance[from_part] -= delta_area
                temp_area_balance[to_part] += delta_area
                new_imbalance = abs(temp_area_balance[UPPER_DIE] - temp_area_balance[BOTTOM_DIE])
                
                self._update_norm_bounds(new_cut, new_imbalance)
                    
                curr_norm_cut, curr_norm_imbalance = self._get_normalized_cost(self.current_total_cutsize, self.current_imbalance)
                new_norm_cut, new_norm_imbalance = self._get_normalized_cost(new_cut, new_imbalance)
                
                is_better = (new_norm_cut < curr_norm_cut and new_norm_imbalance <= curr_norm_imbalance) \
                or (new_norm_cut <= curr_norm_cut and new_norm_imbalance < curr_norm_imbalance)
                
                is_worse = (new_norm_cut > curr_norm_cut and new_norm_imbalance >= curr_norm_imbalance) \
                or (new_norm_cut >= curr_norm_cut and new_norm_imbalance > curr_norm_imbalance)
                
                accept = False
                if is_better:
                    accept = True
                elif not is_worse:
                    accept = True
                else:
                    delta_cost = (new_norm_cut - curr_norm_cut) + (new_norm_imbalance - curr_norm_imbalance)
                    norm_temp = T / T_max
                    if math.exp(-delta_cost / norm_temp) > random.random():
                        accept = True
                        
                candidate_solution = self._build_candidate_solution(node_to_move, from_part, to_part)
                if history is not None:
                    history.append(
                        self._snapshot_state(
                            solution=candidate_solution,
                            cut=new_cut,
                            imbalance=new_imbalance,
                            accepted=accept,
                            temperature=T,
                        )
                    )

                if accept:
                    self.commit_move(node_to_move, delta_cut, delta_area)
                    self._update_pareto_archive((new_cut, new_imbalance), self.current_solution)
            T = T * gamma
        return self.pareto_archive_grid
    
    def generate_regression_dataset(
        self,
        output_path: str,
        T_max: float = 500,
        T_min: float = 0.1,
        gamma: float = 0.95,
        steps_per_T: int = 100,
    ) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        self.run_annealing(T_max=T_max, T_min=T_min, gamma=gamma, steps_per_T=steps_per_T, history=history)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w") as fp:
            json.dump(history, fp, indent=2)
        logging.info(f"Saved regression dataset with {len(history)} samples to {output_path}")
        return history
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hierarchy-aware Multi-Objective Simulated Annealing.")
    parser.add_argument("params", type=Path, help="Path to params JSON used by PlaceDB.")
    parser.add_argument("--output", type=Path, default=None, help="Path to save HMSA results.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    params = Params.Params()
    
    case_name = args.params.stem
    out_dir = f"./hmsa_results/{case_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    # load parameters
    params.load(args.params)
    params.placed_def_input = ""
    logging.info("parameters loaded successfully")
    
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    
    logging.info(f"Found {placedb.num_physical_nodes - placedb.num_terminal_NIs} positioned components")
    
    G = graph_construction(placedb)
    hmsa = HMSA(G, placedb, case_name=case_name)
    # hmsa.generate_regression_dataset(os.path.join(out_dir, "regression_dataset.json"))
    
    pareto_archive = hmsa.run_annealing()
    
    # # First-round candidates
    # candidates = candidate_selection(pareto_archive)
    
    # # Refinement phase: keep archive, reinitialize from each candidate, run at low temperature
    # refine_T_max = 1.0
    # refine_T_min = 0.05
    # refine_gamma = 0.9
    # refine_steps = 400
    # for _, val in candidates.items():
    #     _, solution = val
    #     hmsa.reset_state_from_solution(solution)
    #     hmsa.run_annealing(T_max=refine_T_max, T_min=refine_T_min, gamma=refine_gamma, steps_per_T=refine_steps)
    
    # # Update archive and candidates after refinement
    pareto_archive = hmsa.pareto_archive_grid
    print([(key, value[0]) for key, value in pareto_archive.items()])
    # Save pareto_archive to a single file
    results_path = os.path.join(out_dir, "hmsa_results.json")
    results = {
        "pareto_archive": {
            "description": "Complete Pareto archive containing all non-dominated solutions found during HMSA optimization",
            "solutions": {str(key): {"cost": value[0], "solution": value[1]} for key, value in pareto_archive.items()}
        },
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"HMSA results (Pareto archive and candidates) saved to {results_path}")
    
if __name__ == "__main__":
    main()
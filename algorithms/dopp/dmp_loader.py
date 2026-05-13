import logging
import os

import numpy as np

from algorithms.dopp._place3d_bridge import PLACE3D_ROOT, REPO_ROOT, Params, PlaceDB


def _resolve_dreamplace_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.join(PLACE3D_ROOT, path)


class DreamPlaceLoader:
    """Load the DREAMPlace database and expose macro information lazily."""

    def __init__(self, benchmark, upper_die_macros=None, def_path=None, rand_init=True):
        self.benchmark = benchmark
        self.macros = None
        self.dmp_params = self._load_dmp_params(
            rand_init=rand_init,
        )

        if def_path is not None:
            self.dmp_params.def_input = os.path.abspath(def_path)

        self.dmp_placedb = PlaceDB.PlaceDB()
        self.dmp_placedb(self.dmp_params, upper_die_macros=upper_die_macros)
    
    def _load_dmp_params(self, rand_init=True):
        params = Params.Params()
        json_path = os.path.join(PLACE3D_ROOT, "test", "or_3D", f"{self.benchmark}_3D.json")
        params.load(json_path)
        params.placed_def_input = ""

        project_path_fields = [
            "aux_input",
            "def_input",
            "verilog_input",
            "early_lib_input",
            "late_lib_input",
        ]
        project_list_path_fields = [
            "lef_input",
            "lib_input",
        ]
        for field in project_path_fields:
            value = getattr(params, field, None)
            if isinstance(value, str):
                setattr(params, field, _resolve_dreamplace_path(value))
        for field in project_list_path_fields:
            value = getattr(params, field, None)
            if isinstance(value, list):
                setattr(params, field, [_resolve_dreamplace_path(item) for item in value])

        if isinstance(getattr(params, "verilog_input", None), str) and not os.path.exists(params.verilog_input):
            logging.warning("Ignoring missing verilog_input: %s", params.verilog_input)
            params.verilog_input = None

        if isinstance(getattr(params, "detailed_place_engine", None), str):
            resolved_engine = _resolve_dreamplace_path(params.detailed_place_engine)
            if os.path.exists(resolved_engine):
                params.detailed_place_engine = resolved_engine
            elif getattr(params, "detailed_place_flag", 0):
                raise FileNotFoundError(f"Detailed placement engine not found: {resolved_engine}")
            else:
                logging.warning("Ignoring missing detailed_place_engine because detailed_place_flag=0: %s", resolved_engine)
                params.detailed_place_engine = None

        params.random_center_init_flag = rand_init

        return params
    
    def determine_macro(self):
        if self.macros is not None:
            return self.macros

        placedb = self.dmp_placedb
        physical_node_limit = placedb.num_physical_nodes - placedb.num_terminal_NIs
        physical_nodes = []

        for node_name in placedb.node_names:
            node_name_str = node_name.decode("utf-8") if isinstance(node_name, bytes) else str(node_name)
            node_id = placedb.node_name2id_map[node_name_str]
            if node_id < physical_node_limit:
                physical_nodes.append(node_id)

        if not physical_nodes:
            self.macros = []
            return self.macros

        avg_area = np.mean([
            placedb.node_size_x[node_id] * placedb.node_size_y[node_id]
            for node_id in physical_nodes
        ])

        self.macros = [
            node_id
            for node_id in physical_nodes
            if (
                placedb.node_size_x[node_id] * placedb.node_size_y[node_id] > 10 * avg_area
                or placedb.node_size_y[node_id] > 2 * placedb.row_height
            )
        ]
        return self.macros

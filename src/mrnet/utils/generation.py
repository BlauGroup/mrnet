import os
import numpy as np
from monty.serialization import dumpfn, loadfn
import copy
import pickle


def generate_pre_find_path_files(rn, PRs, cost_from_start, old_solved_PRs, min_cost):
    pickle_in = open(
        os.path.join(test_dir, "unittest_RN_pr_ii_4_before_find_path_cost.pkl"),
        "wb",
    )
    pickle.dump(rn, pickle_in)
    pickle_in = open(
        os.path.join(test_dir, "unittest_find_path_cost_PRs_IN.pkl"),
        "wb",
    )
    pickle.dump(PRs, pickle_in)
    dumpfn(
        cost_from_start,
        os.path.join(test_dir, "unittest_find_path_cost_cost_from_start_IN.json"),
    )
    dumpfn(
        old_solved_PRs,
        os.path.join(test_dir, "unittest_find_path_cost_old_solved_prs_IN.json"),
    )
    dumpfn(
        min_cost,
        os.path.join(test_dir, "unittest_find_path_cost_min_cost_IN.json"),
    )


def generate_pre_id_solved_PRs_files(rn, PRs, cost_from_start, solved_PRs):
    pickle_in = open(
        os.path.join(
            test_dir,
            "unittest_RN_pr_ii_4_before_identify_solved_PRs.pkl",
        ),
        "wb",
    )
    pickle.dump(rn, pickle_in)
    with open(
        os.path.join(test_dir, "unittest_find_path_cost_PRs_IN.pkl"),
        "wb",
    ) as handle:
        pickle.dump(PRs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dumpfn(
        cost_from_start,
        os.path.join(
            test_dir,
            "unittest_identify_solved_PRs_cost_from_start_IN.json",
        ),
    )
    dumpfn(
        solved_PRs,
        os.path.join(test_dir, "unittest_identify_solved_PRs_solved_PRs_IN.json"),
    )


def generate_characterize_path_files(rn, old_solved_PRs, dist_and_path):
    pickle_in = open(
        os.path.join(
            test_dir,
            "unittest_RN_before_characterize_path.pkl",
        ),
        "wb",
    )
    pickle.dump(rn, pickle_in)
    pickle_in = open(
        os.path.join(test_dir, "unittest_characterize_path_PRs_IN.pkl"),
        "wb",
    )
    pickle.dump(old_solved_PRs, pickle_in)
    dumpfn(
        dist_and_path,
        os.path.join(
            test_dir,
            "unittest_characterize_path_path_IN.json",
        ),
    )


def generate_pre_update_eweights_files(rn, min_cost):
    pickle_in = open(
        os.path.join(
            test_dir,
            "unittest_RN_pr_ii_4_before_update_edge_weights.pkl",
        ),
        "wb",
    )
    pickle.dump(rn, pickle_in)
    pickle_in = open(
        os.path.join(test_dir, "unittest_update_edge_weights_orig_graph_IN.pkl"),
        "wb",
    )
    pickle.dump(rn.graph, pickle_in)
    dumpfn(
        min_cost,
        os.path.join(test_dir, "unittest_update_edge_weights_min_cost_IN.json"),
    )

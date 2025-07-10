import mesa
import pandas as pd

from language.model import NLanguageModel

RESULTS = "results/"

# # Parameters for individual update algorithm
# params_individual = {
#     "num_nodes": [20, 40, 80],
#     "N": 3,
#     "update_algorithm": "individual",
#     "agent_state_mode": "onehot",
#     "grammar_percentage": 0.2,
#     "graph_seed": range(100),
#     "graph_type": ["barabasi_albert", "small_world"],
# }

# # Parameters for threshold update algorithm
# params_threshold = {
#     "num_nodes": [20, 40, 80],
#     "N": 3,
#     "update_algorithm": "threshold",
#     "threshold_val": [[0.6, 0.2, 0.2], [0.3, 0.35, 0.35], [0.1, 0.45, 0.45]],
#     "sink_state_idx": [None, 0],  # Only test sink state for state 0
#     "agent_state_mode": "onehot",
#     "grammar_percentage": 0.2,
#     "graph_seed": list(range(100)),
#     "graph_type": ["barabasi_albert", "small_world"],
# }

# Parameters for reward update algorithm
params_reward = {
    "num_nodes": [20, 40, 80],
    "N": 3,
    "update_algorithm": "reward",
    "alpha": [0, 0.025, 0.05],
    "sink_state_idx": 0,
    "agent_state_mode": "onehot",
    "grammar_percentage": 0.2,
    "graph_seed": list(range(100)),
    "graph_type": ["barabasi_albert", "small_world"],
}


def main():
    # # Individual
    # results_ind = mesa.batch_run(
    #     NLanguageModel,
    #     parameters=params_individual,
    #     data_collection_period=10,
    #     number_processes=None,
    #     max_steps=400,
    # )
    # pd.DataFrame(results_ind).to_csv(RESULTS + "nlang-3state-individual.csv")

    # # Threshold
    # results_thr = mesa.batch_run(
    #     NLanguageModel,
    #     parameters=params_threshold,
    #     data_collection_period=1,
    #     number_processes=None,
    #     max_steps=40,
    # )
    # pd.DataFrame(results_thr).to_csv(RESULTS + "nlang-3state-threshold.csv")

    # Reward
    results_rew = mesa.batch_run(
        NLanguageModel,
        parameters=params_reward,
        data_collection_period=25,
        number_processes=None,
        max_steps=1000,
    )
    pd.DataFrame(results_rew).to_csv(RESULTS + "nlang-3state-reward-1.csv")


if __name__ == "__main__":
    main()

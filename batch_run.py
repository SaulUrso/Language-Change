import mesa
import pandas as pd

from language.model import LanguageModel

RESULTS = "results/"

params_individual = {
    "num_nodes": [20, 40, 80],
    "update_algorithm": "individual",
    "agent_state_mode": "binary",
    "grammar_percentage": 0.3,
    "graph_seed": range(100),
    "graph_type": ["barabasi_albert", "small_world"],
}

params_threshold = {
    "num_nodes": [20, 40, 80],
    "update_algorithm": "threshold",
    "threshold_val": [0.3, 0.5, 0.8],
    "sink_state_1": [True, False],
    "agent_state_mode": "binary",
    "grammar_percentage": 0.3,
    "graph_seed": range(100),
    "graph_type": ["barabasi_albert", "small_world"],
}

params_reward_1 = {
    "num_nodes": [20, 40, 80],
    "update_algorithm": "reward",
    "agent_state_mode": "intermediate",
    "grammar_percentage": [[0.1, 0.2], [0.2, 0.1]],
    "logistic": [True],
    "alpha": [0.025, 0.05, 0.0],
    "graph_seed": range(100),
    "graph_type": ["barabasi_albert", "small_world"],
}

params_reward_2 = {
    "num_nodes": [20, 40, 80],
    "update_algorithm": "reward",
    "agent_state_mode": "intermediate",
    "grammar_percentage": [[0.1, 0.2], [0.2, 0.1]],
    "logistic": [False],
    "graph_seed": range(100),
    "graph_type": ["barabasi_albert", "small_world"],
}

params_reward_1_bin = {
    "num_nodes": [20, 40, 80],
    "update_algorithm": "reward",
    "agent_state_mode": "binary",
    "grammar_percentage": [0.3],
    "logistic": [True],
    "alpha": [0.025, 0.05, 0.0],
    "graph_seed": range(100),
    "graph_type": ["barabasi_albert", "small_world"],
}

params_reward_2_bin = {
    "num_nodes": [20, 40, 80],
    "update_algorithm": "reward",
    "agent_state_mode": "binary",
    "grammar_percentage": [0.3],
    "logistic": [False],
    "graph_seed": range(100),
    "graph_type": ["barabasi_albert", "small_world"],
}


def main():
    results = mesa.batch_run(
        LanguageModel,
        parameters=params_reward_1,
        data_collection_period=25,
        number_processes=None,
    )
    results2 = mesa.batch_run(
        LanguageModel,
        parameters=params_reward_2,
        data_collection_period=25,
        number_processes=None,
    )
    results_df = pd.DataFrame(results)
    results2_df = pd.DataFrame(results2)
    results_df = pd.concat([results_df, results2_df], ignore_index=True)
    results_df.to_csv(RESULTS + "reward-int.csv")

    results_bin_1 = mesa.batch_run(
        LanguageModel,
        parameters=params_reward_1_bin,
        data_collection_period=25,
        number_processes=None,
    )
    results_bin_2 = mesa.batch_run(
        LanguageModel,
        parameters=params_reward_2_bin,
        data_collection_period=25,
        number_processes=None,
    )
    results_bin_1_df = pd.DataFrame(results_bin_1)
    results_bin_2_df = pd.DataFrame(results_bin_2)
    results_bin_df = pd.concat([results_bin_1_df, results_bin_2_df], ignore_index=True)
    results_bin_df.to_csv(RESULTS + "reward-bin.csv")

    results = mesa.batch_run(
        LanguageModel,
        parameters=params_threshold,
        data_collection_period=1,
        number_processes=None,
        max_steps=40,
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS + "threshold-1.csv")

    results = mesa.batch_run(
        LanguageModel,
        parameters=params_individual,
        data_collection_period=10,
        number_processes=None,
        max_steps=400,
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS + "individual.csv")


if __name__ == "__main__":
    main()

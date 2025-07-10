import mesa
import networkx as nx
import numpy as np
from mesa.discrete_space import Network

from language.agent import LanguageAgent, NLanguageAgent

# def compute_average_state(model):
#     agent_states = [agent.state for agent in model.agents]
#     return sum(agent_states) / len(agent_states)


# Data collection: mean state vector
def compute_average_state(model):
    agent_states = np.array([agent.state for agent in model.agents])
    return agent_states.mean(axis=0)


def compute_state_amount(chosen_state):
    def inner(model):
        agent_states = [agent.state for agent in model.agents]
        return agent_states.count(chosen_state) / len(agent_states)

    return inner


def make_graph(graph_type, num_nodes, seed, **kwargs):
    """
    Create a graph of the specified type.
    graph_type: "barabasi_albert" or "small_world"
    num_nodes: number of nodes
    seed: random seed
    kwargs: additional arguments for graph creation
    """
    if graph_type == "barabasi_albert":
        m = kwargs.get("m", 1)
        return nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    elif graph_type == "small_world":
        k = kwargs.get("k", 2)
        p = kwargs.get("p", 0.5)
        return nx.newman_watts_strogatz_graph(num_nodes, k, p, seed=seed)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


class LanguageModel(mesa.Model):

    def __init__(
        self,
        num_nodes=10,
        # update algorithm args
        update_algorithm="individual",
        threshold_val=None,
        sink_state_1=False,
        logistic=True,
        alpha=0.025,
        # grammar generation
        agent_state_mode="binary",
        grammar_percentage=None,
        # graph arguments
        graph_type="barabasi_albert",
        graph_seed=0,
        # generic model arguments
        seed=0,
        rng=None,
        **kwargs,
    ):
        super().__init__(seed=seed, rng=rng)

        graph = make_graph(
            graph_type, 
            num_nodes,
            seed=int(graph_seed),
            **kwargs)
        
        self.grid = Network(
            graph, 
            capacity=1,
            random=self.random)

        if agent_state_mode == "binary":
            if grammar_percentage is None:
                agent_states = [self.random.randint(0, 1) for _ in range(num_nodes)]
            else:
                num_ones = int(num_nodes * grammar_percentage)
                num_zeros = num_nodes - num_ones
                agent_states = [1] * num_ones + [0] * num_zeros
                self.random.shuffle(agent_states)
        elif agent_state_mode == "intermediate":
            if grammar_percentage is None:
                agent_states = [
                    self.random.choice([0, 0.5, 1]) for _ in range(num_nodes)
                ]
            else:
                num_ones = int(num_nodes * grammar_percentage[0])
                num_halves = int(num_nodes * grammar_percentage[1])
                num_zeros = num_nodes - num_ones - num_halves
                agent_states = [1] * num_ones + [0.5] * num_halves + [0] * num_zeros
                self.random.shuffle(agent_states)

        # data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Mean State": compute_average_state,
            },
            agent_reporters={"State": "state"},
        )

        # agent creation
        LanguageAgent.create_agents(
            self,
            num_nodes,
            agent_states,
            update_algorithm,
            threshold_val,
            sink_state_1,
            logistic,
            alpha,
            list(self.grid.all_cells),
        )

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")


class NLanguageModel(mesa.Model):
    def __init__(
        self,
        num_nodes=10,
        N=3,  # number of possible states
        update_algorithm="individual",
        threshold_val=None,  # now a vector of length N, summing to 1
        sink_state_idx=None,
        logistic=True,
        alpha=0.025,
        agent_state_mode="onehot",
        grammar_percentage=None,
        graph_type="barabasi_albert",
        graph_seed=0,
        seed=0,
        rng=None,
        **kwargs,
    ):
        super().__init__(seed=seed, rng=rng)
        graph = make_graph(graph_type, num_nodes, seed=int(graph_seed), **kwargs)
        self.grid = Network(graph, capacity=1, random=self.random)

        # Agent state initialization
        if agent_state_mode == "onehot":
            if grammar_percentage is None:
                agent_states = [
                    np.eye(N)[self.random.randint(0, N - 1)] for _ in range(num_nodes)
                ]
            else:
                # grammar percentage specifies only the percentage of nodes of state zero
                # assume the other grammars get the remaining probability weight divided equally
                num_state0 = int(num_nodes * grammar_percentage)
                remaining = num_nodes - num_state0

                num_per_other = remaining // (N - 1)
                extras = remaining % (N - 1)
                agent_states = [np.eye(N)[0]] * num_state0
                for i in range(1, N):
                    count = num_per_other + (1 if i <= extras else 0)
                    agent_states += [np.eye(N)[i]] * count
                self.random.shuffle(agent_states)

        else:
            raise ValueError(f"Unknown agent_state_mode: {agent_state_mode}")

        # threshold_val: if None, use uniform vector
        if threshold_val is None:
            threshold_val_vec = np.ones(N) / N
        else:
            threshold_val_vec = threshold_val

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Mean State Vector": compute_average_state,
            },
            # agent_reporters={"State": "state"},
        )

        NLanguageAgent.create_agents(
            self,
            num_nodes,
            agent_states,
            update_algorithm,
            threshold_val_vec,
            sink_state_idx,
            alpha,
            list(self.grid.all_cells),
        )

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")

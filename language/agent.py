import numpy as np
from mesa.discrete_space import FixedAgent
from scipy.special import expit


def softmax_with_bias(state, alpha, preferred_index):
    """
    Compute softmax over a 3-state distribution,
    with sharpness and bias controlled by a single alpha parameter.

    Parameters:
        state (list of float): a 3-element vector that sums to 1
        alpha (float): controls both gain (sharpness) and bias
        preferred_index (int): index (0, 1, or 2) to bias toward

    Returns:
        chosen_state (int): the index (0, 1, or 2) selected
        probabilities (list of float): softmax probabilities
    """
    epsilon = 0.1
    gain = (abs(alpha) + epsilon) * 30  # N * 10

    N = len(state)

    # Calculate scores with gain and bias
    scores = []
    for i, s in enumerate(state):
        bias = 1 / N - 1 / gain  # 1 / N
        bias *= 1 if i == preferred_index else (-1 / (N - 1))  # -1/(N-1)
        score = (s + bias) * (gain / N * 2)  # * 5
        scores.append(score)

    # Compute softmax probabilities
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores)

    # Sample one category
    chosen_state = np.random.choice(3, p=probabilities)

    return chosen_state, probabilities.tolist()


class LanguageAgent(FixedAgent):

    def __init__(
        self,
        model,
        initial_state,
        update_algorithm,
        threshold_val,
        sink_state_1,
        logistic,
        alpha,
        cell,
    ):
        super().__init__(model)

        self.cell = cell
        self.update_algorithm = update_algorithm
        self.state = initial_state

        # for threshold
        self.threshold_val = threshold_val
        self.sink_state_1 = sink_state_1

        # for reward
        self.logistic = logistic
        self.alpha = alpha

    def adopt_rand_neighbor_grammar(self):
        neigh_cell = self.cell.neighborhood.select_random_cell()
        neigh = neigh_cell.agents[0]
        self.state = neigh.state

    def adopt_threshold_grammar(self):
        neighbors = [n for n in self.cell.neighborhood.agents]
        grammar_1_neighs = [n for n in neighbors if n.state == 1]

        if len(grammar_1_neighs) / len(neighbors) >= self.threshold_val:
            self.state = 1
        else:
            if not self.sink_state_1:
                self.state = 0

    def speak(self):
        if self.logistic:
            gain = (self.alpha + 0.1) * 20
            filter_val = expit((gain * self.state - 1) * 5)
            if self.model.random.random() <= filter_val:
                spoken_state = 1
            else:
                spoken_state = 0

        else:
            biased_val = 1.5 * self.state
            if biased_val >= 1:
                biased_val = 1

            if self.model.random.random() <= biased_val:
                spoken_state = 1
            else:
                spoken_state = 0

        return spoken_state

    def listen(self, heard_state):
        gamma = 0.01
        prob = self.model.random.random()

        if prob <= self.state:
            self.state += gamma * (heard_state - self.state)
        else:
            self.state = gamma * heard_state + (1 - gamma) * self.state

    def step(self):
        if self.update_algorithm == "individual":
            self.adopt_rand_neighbor_grammar()
        elif self.update_algorithm == "threshold":
            self.adopt_threshold_grammar()

        elif self.update_algorithm == "reward":
            spoken_state = self.speak()
            for n in self.cell.neighborhood.agents:
                n.listen(spoken_state)
        else:
            raise ValueError(
                "Update algorithm must be one of 'individual' 'threshold' or 'reward'"
            )


class NLanguageAgent(FixedAgent):
    def __init__(
        self,
        model,
        initial_state,  # should be a one-hot numpy array of length N
        update_algorithm,
        threshold_val,  # now a vector of length N, summing to 1
        sink_state_idx,
        alpha,
        cell,
    ):
        super().__init__(model)
        self.cell = cell
        self.update_algorithm = update_algorithm
        self.state = np.array(initial_state, dtype=float)  # one-hot vector
        self.N = len(initial_state)
        # If threshold_val is a scalar, convert to uniform vector
        if np.isscalar(threshold_val):
            self.threshold_val = np.ones(self.N) / self.N
        else:
            self.threshold_val = np.array(threshold_val, dtype=float)
            # self.threshold_val = self.threshold_val / self.threshold_val.sum()
        self.sink_state_idx = sink_state_idx  # index of sink state, or None
        self.alpha = alpha

    def adopt_rand_neighbor_grammar(self):
        neigh_cell = self.cell.neighborhood.select_random_cell()
        neigh = neigh_cell.agents[0]
        self.state = neigh.state

    def adopt_threshold_grammar(self):
        neighbors = [n for n in self.cell.neighborhood.agents]
        state_counts = np.sum([n.state for n in neighbors], axis=0)
        proportions = state_counts / len(neighbors)
        # For each state, check if its proportion exceeds its threshold
        above_threshold = proportions >= self.threshold_val

        if self.sink_state_idx is not None and self.state[self.sink_state_idx] == 1:
            # If in sink state, do not change state
            return
        # there is a change
        if np.any(above_threshold):
            # take most freq state among neighbors
            max_idx = np.argmax(proportions * above_threshold)
            self.state = np.zeros(self.N)
            self.state[max_idx] = 1

    def speak(self):
        # Use softmax_with_bias, biasing toward sink_state_idx if set, else no bias
        if self.sink_state_idx is not None:
            preferred_index = self.sink_state_idx
        else:
            raise ValueError("sink_state_idx must be specified.")
        chosen_state, probabilities = softmax_with_bias(
            self.state, self.alpha, preferred_index
        )
        spoken_state = np.zeros(self.N)
        spoken_state[chosen_state] = 1
        return spoken_state

    def listen(self, heard_state):
        gamma = 0.01
        extracted_idx = self.model.random.choices(range(self.N), weights=self.state)[0]
        heard_idx = np.argmax(heard_state)
        if extracted_idx == heard_idx:
            delta = gamma
        else:
            delta = -gamma

        self.state[extracted_idx] += delta
        for i in range(self.N):
            if i != extracted_idx:
                self.state[i] -= delta / (self.N - 1)

        self.state = np.clip(self.state, 0, 1)
        total = self.state.sum()
        self.state /= total

    def step(self):
        if self.update_algorithm == "individual":
            self.adopt_rand_neighbor_grammar()
        elif self.update_algorithm == "threshold":
            self.adopt_threshold_grammar()
        elif self.update_algorithm == "reward":
            spoken_state = self.speak()
            for n in self.cell.neighborhood.agents:
                n.listen(spoken_state)
        else:
            raise ValueError(
                "Update algorithm must be one of 'individual' 'threshold' or 'reward'"
            )

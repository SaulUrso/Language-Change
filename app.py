import solara
from matplotlib.figure import Figure
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_space_component,
)
from mesa.visualization.utils import update_counter

from language.model import LanguageModel


def agent_portrayal(agent):
    state = getattr(agent, "state", 0)
    gray_level = 1 - state
    color = (
        1 - gray_level,
        0,
        gray_level,
    )
    return {
        "color": color,
        "size": 30,
    }


model_params = {
    "update_algorithm": {
        "type": "Select",
        "label": "Update Algorithm",
        "values": ["individual", "threshold", "reward"],
        "value": "individual",
    },
    "threshold_val": Slider(
        label="Threshold",
        value=0.3,
        min=0.0,
        max=1.0,
        step=0.01,
        dtype=float,
    ),
    "sink_state_1": {
        "type": "Checkbox",
        "label": "Sink State 1 (threshold only)",
        "value": False,
    },
    "logistic": {
        "type": "Checkbox",
        "label": "Logistic (reward only)",
        "value": True,
    },
    "alpha": Slider(
        label="Alpha (reward only)",
        value=0.025,
        min=0.0,
        max=0.05,
        step=0.005,
        dtype=float,
    ),
    "grammar_percentage": Slider(
        label="Grammar 1 proportion",
        value=0.3,
        min=0,
        max=1,
        step=0.01,
        dtype=float,
    ),
    "agent_state_mode": {
        "type": "Select",
        "label": "Agent State Mode",
        "values": ["binary", "intermediate"],
        "value": "binary",
    },
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Grammar seed",
    },
    "num_nodes": Slider(
        label="Number of agents",
        value=10,
        min=10,
        max=100,
        step=1,
    ),
    "graph_seed": {
        "type": "InputText",
        "value": 42,
        "label": "Graph seed",
    },
    "graph_type": {
        "type": "Select",
        "label": "Graph Type",
        "values": ["barabasi_albert", "small_world"],
        "value": "barabasi_albert",
    },
    "m": Slider(
        label="Barabasi-Albert m",
        value=1,
        min=1,
        max=10,
        step=1,
    ),
    "k": Slider(
        label="Small World k",
        value=4,
        min=2,
        max=20,
        step=2,
    ),
    "p": Slider(
        label="Small World p",
        value=0.1,
        min=0.0,
        max=1.0,
        step=0.01,
    ),
}


# @solara.component
# def Histogram(model):
#     update_counter.get()  # This is required to update the counter
#     # Note: you must initialize a figure using this method instead of
#     # plt.figure(), for thread safety purpose
#     fig = Figure()
#     ax = fig.subplots()
#     wealth_vals = [agent.wealth for agent in model.agents]
#     # Note: you have to use Matplotlib's OOP API instead of plt.hist
#     # because plt.hist is not thread-safe.
#     ax.hist(wealth_vals, bins=10)
#     solara.FigureMatplotlib(fig)


@solara.component
def StatesPercentages(model):

    fig = Figure()

    update_counter.get()  # Ensure reactivity

    data = model.datacollector.get_model_vars_dataframe()
    n_cols = len(data.columns)
    axs = fig.subplots(n_cols, 1, squeeze=False).flatten()[:n_cols]

    # Plot each of the three model collection functions as a lineplot
    for i, col in enumerate(data.columns[:3]):
        axs[i].plot(data.index, data[col])
        axs[i].set_ylabel(col)
        axs[i].set_xlabel("Step")
        axs[i].legend()

    fig.tight_layout()

    solara.FigureMatplotlib(fig)


@solara.component
def Page():
    # Create initial model instance
    model = LanguageModel()

    SpaceGraph = make_space_component(agent_portrayal)

    page = SolaraViz(
        model,
        components=[SpaceGraph, StatesPercentages],
        model_params=model_params,
        name="Language Model",
    )
    # This is required to render the visualization in the Jupyter notebook
    return page

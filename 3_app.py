import numpy as np
import solara
from matplotlib.figure import Figure
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_space_component,
)
from mesa.visualization.utils import update_counter

from language.model import NLanguageModel


def agent_portrayal(agent):
    state = getattr(agent, "state", np.zeros(3))
    # Color: red, green, blue for 3 grammars
    color = tuple(float(x) for x in state)
    return {
        "color": color,
        "size": 30,
    }


# Sliders for each threshold value
threshold_sliders = [
    Slider(
        label=f"Threshold Grammar {i}",
        value=1 / 3,
        min=0.0,
        max=1.0,
        step=0.01,
        dtype=float,
    )
    for i in range(3)
]


def get_threshold_val():
    return [slider.value for slider in threshold_sliders]


model_params = {
    "update_algorithm": {
        "type": "Select",
        "label": "Update Algorithm",
        "values": ["individual", "threshold", "reward"],
        "value": "individual",
    },
    "sink_state_idx": {
        "type": "Select",
        "label": "Sink State (threshold only)",
        "values": [None, 0, 1, 2],
        "value": None,
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
        label="Percent of state 0",
        value=0.3,
        min=0.0,
        max=1,
        step=0.01,
        dtype=float,
    ),
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


@solara.component
def StatesPercentages(model):
    fig = Figure()
    update_counter.get()  # Ensure reactivity
    data = model.datacollector.get_model_vars_dataframe()
    # data is of the form 1 column containing a list of 3 elements in each entry
    percentages = np.zeros((1, 3)) if data.empty else np.array(data.iloc[:, 0].tolist())
    if not data.empty:
        ax = fig.subplots(3, 1, sharex=True)
        colors = ["red", "green", "blue"]
        labels = ["State 0", "State 1", "State 2"]
        steps = np.arange(len(percentages))
        for i in range(3):
            ax[i].plot(steps, percentages[:, i], color=colors[i])
            ax[i].set_ylabel(labels[i])
            ax[i].set_ylim(0, 1)
        ax[-1].set_xlabel("Step")
        fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def Page():
    model = NLanguageModel(N=3)
    SpaceGraph = make_space_component(agent_portrayal)
    page = SolaraViz(
        model,
        components=[SpaceGraph, StatesPercentages],
        model_params=model_params,
        name="3-Grammar Language Model",
    )
    return page

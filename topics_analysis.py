import plotly.graph_objs as go
from ipywidgets import interactive, HBox, VBox
import plotly.offline as py
import pandas as pd

with open("colors.txt", 'r') as f:
    lines = f.read()
from random import shuffle
colors = lines.replace('\n', '').replace(' ','').split(',')[:64]
shuffle(colors)


def plot_topics_pies(df, group='date', path=None):

    color_mapping = {t : c for t, c in zip(df['topics'].unique(), colors)}
    fig = go.Figure()

    # Add traces, one for each slider step
    labels = []
    for n, grp in df.groupby(group):
        fig.add_trace(
            go.Pie(
                visible=False,
                name=f"{group} = " + str(n),
                labels=grp['topics'],
                values=grp['topic_counts'],
            marker_colors=grp['topics'].map(color_mapping))
        )
        labels.append(n)

    # Make 1st trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Slider switched to {group}: " + str(labels[i])}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": f"{group}: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        autosize=False,
        width=1000,
        height=1000
    )

    fig.show()
    if path is not None:
        fig.write_html(path)
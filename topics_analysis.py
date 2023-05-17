import plotly.graph_objs as go
from ipywidgets import interactive, HBox, VBox
import plotly.offline as py
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def set_up_mapping(topics=None):

    if topics is None:
        with open("wikipedia_core_events_semantic/topics_list.txt", 'r') as f:
            lines = f.read()
        topics = lines.replace('\n', '').replace("'", '').split(',')

    viridis = mpl.colormaps['viridis'].resampled(23)  # geography
    plasma = mpl.colormaps['plasma'].resampled(22)  # culture
    spring = mpl.colormaps['spring'].resampled(7) # history
    cool = mpl.colormaps['cool'].resampled(12) # stem

    color_mapping = {}
    color_mapping.update({t : c for t, c in zip([t for t in topics if 'geography' in t], viridis.colors)})
    color_mapping.update({t: c for t, c in zip([t for t in topics if 'culture' in t], plasma.colors)})
    color_mapping.update({t: c for t, c in zip([t for t in topics if 'history' in t], spring(np.arange(0,spring.N)))})
    color_mapping.update({t: c for t, c in zip([t for t in topics if 'stem' in t], cool(np.arange(0,cool.N)))})

    return color_mapping

def plot_temporal(df, kind='bar', group='date', labels='topics', values='topic_counts',
                     mapping=True, log=False, path=None):

    color_mapping = set_up_mapping()
    fig = go.Figure()

    def add_trace():
        if kind == 'bar':
            fig.add_trace(go.Bar(
                visible=False,
                name=f"{group} = " + str(n),
                x=grp[labels],
                y=grp[values]
        ))
        elif kind == 'pie':
            if mapping:
                fig.add_trace(
                    go.Pie(
                        visible=False,
                        name=f"{group} = " + str(n),
                        labels=grp[labels],
                        values=grp[values],
                        marker_colors=grp[labels].map(color_mapping))
                )
            else:
                fig.add_trace(
                    go.Pie(
                        visible=False,
                        name=f"{group} = " + str(n),
                        labels=grp[labels],
                        values=grp[values])
                )


    # Add traces, one for each slider step
    l = []
    for n, grp in df.groupby(group):
        add_trace()
        l.append(n)

    # Make 1st trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Slider switched to {group}: " + str(l[i])}],  # layout attribute
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
    if log and kind == 'bar':
        fig.update_yaxes(type="log")

    fig.show()
    if path is not None:
        fig.write_html(path)

def plot_simple(df, kind='bar', labels='topics', values='topic_counts', mapping=True, log=False, path=None):

    color_mapping = set_up_mapping()
    fig = go.Figure()

    def add_trace():
        if kind == 'bar':
            fig.add_trace(go.Bar(
                x=df[labels],
                y=df[values]
            ))
        elif kind == 'pie':
            if mapping:
                fig.add_trace(
                    go.Pie(
                        labels=df[labels],
                        values=df[values],
                        marker_colors=df[labels].map(color_mapping))
                )
            else:
                fig.add_trace(
                    go.Pie(
                        labels=df[labels],
                        values=df[values])
                )

    add_trace()
    fig.update_layout(
        autosize=False,
        width=1200,
        height=1000
    )
    if log and kind == 'bar':
        fig.update_yaxes(type="log")

    fig.show()
    if path is not None:
        fig.write_html(path)

def plot_perc_topics(df, fontsize=10, path="/home/descourt/interm_results/data_presentation/perc_pages_90_topics_highvol.jpg"):

    fig, axs = plt.subplots(figsize=(20,15), nrows=2, ncols=2, sharex=True)
    df = df.sort_values('unique_topic')

    for ax, (_, group) in zip(axs.flatten(), df.groupby('subtopic')):
        n = len(group['color'].values)
        for std, mean, color, pos in zip(group[('90_perc_views_rank', 'std')],group[('90_perc_views_rank', 'mean')], group['color'], range(n)):
            ax.plot((mean - 1.96*std/np.sqrt(93) ,mean + 1.96*std/np.sqrt(93)), (pos,pos),'o-',color=color)
        ax.set_yticks(range(n), list(group['unique_topic']), fontsize=fontsize)
        ax.tick_params('x', labelbottom=True)

    fig.supylabel('Topics', fontsize=15)
    fig.supxlabel('Percentage of pages', fontsize=15)
    fig.suptitle('Percentage of pages contributing to 90% of views per topic', fontsize=15)

    fig.tight_layout()
    plt.show()
    plt.savefig(path)
"""
3D visualization primitives based on Plotly.
We might want to instead use a more powerful library like Open3D.
Plotly however supports animations, buttons and sliders.

1) Initialize a figure with `fig = init_figure()`
2) Plot points, cameras, lines, or create a slider animation.
3) Call `fig.show()` to render the figure.
"""

import plotly.graph_objects as go
import numpy as np

from ..pixlib.geometry.utils import to_homogeneous


def init_figure(height=800):
    """Initialize a 3D figure."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        scene_camera=dict(
            eye=dict(x=0., y=-.1, z=-2), up=dict(x=0, y=-1., z=0)),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            aspectmode='data', dragmode='orbit'),
        margin=dict(l=0, r=0, b=0, t=0, pad=0))  # noqa E741
    return fig


def plot_points(fig, pts, color='rgba(255, 0, 0, 1)', ps=2):
    """Plot a set of 3D points."""
    x, y, z = pts.T
    tr = go.Scatter3d(
        x=x, y=y, z=z, mode='markers', marker_size=ps,
        marker_color=color, marker_line_width=.2)
    fig.add_trace(tr)


def plot_camera(fig, R, t, K, color='rgb(0, 0, 255)'):
    """Plot a camera as a cone with camera frustum."""
    x, y, z = t
    u, v, w = R @ -np.array([0, 0, 1])
    tr = go.Cone(
        x=[x], y=[y], z=[z], u=[u], v=[v], w=[w], anchor='tip',
        showscale=False, colorscale=[[0, color], [1, color]],
        sizemode='absolute')
    fig.add_trace(tr)

    W, H = K[0, 2]*2, K[1, 2]*2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners/2) @ R.T + t
    x, y, z = corners.T
    tr = go.Scatter3d(
        x=x, y=y, z=z, line=dict(color='rgba(0, 0, 0, .5)'),
        marker=dict(size=0.0001), showlegend=False)
    fig.add_trace(tr)


def create_slider_animation(fig, traces):
    """Create a slider that animates a list of traces (e.g. 3D points)."""
    slider = {'steps': []}
    frames = []
    fig.add_trace(traces[0])
    idx = len(fig.data) - 1
    for i, tr in enumerate(traces):
        frames.append(go.Frame(name=str(i), traces=[idx], data=[tr]))
        step = {"args": [
                [str(i)],
                {"frame": {"redraw": True},
                 "mode": "immediate"}],
                "label": i,
                "method": "animate"}
        slider['steps'].append(step)
    fig.frames = tuple(frames)
    fig.layout.sliders = (slider,)

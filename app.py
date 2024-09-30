# app.py

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from kmeans import KMeans

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global variables
X = np.random.randn(300, 2)
initial_centroids = []
kmeans_instance = None
current_step = 0

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("KMeans Clustering Visualization"),
            dbc.Row([
                dbc.Col([
                    html.Label("Initialization Method"),
                    dcc.Dropdown(
                        id='init-method',
                        options=[
                            {'label': 'Random', 'value': 'random'},
                            {'label': 'Farthest First', 'value': 'farthest'},
                            {'label': 'KMeans++', 'value': 'kmeans++'},
                            {'label': 'Manual', 'value': 'manual'}
                        ],
                        value='random',
                        clearable=False
                    ),
                ], width=6),
                dbc.Col([
                    html.Label("Number of Clusters (k)"),
                    dcc.Input(
                        id='num-clusters',
                        type='number',
                        value=3,
                        min=1,
                        step=1,
                        style={'width': '100%'}
                    ),
                ], width=6),
            ]),
            html.Br(),
            dbc.Button("Generate New Dataset", id='generate-dataset', color='primary', className='mr-2'),
            dbc.Button("Step", id='step-button', color='secondary', className='mr-2'),
            dbc.Button("Run to Convergence", id='run-button', color='success', className='mr-2'),
            dbc.Button("Reset", id='reset-button', color='danger'),
            html.Br(), html.Br(),
            dcc.Graph(id='cluster-graph', figure=go.Figure()),
        ], width=12)
    ])
])

# Callback to update graph
@app.callback(
    Output('cluster-graph', 'figure'),
    Input('step-button', 'n_clicks'),
    Input('run-button', 'n_clicks'),
    Input('generate-dataset', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input('cluster-graph', 'clickData'),
    State('init-method', 'value'),
    State('num-clusters', 'value'),
    prevent_initial_call=True
)
def update_graph(step_clicks, run_clicks, gen_clicks, reset_clicks, clickData, init_method, num_clusters):
    global X, initial_centroids, kmeans_instance, current_step

    triggered_id = ctx.triggered_id

    # Validate the number of clusters
    if num_clusters is None or num_clusters < 1:
        num_clusters = 1  # Default to 1 if invalid input

    if triggered_id == 'generate-dataset':
        X = np.random.randn(300, 2)
        initial_centroids = []
        kmeans_instance = None
        current_step = 0
        fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color='grey')))
        return fig

    elif triggered_id == 'reset-button':
        initial_centroids = []
        kmeans_instance = None
        current_step = 0
        fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color='grey')))
        return fig

    elif init_method == 'manual' and triggered_id == 'cluster-graph':
        if clickData:
            point = clickData['points'][0]
            initial_centroids.append([point['x'], point['y']])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color='grey')))
            centroids_array = np.array(initial_centroids)
            fig.add_trace(go.Scatter(x=centroids_array[:, 0], y=centroids_array[:, 1],
                                     mode='markers', marker=dict(color='red', symbol='x', size=12)))
            return fig
        else:
            return dash.no_update

    elif triggered_id in ['step-button', 'run-button']:
        if kmeans_instance is None:
            if init_method == 'manual':
                if not initial_centroids:
                    return dash.no_update
                initial_centroids_array = np.array(initial_centroids)
                n_clusters = len(initial_centroids)
                kmeans_instance = KMeans(n_clusters=n_clusters, init_method='manual',
                                         initial_centroids=initial_centroids_array)
            else:
                kmeans_instance = KMeans(n_clusters=num_clusters, init_method=init_method)
            kmeans_instance.fit(X)
            current_step = 0

        if triggered_id == 'step-button':
            if current_step < len(kmeans_instance.history):
                centroids, labels = kmeans_instance.history[current_step]
                current_step += 1
            else:
                centroids = kmeans_instance.centroids
                labels = kmeans_instance.labels
        else:
            centroids = kmeans_instance.centroids
            labels = kmeans_instance.labels
            current_step = len(kmeans_instance.history)

        fig = go.Figure()
        for idx in range(kmeans_instance.n_clusters):
            cluster_points = X[labels == idx]
            if cluster_points.size > 0:
                fig.add_trace(go.Scatter(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    mode='markers',
                    name=f'Cluster {idx + 1}'
                ))
        fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1],
                                 mode='markers', marker=dict(color='black', symbol='x', size=12),
                                 name='Centroids'))
        return fig

    else:
        fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color='grey')))
        return fig

if __name__ == '__main__':
    app.run_server(debug=False, port=3000, host='0.0.0.0')

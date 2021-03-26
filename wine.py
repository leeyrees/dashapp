import dash
from dash.dependencies import Input, Output, State

import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_table


app = dash.Dash(__name__, title="Dash App Wine Dataset")

import numpy as np
import pandas as pd
import json
from dash.dependencies import Input, Output


df_url = 'https://raw.githubusercontent.com/leeyrees/datasets/main/winequalityN.csv'
df = pd.read_csv(df_url).dropna()

PAGE_SIZE = 5


app.layout = html.Div([
    dash_table.DataTable(
    columns=[
        {'name': 'type', 'id': 'type', 'type': 'text'},
        {'name': 'fixed acidity', 'id': 'fixed acidity', 'type': 'numeric'},
        {'name': 'volatile acidity', 'id': 'volatile acidity', 'type': 'numeric'},
        {'name': 'citric acid', 'id': 'citric acid', 'type': 'numeric'},
        {'name': 'residual sugar', 'id' :'residual sugar', 'type': 'numeric'},
        {'name': 'chlorides', 'id' :'chlorides', 'type': 'numeric'},
        {'name': 'free sulfur dioxide', 'id' :'free sulfur dioxide', 'type': 'numeric'},
        {'name': 'total sulfur dioxide', 'id' :'total sulfur dioxide', 'type': 'numeric'},
        {'name': 'density', 'id' :'density', 'type': 'numeric'},
        {'name': 'pH', 'id': 'pH', 'type': 'numeric'},
        {'name': 'sulphates', 'id': 'sulphates', 'type': 'numeric'},
        {'name': 'alcohol', 'id': 'alcohol', 'type': 'numeric'},
        {'name': 'quality', 'id': 'quality', 'type': 'numeric'}
    ],
    data=df.to_dict('records'),
    filter_action='native',

    style_table={
        'height': 400,
    },
    style_data={
        'width': '150px', 'minWidth': '150px', 'maxWidth': '150px',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
    }
),
    dcc.Graph(id="scatter-plot"),
    html.P("quality"),
    dcc.RangeSlider(
        id='range-slider',
        min=3, max=9, step=1,
        marks={0: '3', 1: '4', 2:'5', 3:'6', 4:'7', 5:'8', 6: '9'},
        value=[0, 6]
    ),
])


@app.callback(
    Output("scatter-plot", "figure"), 
    [Input("range-slider", "value")])
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (df['quality'] > low) & (df['quality'] < high)
    fig = px.scatter(
        df[mask], x="total sulfur dioxide", y="alcohol", 
        color="type", size='pH', 
        hover_data=['quality'])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

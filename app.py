import dash
from dash.dependencies import Input, Output, State

import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_table

app = dash.Dash(__name__, title="Dash App")

import numpy as np
import pandas as pd
import json


df_url_1 = 'https://raw.githubusercontent.com/leeyrees/datasets/main/MyData.csv'
df_1 = pd.read_csv(df_url_1).dropna()

df_url_2 = 'https://raw.githubusercontent.com/leeyrees/datasets/main/winequalityN.csv'
df_2 = pd.read_csv(df_url_2).dropna()

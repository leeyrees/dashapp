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

df_major = df_1['Major'].sort_values().unique()
opt_major = [{'label': x + 'Major', 'value': x} for x in df_major]



min_bpwt = min(df_1['BackpackWeight'].dropna())
max_bpwt = max(df_1['BackpackWeight'].dropna())

def slider_map(min, max, steps=10):
    scale = np.logspace(np.log10(min), np.log10(max), steps, endpoint=False)
    return {i/10: '{}'.format(round(scale[i],2)) for i in range(steps)}

table_1_tab = dash_table.DataTable(
                id='my-table-1',
                columns=[{"name": i, "id": i} for i in df_1.columns]
            )
table_2_tab = dash_table.DataTable(
                id='my-table-2',
                columns=[{"name": i, "id": i} for i in df_2.columns]
            )


app.layout= html.Div([
    html.Div([html.H1(app.title, className="app-header--title")],
        className= "app-header",
    ),
    html.Div([  
        
        html.Label(["Select the major:", 
            dcc.Dropdown('my-dropdown', options= opt_major, value= [opt_major[0]['value']], multi=True)
        ]),
        html.Label(["Range of values for back pack weight:", 
                 dcc.RangeSlider(id="range",
                     max= 1,
                     min= 0,
                     step= 1/100,
                     marks= slider_map(min_bpwt, max_bpwt),
                     value= [0,1],
                 )
        ]),
    html.Div([ 
        html.Div(id='data', style={'display': 'none'}),
        html.Div(id='dataRange', style={'display': 'none'}),
        dcc.Tabs(id="tabs", value='tab-t', children=[
            dcc.Tab(label='Table about Backpack', value='tab-t1'),
            dcc.Tab(label='Table about Wine', value='tab-t2'),
        ]),
        html.Div(id='tabs-content')
    ],
    className= "app-body")
])
])
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-t1':
        return table_1_tab
    elif tab == 'tab-t2':
        return table_2_tab
@app.callback(
     Output('my-table-1', 'data'),
     Input('data', 'children'), 
     State('tabs', 'value'))

def update_table(data, tab):
    if tab != 'tab-t1':
        return None
    dff_1 = pd.read_json(data, orient='split')
    return dff_1.to_dict("records")

@app.callback(
     Output('my-table-2', 'data'),
     Input('data', 'children'), 
     State('tabs', 'value'))
def update_table2(data, tab):
    if tab != 'tab-t2':
        return None
    dff_2 = pd.read_json(data, orient='split')
    return dff_2.to_dict("records")
    

@app.callback(Output('data', 'children'), 
    Input('range', 'value'), 
    State('my-dropdown', 'value'))
def filter(range, values):
     filter = df_1['Major'].isin(values) & df_2['BackpackWeight'].between(min_bpwt , max_bpwt)

     # more generally, this line would be
     # json.dumps(cleaned_df)
     return df_1[filter].to_json(date_format='iso', orient='split')


@app.callback(Output('dataRange', 'children'), 
    Input('my-dropdown', 'value'))
def dataRange(values):
    filter = df_1['Major'].isin(values) 
    dff_1 = df[filter]
    min_bpwt = min(dff_1['BackpackWeight'].dropna())
    max_bpwt = max(dff_1['BackpackWeight'].dropna())
    return json.dumps({'min_bpwt': min_bpwt, 'max_bpwt': max_bpwt})




if  __name__ == '__main__':
    app.server.run(debug=True)
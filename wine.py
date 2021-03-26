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
    id='table-sorting-filtering',
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
    page_current= 0,
    page_size= PAGE_SIZE,
    page_action='custom',

    filter_action='custom',
    filter_query='',

    sort_action='custom',
    sort_mode='multi',
    sort_by=[]
),
    dcc.Graph(id="scatter-plot"),
    html.P("quality"),
    dcc.RangeSlider(
        id='range-slider',
        min=3, max=9, step=1,
        marks={3: '3', 4: '4', 5: '5', 6: '6', 7:'7', 8:'8', 9: '9'},
        value=[3, 9]
    ),
])

operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                
                return name, operator_type[0].strip(), value

    return [None] * 3


@app.callback(
    Output('table-sorting-filtering', 'data'),
    Input('table-sorting-filtering', "page_current"),
    Input('table-sorting-filtering', "page_size"),
    Input('table-sorting-filtering', 'sort_by'),
    Input('table-sorting-filtering', 'filter_query'))
def update_table(page_current, page_size, sort_by, filter):
    filtering_expressions = filter.split(' && ')
    dff = df
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff = dff.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )

    page = page_current
    size = page_size
    return dff.iloc[page * size: (page + 1) * size].to_dict('records')

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

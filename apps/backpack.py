import dash
from dash.dependencies import Input, Output, State
from app import app
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors


#app = dash.Dash(__name__, title="Dash App")

import numpy as np
import pandas as pd
import json
from dash.dependencies import Input, Output



df_url = 'https://raw.githubusercontent.com/leeyrees/datasets/main/MyData.csv'
df = pd.read_csv(df_url).dropna()
df['BackProblems'] = df['BackProblems'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Major'] = df['Major'].astype('category')
df['Status'] = df['Status'].astype('category')

X = df.Ratio.values[:, None]
X_train, X_test, y_train, y_test = train_test_split(
    X, df.BackProblems, random_state=42)

models = {'Regression': linear_model.LogisticRegression,
          'Decision Tree': tree.DecisionTreeClassifier,
          'k-NN': neighbors.KNeighborsClassifier}


PAGE_SIZE = 5
layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label = "Data table", children = [
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.H1("BACKPACK DATASET", className="text-center")
                    , className="mb-5 mt-5"),
                ]),
                dbc.Row([
                   html.Div([
                                    html.Br([]),
                                    html.H3(
                                        "\
                                    In this page we can find an analysis of the Backpack dataset, based on the previous \
                                    analysis made in the Shiny App of Leyre Sastre.",
                                        
                                        className="row",
                                    ),
                                ],),

               ]),
               dbc.Row([html.H6(
                   ["Data Table"], className= "subtitle padded"
                ),
                html.Br([]),
                dash_table.DataTable(
                    id='table-sorting-filtering',
                        columns=[
                            {'name': 'BackpackWeight', 'id': 'BackpackWeight', 'type': 'numeric'},
                            {'name': 'BodyWeight', 'id': 'BodyWeight', 'type': 'numeric'},
                            {'name': 'Ratio', 'id': 'Ratio', 'type': 'numeric'},
                            {'name': 'BackProblems', 'id': 'BackProblems', 'type': 'numeric'},
                            {'name': 'Major', 'id' :'Major', 'type': 'text'},
                            {'name': 'Year', 'id' :'Year', 'type': 'numeric'},
                            {'name': 'Sex', 'id' :'Sex', 'type': 'text'},
                            {'name': 'Status', 'id' :'Status', 'type': 'text'},
                            {'name': 'Units', 'id' :'Units', 'type': 'numeric'}
                        ],
                        page_current= 0,
                        page_size= PAGE_SIZE,
                        page_action='custom',

                        filter_action='custom',
                        filter_query='',

                        sort_action='custom',
                        sort_mode='multi',
                        sort_by=[],
                        style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                        style_cell={
                                    'backgroundColor': 'rgb(50, 50, 50)',
                                    'color': 'white'
                        },
                ),
            ]),
            ])
        ]),
        dcc.Tab(label = "Plots", children = [
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Scatter Plot"),

                        html.P("Select the first variable"),
                        html.Br([]),
                        dcc.Dropdown(
                            id = "cont-variable-1",
                            options=[{'label': i, 'value': i} for i in ['BackpackWeight', 'BodyWeight','Year','Units']],
                            placeholder ="Select the first variable: ", 
                            value = "BackpackWeight"
                        ),
                        html.Br([]),
                        html.P("Select the Second variable"),
                        dcc.Dropdown(
                            id = "cont-variable-2",
                            options = [{'label': i, 'value': i} for i in ['BackpackWeight', 'BodyWeight','Year','Units']],
                            placeholder ="Select the second variable: ", 
                            value = "BodyWeight"
                        ),
                        dcc.Graph(id="scatter-plot"),
                        html.P("Year"),
                        dcc.RangeSlider(
                            id='range-slider',
                            min=0, max=6, step=1,
                            marks={0: '0', 1: '1', 2: '2', 3: '3', 4:'4', 5:'5', 6: '6'},
                            value=[0, 6]
                        ),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Box Plot"),
                        html.P("X-axis:"),
                        dcc.Checklist(
                            id='x-axis', 
                            options=[{'value': x, 'label': x} 
                                for x in ['BackProblems', 'Major', 'Sex', 'Status']],
                            value=['BackProblems'], 
                            labelStyle={'display': 'inline-block'}
                        ),
                        html.P("Y-axis:"),
                        dcc.RadioItems(
                            id='y-axis', 
                            options=[{'value': x, 'label': x} 
                                    for x in ['BackpackWeight', 'BodyWeight', 'Ratio','Year','Units']],
                            value='BackpackWeight', 
                            labelStyle={'display': 'inline-block'}
                        ),
                        dcc.Graph(id="box-plot"),
                    ]),
                ]),
            ]),
        ]),
    dcc.Tab(label = "Regression and prediction", children = [
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Prediction of Back problems"),
                    html.Br([]),
                    html.P("Select Model:"),
                    dcc.Dropdown(
                    id='model-name',
                    options=[{'label': x, 'value': x} 
                            for x in models],
                    value='Regression',
                    clearable=False
                    ),
                    dcc.Graph(id="graph")
                ]),
            ]),
        ]),
    ])
    ])
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
    [Input("range-slider", "value"),
    Input('cont-variable-1', 'value'),
    Input('cont-variable-2', 'value')])
def update_bar_chart(slider_range, selected_var_1, selected_var_2):
    low, high = slider_range
    mask = (df['Year'] > low) & (df['Year'] < high)
    fig = px.scatter(
        df[mask], x=selected_var_1, y=selected_var_2, 
        color="BackProblems", size='Ratio', 
        hover_data=['Year'])
    return fig


@app.callback(
    Output("box-plot", "figure"), 
    [Input("x-axis", "value"), 
     Input("y-axis", "value")])
def generate_chart(x, y):
    fig = px.box(df, x=x, y=y)
    return fig


@app.callback(
    Output("graph", "figure"), 
    [Input('model-name', "value")])
def train_and_display(name):
    model = models[name]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')
    ])

    return fig

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', debug=True)


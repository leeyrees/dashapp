import dash
from dash.dependencies import Input, Output, State
from app import app
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_table 
import plotly.graph_objects as go


import numpy as np
import pandas as pd
import json
from dash.dependencies import Input, Output
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
import sklearn.metrics as metrics


df_url = 'https://raw.githubusercontent.com/leeyrees/datasets/main/winequalityN.csv'
df = pd.read_csv(df_url).dropna()
df['type'] = df['type'].astype('category')


PAGE_SIZE = 5

np.random.seed(0)
X = df.drop('type', axis = 1)

y = df['type'].map({'white': 1, 'red': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

MODELS = {'Logistic': linear_model.LogisticRegression,
          'Decision Tree': tree.DecisionTreeClassifier,
          'k-NN': neighbors.KNeighborsClassifier}




layout = html.Div([

        dcc.Tabs([
         
         dcc.Tab(label = "Wine data table", children = [ 

           dbc.Container([
               dbc.Row([
                   dbc.Col(html.H1("WINE DATASET", className="text-center")
                    , className="mb-5 mt-5"),
               ]),
               dbc.Row([
                   html.Div([
                                    html.Br([]),
                                    html.H3(
                                        "\
                                    In this page we are going to carry out an analysis of the wine dataset \
                                    from the Shiny App of Marta Ilundain (github: ENLACE).",
                                        
                                        className="row",
                                    ),
                                ],),

               ]),
                dbc.Row([ html.H6(
                                        ["Data table"], className="subtitle padded"
                                    ),
                                    html.Br([]),
                                    dash_table.DataTable(
                                    id='table-sorting-filtering1',
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
                                        sort_by=[],
                                    style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                                    style_cell={
                                        'backgroundColor': 'rgb(50, 50, 50)',
                                        'color': 'white'
                                    },
                                    ),

                ]),
           ])]),

            # Row 3
            dcc.Tab(label = "Wine data plots", children = [ 

                dbc.Container([

                    dbc.Row([
                        dbc.Col([
                            html.H3("Boxplot:"), 
                            html.H3("y-axis:"),
                            dcc.RadioItems(
                                id='y-axis', 
                                options=[{'value': x, 'label': x} 
                                        for x in ['fixed acidity', 'citric acid', 'chlorides', 'density']],
                                value='fixed acidity', 
                                labelStyle={'display': 'inline-block'}
                            ),
                            dcc.Graph(id="box-plot1"),
                         ] ),
                        
                        dbc.Col([
                            html.H3("Histogram"),
                            dcc.Dropdown(
                                id = "contvariable",
                                options=[{'label': i, 'value': i} for i in X.columns],
                                value='fixed acidity', 
                                placeholder ="Select a variable: "
                            ),
                            html.Br([]),

                            html.Div(className='submit', children=[

                                html.Button('Update chart', id='submit', n_clicks=0)

                            ]),
                            dcc.Graph(
                                id='hist',
                                figure={}
                            ),
                        ]),
                        dbc.Row([dbc.Col([
                            html.H3("Pie Chart"),
                            html.P("Type: White or Red"),
                            html.P("Values:"),
                            dcc.Dropdown(
                                id='values', 
                                value='Alcohol', 
                                options=[{'value': x, 'label': x} 
                                for x in ['fixed acidity', 'citric acid', 'chlorides', 'density']],
                                clearable=False
                            ),
                            dcc.Graph(id="pie-chart"),
                        ])])
                    ]),

                    dbc.Row([
                        dbc.Col([ 
                        html.H3("Scatter Plot"),

                        html.P("Select the first variable"),
                        html.Br([]),
                        dcc.Dropdown(
                            id = "cont-variable-1",
                            options=[{'label': i, 'value': i} for i in X.columns],
                            placeholder ="Select the first variable: ", 
                            value = "fixed acidity"
                        ),
                        html.Br([]),
                        html.P("Select the Second variable"),
                        dcc.Dropdown(
                            id = "cont-variable-2",
                            options = [{'label': i, 'value': i} for i in X.columns ],
                            placeholder ="Select the second variable: ", 
                            value = "citric acid"
                        ),
                        dcc.Graph(id="scatter-plot1"),
                        html.H3("quality"),
                        dcc.RangeSlider(
                            id='range-slider',
                            min=3, max=9, step=1,
                            marks={3: '3', 4: '4', 5: '5', 6: '6', 7:'7', 8:'8', 9: '9'},
                            value=[3, 9]
                        ),
                        ]),
                    ]),
                ]),
            ]),
            dcc.Tab(label = "Wine intercative plots", children = [ 

                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                        html.H1("Train Model:"),
                        dcc.Dropdown(
                            id='model-name',
                            options=[{'label': x, 'value': x} 
                                    for x in MODELS],
                            value='Logistic',
                            clearable=False
                        ),
                        dcc.Graph(id="graph1"),
                        ])

                    ]),
                ]),
            ]),
        ]),
]),
            
    
                            

    
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
    Output('table-sorting-filtering1', 'data'),
    Input('table-sorting-filtering1', "page_current"),
    Input('table-sorting-filtering1', "page_size"),
    Input('table-sorting-filtering1', 'sort_by'),
    Input('table-sorting-filtering1', 'filter_query'))
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
    Output("scatter-plot1", "figure"), 
    [Input("range-slider", "value"),
    Input('cont-variable-1', 'value'),
    Input('cont-variable-2', 'value')])
def update_bar_chart(slider_range, selected_var_1, selected_var_2):
    low, high = slider_range
    mask = (df['quality'] > low) & (df['quality'] < high)
    fig = px.scatter(
        df[mask], x=selected_var_1, y=selected_var_2,  
        color="type", 
        hover_data=['quality'])
    return fig


@app.callback(
    Output("graph1", "figure"), 
    [Input('model-name', "value")])
def train_and_display(name):
    model = MODELS[name]()
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate', 
            y='True Positive Rate'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    return fig

@app.callback(
    Output("box-plot1", "figure"), 
    [Input("y-axis", "value")])
def generate_chart(y):
    fig = px.box(df, x="type", y=y)
    return fig

@app.callback(
    Output('hist', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('contvariable', 'value')]
)
def update_chart(clicks, x):
    if clicks:
        return px.histogram(df,x=x)
    else:
        return px.histogram()
@app.callback(
    Output("pie-chart", "figure"),  
     Input("values", "value"))
def generate_chart(names, values):
    fig = px.pie(df, values=values, names="type")
    return fig


#@app.callback(
#    Output('hist', 'figure'),
 #   Input('cont-variable', 'value'))
#def update_hist(x):
 #   fig = px.histogram(df, x=x)
  #  return fig


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', debug=True)
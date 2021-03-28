import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
from app import app
import dash_html_components as html
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_table 
import plotly.graph_objects as go


df_url = 'https://raw.githubusercontent.com/leeyrees/datasets/main/winequalityN.csv'
df = pd.read_csv(df_url).dropna()
df['type'] = df['type'].astype('category')

PAGE_SIZE=6

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

layout = html.Div([dcc.Location(id="url"), sidebar, content])

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



@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return [
            html.H6(
                                        ["Data table"], className="subtitle padded"
                                    ),
                                    html.Br([]),
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
                                        sort_by=[]),
                                        dcc.Link('Go to App 1', href='/pages/backpack'),
        ]
    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


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


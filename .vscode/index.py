import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from pages import wine2, backpack


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/pages/backpack.py':
        return backpack.layout
    elif pathname == '/pages/wine2.py':
        return wine2.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)
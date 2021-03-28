import dash_html_components as html
import dash_bootstrap_components as dbc
import dash
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.LUX]
# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to our DASH project", className="text-center")
                    , className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='This app shows the statistical analysis of two datasets: a wine dataset and a backpack data set' 
            'Regarding the sources of these datasets, the Backpack one can be found in the R library Stat2Data. The wine dataset can be found in the following link:')),
                                           
            ]),
        html.Br([]),
        dbc.Row([
            dbc.Col(dbc.Button("Wine dataset", href="https://www.kaggle.com/rajyellow46/wine-quality",
                                                                   color="primary"),
                                                        className="mt-3") 
            
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='The app has a section for each dataset. In these sections, we can visualize the data through tables and graphs. Another feature of this app  '
                                     'is the possibility of analysing the results of predicting the data by using different ML methods and training and testing.')
                    , className="mb-5")
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H3(children='References',
                                               className="text-center"),
                                       dbc.Row([dbc.Col(dbc.Button("Dash Plotly", href="https://dash.plotly.com/",
                                                                   color="primary"),
                                                        className="mt-3"),
                                                dbc.Col(dbc.Button("Covid Dash App", href="https://github.com/meredithwan/covid-dash-app/blob/master/apps/home.py",
                                                                   color="primary"),
                                                        className="mt-3")], justify="center")
                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-5"),

    ]),

]),
])

# needed only if running this as a single page app
# if __name__ == '__main__':
#     app.run_server(host='127.0.0.1', debug=True)
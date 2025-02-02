
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, MATCH, ALL
from sqlalchemy import create_engine
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback

import base64
import datetime
import io

from layout.navbar import navbar
from layout.dashboard import dashboard
import pandas as pd
from helper.AI_Implementation import run_fire_occupied_model
from helper.AI_Implementation_v2 import run_severity_model
from helper.part_1 import run_current_data



app = Dash(
    __name__,
    title="Wildfire & Response Prevention Dashboard",
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200",  # Icons
        "https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&display=swap",  # Font
    ],
)
server = app.server

app.layout = html.Div(
    [
    
        
        navbar,
        
        dbc.Container(
            
            dbc.Stack(
                [
                    dcc.Markdown(
                        
                        link_target="_blank",
                        id="attribution",
                    ),
                    dashboard,
                ],
                gap=3,
            ),
            id="content",
            className="p-3",
        ),
        
       

    ],
    id="page",
    style= {
        "background-image": 'url("assets/fire_wallpaper.jpg")', 
        "background-size": "cover", 
        "background-position": "center", 
        "background-repeat": "no-repeat",
        "background-attachment": "fixed",  
        "overflow":"hidden"}
)


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([


        html.H4('Simple interactive table'),
        html.P(id='table_out'),
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} 
                    for i in df.columns],
            data=df.to_dict('records'),
            style_cell=dict(textAlign='left'),
            style_header=dict(backgroundColor="paleturquoise"),
            style_data=dict(backgroundColor="lavender")
        ), 


        html.Hr(),  
    ])

@callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == "__main__":
    run_fire_occupied_model()
    run_severity_model()
    run_current_data()
    if os.environ.get("environment") == "heroku":
        app.run(debug=False)
    else:
        app.run(debug=True)

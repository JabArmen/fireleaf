import dash_bootstrap_components as dbc
from dash import html




navbar = dbc.Row(
    [
        dbc.Col(
                html.Img(
                    src="assets/fireleaf logo.png",
                    alt="Source Code",
                    id="github-logo",
                    style={"height":"175px"}
                ),
           md=2 
        ),
        dbc.Col(
                html.H1("Wildfire Prevention & Prediction Program",style={"font-size": "50px", "color": "white"}),
            md=10,
            style={"align-content": "center"}
        ),
    ],
    id="navbar",
    style={"height":"100px", "font-size":"5rem","background-image":"assets/SAP-Logo.png",
       }

)

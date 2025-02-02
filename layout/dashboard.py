import json
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc
import datetime
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import folium
import helper.AI_Implementation_v2 as helper
import plotly.express as px
import numpy as np
    

SAVED_MODEL_PATH = 'data/rf_model.pkl'
model = joblib.load(SAVED_MODEL_PATH)
print("Loaded saved model from", SAVED_MODEL_PATH)
data = pd.read_csv('data/merged_data.csv')

features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'vegetation_index', 'human_activity_index']
target = 'fire_occurred'
X = data[features]
y = data[target].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
f1_scr = f1_score(y_test, y_pred)
confusion_columns = np.array([i for i in range(len(confusion))])
confusion = go.Figure(data=[go.Table(
  header = dict(
    values = [""]+ confusion_columns.tolist(),
    line_color='darkslategray',
    fill_color='rgba(65,105,225,0.25)',
    align='center',
    font=dict(color='white', size=12),
    font_size=46,
    height=160
  ),
  cells=dict(
    values=  np.insert(confusion, 0, confusion_columns, axis=0),
    line_color='darkslategray',
    fill=dict(color=['rgba(175, 238,238,0.3)', 'rgba(255,255,255,0.3)']),
    align='center',
    font_size=46,
    height=120,
    ),
  
    )
],
                      )

def generate_colors(df):
    colors = []
    for severity in df["severity"]:
        if severity == 3:
            colors.append("red")
        elif severity == 2:
            colors.append("orange")
        else:
            colors.append("yellow") 
    return colors
    
    
data = {
    'fire_start_date': [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 2),
        datetime.date(2024, 1, 3),
        datetime.date(2024, 1, 4),
        datetime.date(2024, 1, 5),
        datetime.date(2024, 1, 6),
        datetime.date(2024, 1, 7)
    ],
    'fire_count': [5, 8, 3, 10, 7, 6, 4]
}
fire_counts_over_time = pd.DataFrame(data)

part1_map_df = pd.read_csv('data/assigned_firefighting_units.csv') 

existing_fire_part1 = part1_map_df[part1_map_df["severity"] > 0]

fig = go.Figure(go.Scattermapbox(
    lat= existing_fire_part1["latitude"],
    lon= existing_fire_part1["longitude"],
    mode='markers',
    marker=dict(size=16, color=generate_colors(existing_fire_part1)),
    text= [f"Severity: {x}"for x in existing_fire_part1["severity"]],
))

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": 44, "lon": -73},
    mapbox_zoom=6
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



part2_map_df = pd.read_csv('data/new_future_fire_predictions_2025.csv')

existing_fire_part2 = part2_map_df[part2_map_df['severity'] > 0]

map2 = go.Figure(go.Scattermapbox(
    lat= existing_fire_part2["latitude"],
    lon= existing_fire_part2["longitude"],
    mode='markers',
    marker=dict(size=16, color=generate_colors(existing_fire_part2)),
    text= [f"Severity: {x}"for x in existing_fire_part2["severity"]],
))

map2.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": 44, "lon": -73},
    mapbox_zoom=6
)
map2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


part1_adressed = part1_map_df[part1_map_df["assigned_unit"].notna()]
part1_missed = part1_map_df[part1_map_df["assigned_unit"].isna()]

part1_total_count = part1_map_df['severity'].count()
part1_adressed_count = part1_map_df["assigned_unit"].notnull().sum()
part1_missed_count= part1_total_count - part1_adressed_count

part1_addressed_low_severity =  part1_adressed[(part1_map_df['severity'] == 1) ].count() ['severity'] 
part1_addressed_mid_severity =  part1_adressed[(part1_map_df['severity'] == 2) ].count()['severity'] 
part1_addressed_high_severity =  part1_adressed[(part1_map_df['severity'] == 3) ].count()['severity'] 


part1_missed_low_severity =  part1_missed[(part1_map_df['severity'] == 1) ].count()['severity'] 
part1_missed_mid_severity =  part1_missed[(part1_map_df['severity'] == 2) ].count()['severity'] 
part1_missed_high_severity =  part1_missed[(part1_map_df['severity'] == 3) ].count()['severity'] 

part1_damage_cost= (part1_missed_low_severity * 50000) + (part1_missed_mid_severity * 100000) + (part1_missed_high_severity * 200000)


sunburst_data = {
    "id": [
        "TotalFires",      
        "Adressed",        
        "Missed",          
        "High_A", "Medium_A", "Low_A",  
        "High_M", "Medium_M", "Low_M"   
    ],
    "parent": [
        "",           
        "TotalFires", 
        "TotalFires", 
        "Adressed", "Adressed", "Adressed",  
        "Missed", "Missed", "Missed"         
    ],
    "names": [
        "Total Fires",
        "Adressed",
        "Missed",
        "High <br> severity", "Medium <br> severity", "Low <br> severity",
        "High <br> severity", "Medium <br> severity", "Low <br> severity"
    ],
    "values": [
        part1_total_count,    
        part1_adressed_count,   
        part1_missed_count,    
        part1_addressed_high_severity , part1_addressed_mid_severity, part1_addressed_low_severity,   
        part1_missed_high_severity , part1_missed_mid_severity, part1_missed_low_severity
    ]
}

sunburst = px.sunburst(
    sunburst_data,
    ids="id",  
    names="names",  
    parents="parent",
    values="values",
)

sunburst.update_traces(
    insidetextorientation="radial",
    textfont_size=16,
    textinfo="label+value"         
)

sunburst.update_traces(
    hovertemplate="%{label}: %{value}<extra></extra>"
)

sunburst.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
sunburst.update_layout(
    autosize=False,
    )
sunburst.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

confusion.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor='rgba(0,0,0,0)'
)
sunburst.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

sunburst.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

resource_chart = go.Figure(go.Bar(
                                 x=[part1_adressed[(part1_map_df['assigned_unit'] == 'Ground Crews') ].count() ['assigned_unit'] , part1_adressed[(part1_map_df['assigned_unit'] == 'Smoke Jumpers') ].count() ['assigned_unit'], part1_adressed[(part1_map_df['assigned_unit'] == 'Fire Engines') ].count() ['assigned_unit'], part1_adressed[(part1_map_df['assigned_unit'] == 'Tanker Planes') ].count() ['assigned_unit'],part1_adressed[(part1_map_df['assigned_unit'] == 'Helicopters') ].count() ['assigned_unit']],
                                 y=['Ground Crews', 'Smoke Jumpers', 'Fire Engines', 'Tanker Planes','Helicopters'],
                                 orientation='h'))

resource_chart.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
severities_chart = go.Figure(go.Bar(
                                 x=[existing_fire_part2[existing_fire_part2['severity']==1].count()['severity'],existing_fire_part2[existing_fire_part2['severity']==2].count()['severity'], existing_fire_part2[existing_fire_part2['severity']==3].count()['severity']],
                                 y=['Low', 'Medium', 'High'],
                                 orientation='h'))

severities_chart.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

#DASHBOARD
dashboard = html.Div([
    html.H1("PART 1", style={"color": "white"}),
    dbc.Row(
    dbc.Col(
        [
            dbc.Row(
                [dbc.Col(
                        dcc.Graph(
                                id="part1-map",
                                figure=fig,
                                style={'height': '500px', "backgroundColor": "rgba(0,0,0,0)"}
    )),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(children=[html.H1(f"{(part1_adressed['cost'].sum()):,}$", style={'textAlign': 'center',
                       'color': '#dd1e35',} ),html.H3("Operational Cost",style={'textAlign': 'center',
                       'color': '#dd1e35',}),],body=True,className="mb-1", style = {"backgroundColor": "rgba(0.5,0.5,0.5,0.5)"})),
                    
                    dbc.Col(dbc.Card(children=[html.H1(f"{part1_damage_cost:,}$", style={'textAlign': 'center',
                       'color': '#eb8715',} ),html.H3("Damage Cost",style={'textAlign': 'center',
                       'color': '#eb8715',}),],body=True,className="mb-1",  style = {"backgroundColor": "rgba(0.5,0.5,0.5,0.5)"})),
                ],
                style={"margin-top": "30px"}
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                    dcc.Graph(
                                id="sunburst-part1",
                                figure=sunburst,
                                style={ "padding": 0, "backgroundColor": "rgba(0,0,0,0)" }),
                    md=6,  
                ),
                    dbc.Col(
                        dcc.Graph(
                            id = "part1-resource-chart",
                            
                            style = {"backgroundColor": "rgba(0,0,0,0)"},
                            figure = resource_chart
                        )
                    ),
                ],
                style = {"backgroundColor": "rgba(0,0,0,0)"}
            ),

        ],
    ),
    id="dashboard",
),
    html.Br(),
    html.H1("PART 2"),
    dbc.Row(
    dbc.Col(
        [
            dbc.Row(
                [dbc.Col(
                        dcc.Graph(
                                id="part2-map",
                                figure=map2,
                                style={'height': '500px', "backgroundColor": "rgba(0,0,0,0)"}
    )),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(children=[html.H1(f"{round(accuracy, 4)}%", style={'textAlign': 'center',
                       'color': '#dd1e35',} ),html.H3("Accuracy",style={'textAlign': 'center',
                       'color': '#dd1e35',}),],body=True,className="mb-1", style = {"backgroundColor": "rgba(0.5,0.5,0.5,0.5)"})),
                    
                    dbc.Col(dbc.Card(children=[html.H1(f"{round(f1_scr, 4)}", style={'textAlign': 'center',
                       'color': '#eb8715',} ),html.H3("F1 score",style={'textAlign': 'center',
                       'color': '#eb8715',}),],body=True,className="mb-1",  style = {"backgroundColor": "rgba(0.5,0.5,0.5,0.5)"})),
                ],
                style={"margin-top": "30px"}
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                    dcc.Graph(
                                id="part2-confusion",
                                figure=confusion,
                                style={ "padding": 0, "width": "100%", "height": "100%", "backgroundColor": "rgba(0,0,0,0)"}),
                    md=6,  
                ),
                    dbc.Col(
                        
                        dcc.Graph(
                            id = "part2-severities",
                            
                            figure = severities_chart
                        ),
                    ),
                ]
            ),

        ],
    ),),
    html.Br(),
    html.Div(id='output-data-upload'),
    dcc.Upload(
            id='upload-data',
            children=html.Button([
                'Upload File',
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=True
        ),
    ],
    style={   
    "background-color": "#ffffff75",
    "opacity": "6",
    "border-radius": "25px",
    "padding": "20px"
    
    }
)

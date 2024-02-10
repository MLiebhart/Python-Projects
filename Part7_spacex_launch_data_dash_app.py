# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)
# Set Dashboard background image
image_url = "https://cdn.dribbble.com/users/610788/screenshots/5157282/media/d4fac93d6c0c41afc08efb465d986094.png"
# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': 'white',
                                               'font-size': 40, 'background-image': f'url({image_url})', 
                                                'background-size': 'cover', 'background-repeat': 'no-repeat',
                                                'background-position': 'center', 'height': '100vh'}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                dcc.Dropdown(id='site-dropdown',  options=[
                                    {'label': 'All Sites', 'value': 'ALL'},
                                    {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                                    {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'},
                                    {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                                    {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                                    ],
                                    value='ALL',
                                    placeholder='Select a Launch Site',
                                    searchable=True
                                    ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(id='payload-slider',
                                    min=0, max=10000, step=1000,
                                    marks={0: '0', 5000: '5000',
                                            10000: '10000'},
                                    value=[min_payload, max_payload]
                                    ),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output

@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))
def get_pie_chart(entered_site):
    filtered_df = spacex_df
    if entered_site == 'ALL':
        fig = px.pie(spacex_df, values='class', 
        names='Launch Site', 
        title='Success Across All Launch Sites',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_layout(title_x=0.5, title_font_family='Bookman Old Style', font=dict(size=16, color='white'), paper_bgcolor='rgba(0,0,0,0.7)')
        return fig
    else:
        # return the outcomes piechart for a selected site
        filtered_df=spacex_df[spacex_df['Launch Site']== entered_site]
        filtered_df=filtered_df.groupby(['Launch Site', 'class']).size().reset_index(name='class count')
        fig=px.pie(filtered_df, values='class count', 
        names='class', 
        title=f'Launch Outcomes at site {entered_site}', 
        hole=0.3,
        color_discrete_sequence=['red', 'green']
        )
        fig.update_layout(title_x=0.5, title_font_family='Bookman Old Style', font=dict(size=16, color='white'), paper_bgcolor='rgba(0,0,0,0.7)')
        return fig

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='success-payload-scatter-chart',component_property='figure'),
                [Input(component_id='site-dropdown',component_property='value'),
                Input(component_id='payload-slider',component_property='value')])

def scatter(entered_site, payload):
    if entered_site=='ALL':
        fig=px.scatter(spacex_df, x='Payload Mass (kg)', y='class', 
            color='Booster Version Category', title='Success and Payload Mass', 
            color_discrete_sequence=px.colors.qualitative.Dark2
            )
        fig.update_layout(title_x=0.5, title_font_family='Bookman Old Style', font=dict(size=16, color='white'), paper_bgcolor='rgba(0,0,0,0.7)')
        return fig

    else:
        filtered_df = spacex_df[spacex_df['Payload Mass (kg)'].between(payload[0], payload[1])]
        fig=px.scatter(filtered_df, x='Payload Mass (kg)', y='class', 
            color='Booster Version Category', title=f'Success and Payload Mass at Site {entered_site}', 
            color_discrete_sequence=px.colors.qualitative.Dark2
            )
        fig.update_layout(title_x=0.5, title_font_family='Bookman Old Style', font=dict(size=16, color='white'), paper_bgcolor='rgba(0,0,0,0.7)')
        return fig

# Run the app
if __name__ == '__main__':
    app.run_server()

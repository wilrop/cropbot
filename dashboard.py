import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback_context

# Bulma stylesheet
external_stylesheets = [
    'https://cdn.jsdelivr.net/npm/bulma@0.9.2/css/bulma.min.css']

app = Dash(__name__, title="CropBot Dashboard",
           external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)


def make_world_state_figure(old_fig):
    try:
        world_state = pd.read_csv('world_view.csv')
        fig = px.scatter(world_state,
                         x="X",
                         y="Y",
                         color="Unvisited",
                         size='Risk',
                         hover_data={'X': False,
                                     'Y': False,
                                     'Unvisited': False,
                                     'Risk': False,
                                     'Value': True,
                                     'Robot': False},
                         color_continuous_scale=px.colors.sequential.Cividis_r,
                         text='Robot')
        fig.update_traces(textfont_size=40, textposition='middle center')
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                title=''
            ),
            yaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                title=''
            ),
            plot_bgcolor='rgba(185, 217, 155, 0.63)'
        )
        return fig
    except Exception:
        return old_fig


fig = make_world_state_figure(go.Figure())

app.layout = html.Div(children=[
    dcc.Markdown(children="CropBot Dashboard", className='title has-text-centered block is-vcentered'),
    html.Div([
        html.Div([
            html.Div([html.Div(id='weight1-output', className='title is-6 has-text-centered'),
                      dcc.Slider(0, 20, 0.01, value=10, id='weight1', marks={
                          0: '0',
                          5: '5',
                          10: '10',
                          15: '15',
                          20: '20'
                      }, className='')], className='column is-one-third'),
            html.Div([html.Div(id='weight2-output', className='title is-6 has-text-centered	'),
                      html.Div([dcc.Slider(0, 20, 0.01, value=10, id='weight2', marks={
                          0: '0',
                          5: '5',
                          10: '10',
                          15: '15',
                          20: '20'
                      })], className='')
                      ], className='column is-one-third'),
            html.Div([html.Div(id='weight3-output', className='title is-6 has-text-centered'),
                      dcc.Slider(0, 20, 0.01, value=10, id='weight3', marks={
                          0: '0',
                          5: '5',
                          10: '10',
                          15: '15',
                          20: '20'
                      }, className='')], className='column is-one-thirds')
        ], className='columns is-centered'),
    ], className='block'),
    dcc.Graph(
        id='world-view',
        figure=fig,
    ),
    html.Div([
        html.Div([
            html.Div([
                dcc.Markdown(children="Question 1: Do your crops require frequent interventions?",
                             className='title is-6'),
                dcc.RadioItems(['Yes', 'No', 'Uncertain'], 'Uncertain', labelStyle={'display': 'block'}, id='q1'),
            ], className='column is-one-fifth', style={'margin-left': '40px'}),
            html.Div([
                dcc.Markdown(children="Question 2: Which crop would you rather check?", className='title is-6'),
                dcc.RadioItems(['A closer crop that has not been checked in a while',
                                'A further crop which you expect might need an intervention'],
                               'A further crop which you expect might need an intervention',
                               labelStyle={'display': 'block'}, id='q2')
            ], className='column is-one-fifth'),
            html.Div([
                dcc.Markdown(children="Question 3: Would you like to turn on energy saving?", className='title is-6'),
                dcc.RadioItems(['Yes', 'No'], 'No', labelStyle={'display': 'block'}, id='q3')
            ], className='column is-one-fifth'),
            html.Div([
                dcc.Markdown(children="Question 4: Do you have a history of problems at specific locations?",
                             className='title is-6'),
                dcc.RadioItems(['Yes', 'No', 'Uncertain'], 'Uncertain', labelStyle={'display': 'block'}, id='q4')
            ], className='column is-one-fifth'),
            html.Div([
                dcc.Markdown(children="Question 5: How often do you monitor the field yourself?",
                             className='title is-6'),
                dcc.RadioItems(['Monthly', 'Weekly', 'Daily', 'Multiple times per day'], 'Weekly',
                               labelStyle={'display': 'block'},
                               id='q5')
            ], className='column is-one-fifth')
        ], className='columns is-centered'),
        html.Div([
            html.Button('Submit', id='submit-questions', n_clicks=0)
        ], className='has-text-centered'),
    ]),
    dcc.Interval(
        id='monitor-interval',
        interval=500,
        n_intervals=0
    )
])


def handle_preferences(value1, value2, value3, value4, value5):
    weight1 = 5
    weight2 = 5
    weight3 = 5

    if value1 == 'Yes':
        weight3 += 1
    elif value1 == 'No':
        weight3 -= 1

    if value2 == 'A closer crop that has not been checked in a while':
        weight1 += 1
        weight2 += 1
        weight3 += 1
    elif value2 == 'A further crop which you expect might need an intervention':
        weight1 += -1
        weight2 += -1
        weight3 += 2

    if value3 == 'Yes':
        weight2 += 1
    else:
        weight2 -= 1

    if value4 == 'Yes':
        weight3 += 1
    elif value4 == 'No':
        weight3 -= 1

    if value5 == 'Monthly':
        weight1 += 2
    elif value5 == 'Weekly':
        weight1 += 1
    elif value5 == 'Daily':
        weight1 -= 1
    else:
        weight1 -= 2

    weight1_str = f'Weight for interventions: {weight1}'
    weight2_str = f'Weight for interventions: {weight2}'
    weight3_str = f'Weight for interventions: {weight3}'

    return weight1_str, weight2_str, weight3_str, weight1, weight2, weight3


def handle_set_weight(weight1, weight2, weight3):
    weight1_str = f'Weight for interventions: {weight1}'
    weight2_str = f'Weight for interventions: {weight2}'
    weight3_str = f'Weight for interventions: {weight3}'
    return weight1_str, weight2_str, weight3_str, weight1, weight2, weight3


def write_weights(weight1, weight2, weight3):
    weight_dict = {'weight1': [weight1], 'weight2': [weight2], 'weight3': [weight3]}
    df = pd.DataFrame.from_dict(weight_dict)
    df.to_csv('weights.csv', index=False)


@app.callback(
    [Output('weight1-output', 'children'),
     Output('weight2-output', 'children'),
     Output('weight3-output', 'children'),
     Output('weight1', 'value'),
     Output('weight2', 'value'),
     Output('weight3', 'value')],
    [Input('weight1', 'value'),
     Input('weight2', 'value'),
     Input('weight3', 'value'),
     Input('submit-questions', 'n_clicks')],
    [State('q1', 'value'),
     State('q2', 'value'),
     State('q3', 'value'),
     State('q4', 'value'),
     State('q5', 'value')])
def update_weights(weight1, weight2, weight3, n_clicks, value1, value2, value3, value4, value5):
    write_weights(weight1, weight2, weight3)
    if callback_context.triggered:
        trigger = callback_context.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'submit-questions':
            return handle_preferences(value1, value2, value3, value4, value5)
        else:
            return handle_set_weight(weight1, weight2, weight3)
    else:
        return handle_set_weight(weight1, weight2, weight3)


@app.callback(
    Output('world-view', 'figure'),
    [Input('monitor-interval', 'n_intervals'),
     Input('world-view', 'figure')])
def update_worldview_figure(n_intervals, old_fig):
    return update_worldview_figure(old_fig)


if __name__ == '__main__':
    app.run_server(debug=True)

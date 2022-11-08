import json

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback_context

# Bulma stylesheet
external_stylesheets = ['https://cdn.jsdelivr.net/npm/bulma@0.9.2/css/bulma.min.css']

app = Dash(__name__, title='CropBot Dashboard',
           external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

world_state_file = 'world_state.json'
weights_file = 'weights.json'

start_weight = 10

frequency_str = 'Weight for frequency'
distance_str = 'Weight for distance'
intervention_str = 'Weight for interventions'

question1 = 'Question 1: Do your crops require frequent interventions?'
answers1 = ['Yes', 'No', 'Uncertain']
default_answer1 = answers1[2]

question2 = 'Question 2: Which crop would you rather check?'
answers2 = ['A closer crop that has not been checked in a while',
            'A further crop which you expect might need an intervention']
default_answer2 = answers2[0]

question3 = 'Question 3: Would you like to turn on energy saving?'
answers3 = ['Yes', 'No']
default_answer3 = answers3[1]

question4 = 'Question 4: Do you have a history of problems at specific locations?'
answers4 = ['Yes', 'No', 'Uncertain']
default_answer4 = answers4[2]

question5 = 'Question 5: How often do you monitor the field yourself?'
answers5 = ['Monthly', 'Weekly', 'Daily', 'Multiple times per day']
default_answer5 = answers5[1]

directions = {(0, 0): 'stay', (0, 1): 'forward', (0, -1): 'backward', (-1, 0): 'left', (1, 0): 'right'}


def make_world_state_figure(old_fig):
    """Make a new world state figure.

    Args:
        old_fig (Figure): The old figure in case a new figure can not be constructed.

    Returns:
        Figure: A new figure.
    """
    try:
        with open(world_state_file) as f:
            world_state = json.load(f)

        plan = world_state.pop('plan')
        robot_x, robot_y = world_state.pop('Robot')

        fig = px.scatter(world_state,
                         x='X',
                         y='Y',
                         color='Risk',
                         size='Unvisited',
                         hover_data={'X': False,
                                     'Y': False,
                                     'Unvisited': True,
                                     'Risk': True,
                                     'Value': True},
                         color_continuous_scale=px.colors.sequential.Cividis_r,
                         range_color=(0, 1))

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
            plot_bgcolor='rgba(185, 217, 155, 0.63)',
        )

        fig.add_annotation(x=robot_x,
                           y=robot_y,
                           text="🤖",
                           showarrow=False,
                           font={'size': 40})
        prev = (robot_x, robot_y)
        for idx, next in enumerate(plan):
            move_tpl = tuple(np.array(next) - np.array(prev))
            direction = directions[move_tpl]

            if direction == 'right':
                add_space_x = 0.005
                add_space_y = 0
            elif direction == 'right':
                add_space_x = -0.005
                add_space_y = 0
            elif direction == 'forward':
                add_space_x = 0
                add_space_y = 0.03
            else:
                add_space_x = 0
                add_space_y = -0.03

            if idx == len(plan) - 1:
                arrowhead = 2
                add_space_x = 0
                add_space_y = 0
            else:
                arrowhead = 0
            fig.add_annotation(x=next[0] + add_space_x,
                               y=next[1] + add_space_y,
                               xref="x",
                               yref="y",
                               text="",
                               showarrow=True,
                               axref="x",
                               ayref='y',
                               ax=prev[0],
                               ay=prev[1],
                               arrowhead=arrowhead,
                               arrowwidth=4,
                               arrowcolor='rgba(105, 90, 97, 0.6)')
            prev = next

        return fig
    except Exception as e:
        app.logger.info(str(e))
        return old_fig


fig = make_world_state_figure(go.Figure())

app.layout = html.Div(children=[
    dcc.Markdown(children='CropBot Dashboard', className='title has-text-centered block is-vcentered'),
    html.Div([
        html.Div([
            html.Div([html.Div(id='weight1-output', className='title is-6 has-text-centered'),
                      dcc.Slider(0, 20, 0.01, value=start_weight, id='weight1', marks={
                          0: '0',
                          5: '5',
                          10: '10',
                          15: '15',
                          20: '20'
                      }, className='')], className='column is-one-third'),
            html.Div([html.Div(id='weight2-output', className='title is-6 has-text-centered	'),
                      html.Div([dcc.Slider(0, 20, 0.01, value=start_weight, id='weight2', marks={
                          0: '0',
                          5: '5',
                          10: '10',
                          15: '15',
                          20: '20'
                      })], className='')
                      ], className='column is-one-third'),
            html.Div([html.Div(id='weight3-output', className='title is-6 has-text-centered'),
                      dcc.Slider(0, 20, 0.01, value=start_weight, id='weight3', marks={
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
                dcc.Markdown(children=question1,
                             className='title is-6'),
                dcc.RadioItems(answers1, default_answer1, labelStyle={'display': 'block'}, id='q1'),
            ], className='column is-one-fifth', style={'margin-left': '40px'}),
            html.Div([
                dcc.Markdown(children=question2, className='title is-6'),
                dcc.RadioItems(answers2, default_answer2, labelStyle={'display': 'block'}, id='q2')
            ], className='column is-one-fifth'),
            html.Div([
                dcc.Markdown(children=question3, className='title is-6'),
                dcc.RadioItems(answers3, default_answer3, labelStyle={'display': 'block'}, id='q3')
            ], className='column is-one-fifth'),
            html.Div([
                dcc.Markdown(children=question4, className='title is-6'),
                dcc.RadioItems(answers4, default_answer4, labelStyle={'display': 'block'}, id='q4')
            ], className='column is-one-fifth'),
            html.Div([
                dcc.Markdown(children=question5, className='title is-6'),
                dcc.RadioItems(answers5, default_answer5, labelStyle={'display': 'block'}, id='q5')
            ], className='column is-one-fifth')
        ], className='columns is-centered'),
        html.Div([
            html.Button('Submit', id='submit-questions', n_clicks=0)
        ], className='has-text-centered'),
    ]),
    dcc.Interval(
        id='monitor-interval',
        interval=1000,
        n_intervals=0
    )
])


def handle_preferences(value1, value2, value3, value4, value5):
    weight1 = 5
    weight2 = 5
    weight3 = 5

    if value1 == answers1[0]:
        weight3 += 1
    elif value1 == answers1[1]:
        weight3 -= 1

    if value2 == answers2[0]:
        weight1 += 1
        weight2 += 1
        weight3 += 1
    elif value2 == answers2[1]:
        weight1 += -1
        weight2 += -1
        weight3 += 2

    if value3 == answers3[0]:
        weight2 += 1
    else:
        weight2 -= 1

    if value4 == answers4[0]:
        weight3 += 1
    elif value4 == answers4[1]:
        weight3 -= 1

    if value5 == answers5[0]:
        weight1 += 2
    elif value5 == answers5[1]:
        weight1 += 1
    elif value5 == answers5[2]:
        weight1 -= 1
    else:
        weight1 -= 2

    weight1_str = f'{frequency_str}: {weight1}'
    weight2_str = f'{distance_str}: {weight2}'
    weight3_str = f'{intervention_str}: {weight3}'

    return weight1_str, weight2_str, weight3_str, weight1, weight2, weight3


def handle_set_weight(weight1, weight2, weight3):
    weight1_str = f'{frequency_str}: {weight1}'
    weight2_str = f'{distance_str}: {weight2}'
    weight3_str = f'{intervention_str}: {weight3}'
    return weight1_str, weight2_str, weight3_str, weight1, weight2, weight3


def write_weights(weight1, weight2, weight3):
    weight_dict = {'weights': [weight1, weight2, weight3]}
    with open(weights_file, 'w') as f:
        json.dump(weight_dict, f)


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
    return make_world_state_figure(old_fig)


if __name__ == '__main__':
    app.run_server(debug=True)

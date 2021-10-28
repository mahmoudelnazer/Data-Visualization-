import base64
from io import BytesIO
import time
from typing import Counter

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash.html import Title
from numpy import number
from numpy.lib.function_base import append
from plotly import plot
import plotly.graph_objects as go
import PIL
import requests
from numpy import argmin,argmax

from model import detect, filter_boxes, detr, transform
from model import CLASSES, DEVICE


# Dash component wrappers
def Row(children=None, **kwargs):
    return html.Div(children, className="row", **kwargs)


def Column(children=None, width=1, **kwargs):
    nb_map = {
        1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
        7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}

    return html.Div(children, className=f"{nb_map[width]} columns", **kwargs)


# plotly.py helper functions
def pil_to_b64(im, enc="png"):
    io_buf = BytesIO()
    im.save(io_buf, format=enc)
    encoded = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    return f"data:img/{enc};base64, " + encoded


def pil_to_fig(im, showlegend=False, title=None):
    img_width, img_height = im.size
    fig = go.Figure()
    # This trace is added to help the autoresize logic work.
    fig.add_trace(go.Scatter(
        x=[img_width * 0.05, img_width * 0.95],
        y=[img_height * 0.95, img_height * 0.05],
        showlegend=False, mode="markers", marker_opacity=0, 
        hoverinfo="none", legendgroup='Image'))

    fig.add_layout_image(dict(
        source=pil_to_b64(im), sizing="stretch", opacity=1, layer="below",
        x=0, y=0, xref="x", yref="y", sizex=img_width, sizey=img_height,))

    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        margin=go.layout.Margin(
        l=50, #left margin
        r=0,
        b=0, #right margin
        ))
    fig.update_layout(layout)

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(
        showgrid=False, visible=False, constrain="domain", range=[0, img_width])
    
    fig.update_yaxes(
        showgrid=False, visible=False,
        scaleanchor="x", scaleratio=1,
        range=[img_height, 0])
    
    fig.update_layout(showlegend=showlegend,title=title,title_font_color="#3499FF")

    return fig


def add_bbox(fig, x0, y0, x1, y1, 
             showlegend=True, name=None, color=None, 
             opacity=0.5, group=None, text=None):
    fig.add_trace(go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode="lines",
        fill="toself",
        opacity=opacity,
        marker_color=color,
        hoveron="fills",
        name=name,
        hoverlabel_namelength=0,
        text=text,
        legendgroup=group,
        showlegend=showlegend,
    ))


# colors for visualization
COLORS = ['#fe938c','#86e7b8','#f9ebe0','#208aae','#fe4a49', 
          '#291711', '#5f4b66', '#b98b82', '#87f5fb', '#63326e'] * 50

RANDOM_URLS = open('random_urls.txt').read().split('\n')[:-1]
print("Running on:", DEVICE)

# external JavaScript files
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
    {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
        'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
        'crossorigin': 'anonymous'
    }
]


# external CSS stylesheets
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

# Start Dash
app = dash.Dash(__name__,external_scripts=external_scripts,external_stylesheets=external_stylesheets)
server = app.server  # Expose the server variable for deployments

app.layout = html.Div(className='container', children=[
    Row(html.H1("Image Dataset Exploration Application",style={"color":"#3499FF"})),

    Row(html.H3("Image URL"),style={"color":"gray"}),
    Row([
        Column(width=8, children=[
            dcc.Input(id='input-url', style={'width': '100%'}, placeholder='Insert URL...'),
        ]),
        Column(html.Button("Load Image", id='button-run', n_clicks=0,style={"color":"#3499FF","font-size": "12px"}), width=2),
        Column(html.Button("Random Image", id='button-random', n_clicks=0,style={"color":"#3499FF","font-size": "12px"}), width=2)
    ]),
    Row([
        Column([
            Row(dcc.Graph(id='model-output', style={'width': '80vh'})),
            Row(dcc.Graph(id='number_graph',style={'width': '80vh'}))
        ],width=6),
        Column(dcc.Graph(id='cofidance_graph',style={'height': '125vh'}),width=6)
    ]),
    Row([
        Column(width=7, children=[
            html.H2('Non-maximum suppression',style={"color":"#3499FF"}),
            Row([
                Column(width=3, children=dcc.Checklist(
                    id='checklist-nms', 
                    options=[{'label': 'Enabled', 'value': 'enabled'}],
                    value=[],labelStyle={'display': 'inline',"padding-left":"15px","font-weight":"bold","font-size": "12px","color":"gray"}),style={"color":"gray"},
                    ),

                Column(width=9, children=dcc.Slider(
                    id='slider-iou', min=0, max=1, step=0.05, value=0.5, 
                    marks={0: '0', 1: '1'},tooltip={"placement": "bottom", "always_visible": True})),
            ])
        ]),
        Column(width=5, children=[
            html.H2('Confidence Threshold',style={"color":"#3499FF"}),
            dcc.Slider(
                id='slider-confidence', min=0, max=1, step=0.05, value=0.7, 
                marks={0: '0', 1: '1'},tooltip={"placement": "bottom", "always_visible": True})
        ])
    ])
])

def object_number_graph(lables_list):
    count_dict = Counter(lables_list)
    x_axis = list(count_dict.keys()) 
    y_axis = list(count_dict.values())

    colors= ["#4169E1"]*len(set(lables_list))
    colors[argmax(y_axis)] ="#E6E6FA"

    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        margin=go.layout.Margin(
        l=50, #left margin
        r=0, #right margin
        b=50, #bottom margin
        t=50  #top margin
        ))

    number_graph = go.Figure(data=[go.Bar(
            x=x_axis, y=y_axis,
            text=list(y_axis),
            textposition='auto',
            marker_color=colors,
        )])

    number_graph.update_yaxes(
        showgrid=False, visible=False,
        scaleanchor="x", scaleratio=1,)

    number_graph.update_layout(layout,title="Object Count",title_font_color="#3499FF")
    
    return number_graph

def confidance_graph(lables_list,confidance_list):
    plot_dict = {}

    colors= ["#4169E1"]*len(set(lables_list))

    for label,confidance in zip(lables_list,confidance_list):
        if label not in plot_dict.keys():
            plot_dict[label] = [confidance]
        else:
            plot_dict[label].append(confidance)

    for key in plot_dict.keys():
        plot_dict[key] = sum(plot_dict[key])/len(plot_dict[key])

    x_axis = list(plot_dict.values())
    y_axis = list(plot_dict.keys())

    colors[argmin(x_axis)] = "#E6E6FA"

    confidance_graph = go.Figure(data=[go.Bar(
            x=x_axis, y=y_axis,
            text=list(x_axis),orientation='h',
            textposition='auto',marker_color=colors
        )],layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)'))

    confidance_graph.update_xaxes(
        showgrid=False, visible=False,
        scaleanchor="y", scaleratio=1,)

    confidance_graph.update_layout(title="Avarage Accuracy",title_font_color="#3499FF")

    return confidance_graph

@app.callback(
    [Output('button-run', 'n_clicks'),
     Output('input-url', 'value')],
    [Input('button-random', 'n_clicks')],
    [State('button-run', 'n_clicks')])
def randomize(random_n_clicks, run_n_clicks):
    return run_n_clicks+1, RANDOM_URLS[random_n_clicks%len(RANDOM_URLS)]


@app.callback(
    [Output('model-output', 'figure'),
     Output('slider-iou', 'disabled'),
     Output('number_graph', 'figure'),
     Output('cofidance_graph', 'figure')],
    [Input('button-run', 'n_clicks'),
     Input('input-url', 'n_submit'),
     Input('slider-iou', 'value'),
     Input('slider-confidence', 'value'),
     Input('checklist-nms', 'value')],
    [State('input-url', 'value')])
def run_model(n_clicks, n_submit, iou, confidence, checklist, url):
    apply_nms = 'enabled' in checklist
    try:
        im = PIL.Image.open(requests.get(url, stream=True).raw)
    except:
        return go.Figure().update_layout(title='Incorrect URL')
    
    scores, boxes = detect(im, detr, transform, device=DEVICE)
    scores, boxes = filter_boxes(scores, boxes, confidence=confidence, iou=iou, apply_nms=apply_nms)

    scores = scores.data.numpy()
    boxes = boxes.data.numpy()

    fig = pil_to_fig(im, showlegend=True,title="Image With Ancohor Boxes")
    existing_classes = set()

    label_list = []
    confidence_list = []

    for i in range(boxes.shape[0]):
        class_id = scores[i].argmax()
        label = CLASSES[class_id]

        label_list.append(label)
    
        confidence = scores[i].max()
        x0, y0, x1, y1 = boxes[i]

        confidence_list.append(confidence)

        # only display legend when it's not in the existing classes
        showlegend = label not in existing_classes
        text = f"class={label}<br>confidence={confidence:.3f}"

        add_bbox(
            fig, x0, y0, x1, y1,
            opacity=0.7, group=label, name=label, color=COLORS[class_id], 
            showlegend=showlegend, text=text,
        )

        existing_classes.add(label)

    fig2 = object_number_graph(label_list)
    fig3 = confidance_graph(label_list,confidence_list)
    
    return fig, not apply_nms, fig2, fig3


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
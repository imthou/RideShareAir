# myapp.py

import numpy as np
import pandas as pd

from bokeh.models import Button
from bokeh.plotting import figure, curdoc, vplot
from bokeh.client import push_session

model1_file = 'data/model1_w_surgemulti_forecast.csv'
model2_file = 'data/model2_wo_surgemulti_forecast.csv'

def get_forecast_data(forecast_file):
    forecast = pd.read_csv(forecast_file, parse_dates=['record_time'])
    return forecast

def create_plots(model1, model2, city, display_name):
    if city != 'chicago':
        model1 = model1.query("city_{} == 1 and display_name_{} == 1".format(city, display_name))
        model2 = model2.query("city_{} == 1 and display_name_{} == 1".format(city, display_name))
    else:
        model1 = model1.query("city_denver == 0 and city_seattle == 0 and city_sf == 0 and city_ny == 0 and display_name_{} == 1".format(display_name))
        model2 = model2.query("city_denver == 0 and city_seattle == 0 and city_sf == 0 and city_ny == 0 and display_name_{} == 1".format(display_name))
    p = figure(title="Forecast of {} {} Prices - 4/11/16 to 4/17/16".format(city, display_name),
                    plot_width=1000, plot_height=500, x_axis_type="datetime")
    p.line(model1['record_time'], model1['y_forecast'], line_color='blue', line_width=2, legend="RF Model 1 - With Surge Multiplier", alpha=0.7)
    p.line(model2['record_time'], model2['y_forecast'], line_color='green', line_width=2, legend="RF Model 2 - Without Surge Multiplier", alpha=0.7, line_dash=[4,4])
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Average Price Estimate'

    # add a text renderer to out plot (no data yet)
    r = p.circle(x=[], y=[])
    ds = r.data_source
    return p, ds

# create a callback that will add a number in a random location
def callback():
    global i
    for ds in new_pts:
        ds.data['x'].append(model2['record_time'].iloc[i])
        ds.data['y'].append(model2['y_forecast'].iloc[i])
        ds.trigger('data', ds.data, ds.data)
    i = i + 1

# add a button widget and configure with the call back
# button = Button(label="Press Me")
# button.on_click(callback)

# put the button and plot in a layout and add to the document
# curdoc().add_root(vplot(button, p))

if __name__ == '__main__':
    # create a plot and style its properties
    model1 = get_forecast_data(model1_file)
    model2 = get_forecast_data(model2_file)
    i = 0

    plots = []
    new_pts = []
    for city in ['denver','ny','chicago','seattle','sf']:
        for cartype in ['uberX','uberXL','uberBLACK','uberSUV']:
            p, ds = create_plots(model1, model2, city, cartype)
            plots.append(p)
            new_pts.append(ds)
    ps = vplot(plots[0])


    session = push_session(curdoc())
    curdoc().add_periodic_callback(callback, 1000)
    session.show() # open the document in a browser
    session.loop_until_closed() # run forever

import numpy as np
from numpy import pi
import pandas as pd

from bokeh.client import push_session
from bokeh.driving import cosine
from bokeh.plotting import figure, curdoc

model1_file = 'data/model1_w_surgemulti_forecast.csv'
model2_file = 'data/model2_wo_surgemulti_forecast.csv'

def get_forecast_data(forecast_file):
    forecast = pd.read_csv(forecast_file, parse_dates=['record_time'])
    return forecast.query("city_denver == 1 and display_name_uberX == 1")[['record_time','y_forecast']]

@cosine(w=0.03)
def update(step):
    r2.data_source.data["y"] = y * step

if __name__ == '__main__':
    fig = figure(title="Forecast of Denver UberX Prices - 4/11/16 to 4/17/16",
                    plot_width=1000, plot_height=500, x_axis_type="datetime")
    model1 = get_forecast_data(model1_file)
    model2 = get_forecast_data(model2_file)
    fig.line(model1['record_time'], model1['y_forecast'], line_color='blue', line_width=2, legend="RF Model 1 - With Surge Multiplier", alpha=0.7)
    r2 = fig.line(model2['record_time'], model2['y_forecast'], line_color='green', line_width=2, legend="RF Model 2 - Without Surge Multiplier", alpha=0.7, line_dash=[4,4])
    fig.xaxis.axis_label = 'Time'
    fig.yaxis.axis_label = 'Average Price Estimate'

    x = np.linspace(0, 4*pi, 80)
    y = np.sin(x)

    # open a session to keep our local document in sync with server
    session = push_session(curdoc())

    curdoc().add_periodic_callback(update, 50)

    session.show() # open the document in a browser

    session.loop_until_closed() # run forever

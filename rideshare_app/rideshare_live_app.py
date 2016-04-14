from flask import request, render_template, Flask, session
from pymongo import MongoClient
import pandas as pd
import time
import os
from collections import OrderedDict

from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from bokeh.plotting import figure, curdoc, vplot, ColumnDataSource, show
from bokeh.client import push_session
from bokeh.models import HoverTool, PanTool, BoxZoomTool, ResizeTool, WheelZoomTool, PreviewSaveTool, ResetTool
from bokeh.models.widgets import Panel, Tabs

app = Flask(__name__)

ec2host = os.environ['EC2HOST']
client = MongoClient(host=ec2host,
                        port=27017)
db = client['apidata']
collection = db['uberapi']

model1_file = 'data/model1_w_surgemulti_forecast.csv'
model2_file = 'data/model2_wo_surgemulti_forecast.csv'

NEW_PTS = []
ALL_PLOTS = []

def get_forecast_data(forecast_file):
    """
    Loads the forecast data
    """
    forecast = pd.read_csv(forecast_file, parse_dates=['record_time'])
    forecast['date'] = forecast['record_time'].dt.date
    forecast['hour'] = forecast['record_time'].dt.hour
    name = forecast_file.split('/')[1].split('_')[0]
    forecast['name'] = name
    return forecast

def mongo_query():
    """
    Output: DataFrame

    Queries Mongo Database on EC2
    """
    start_date = (pd.to_datetime('2016-04-11') + pd.Timedelta(hours=7)).value // 10**9
    end_date = (pd.to_datetime('2016-04-17') + pd.Timedelta(hours=7)).value // 10**9
    docs = collection.find({'record_time':{'$gte':start_date,
                                    '$lte':end_date}},
                            {'record_time': 1, 'city':1, 'prices':1, '_id':0})
    data = []
    print "organizing data..."
    for doc in docs:
        df = pd.DataFrame(doc['prices'])
        df['avg_price_est'] = (df['low_estimate'] + df['high_estimate']) / 2.
        df['record_time'] = pd.to_datetime(doc['record_time'], unit='s') - pd.Timedelta(hours=7)
        df['city'] = doc['city']
        df['display_name'].replace(['UberBLACK','UberSUV','UberSELECT','uberT','Yellow WAV','ASSIST','PEDAL','For Hire','#UberTAHOE','uberCAB','WarmUpChi'],
                           ['uberBLACK','uberSUV','uberSELECT','uberTAXI','uberWAV','uberASSIST','uberPEDAL','uberTAXI','uberTAHOE','uberTAXI','uberWARMUP'], inplace=True)
        df['display_name'] = df['display_name'].apply(lambda x: x.lower())
        data.append(df[['record_time','city','display_name','avg_price_est']])
    print "finished collecting docs"
    df = pd.concat(data)
    df = df.set_index('record_time')
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    hourly = df.groupby(['date','hour','city','display_name']).mean().reset_index()
    hourly['record_time'] = pd.to_datetime(hourly['date'].astype(str) + ' ' + hourly['hour'].astype(str) + ":00:00")
    hourly['name'] = 'true values'
    return hourly

def create_plots(model1, model2, live_data, city, display_name):
    """
    Output: Bokeh plot

    Creates individual timeseries plot
    """
    if city != 'chicago':
        model1 = model1.query("city_{} == 1 and display_name_{} == 1".format(city, display_name))
        model2 = model2.query("city_{} == 1 and display_name_{} == 1".format(city, display_name))
    else:
        model1 = model1.query("city_denver == 0 and city_seattle == 0 and city_sf == 0 and city_ny == 0 and display_name_{} == 1".format(display_name))
        model2 = model2.query("city_denver == 0 and city_seattle == 0 and city_sf == 0 and city_ny == 0 and display_name_{} == 1".format(display_name))
    cartype = display_name.lower()
    live_data = live_data.query("display_name == @cartype and city == @city")

    source1 = ColumnDataSource(
        data=dict(
            d=model1['date'].astype(str),
            h=model1['hour'],
            f=model1['y_forecast'],
            n=model1['name']
        )
    )

    source2 = ColumnDataSource(
        data=dict(
            d=model2['date'].astype(str),
            h=model2['hour'],
            f=model2['y_forecast'],
            n=model2['name']
        )
    )

    source3 = ColumnDataSource(
        data=dict(
            d=live_data['date'].astype(str),
            h=live_data['hour'],
            f=live_data['avg_price_est'],
            n=live_data['name']
        )
    )

    hover = HoverTool(
        tooltips=[
            ("Model", "@n"),
            ("Date", "@d"),
            ("Hour", "@h"),
            ("Average Price", "@f")
        ]
    )

    p = figure(title="Forecast of {} {} Prices - 4/11/16 to 4/17/16".format(city, display_name),
                    plot_width=1000, plot_height=500, x_axis_type="datetime",
                    tools=[hover, PanTool(), BoxZoomTool(), ResizeTool(), WheelZoomTool(), PreviewSaveTool(), ResetTool()])

    p.line(model1['record_time'], model1['y_forecast'], line_color='blue', line_width=2, legend="RF Model 1 - With Surge Multiplier", alpha=0.5, source=source1)
    p.line(model2['record_time'], model2['y_forecast'], line_color='green', line_width=2, legend="RF Model 2 - Without Surge Multiplier", alpha=0.5, source=source2) # line_dash=[4,4]
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Average Price Estimate'
    p.xgrid[0].ticker.desired_num_ticks = 20

    # add a text renderer to out plot (no data yet)
    r = p.circle(x=live_data['record_time'], y=live_data['avg_price_est'], legend="True Average Prices", source=source3, color='red')
    ds = r.data_source
    return p, ds

def build_plot():
    """
    Builds bokeh plot for each city and cartype
    """
    global NEW_PTS
    global ALL_PLOTS

    model1 = get_forecast_data(model1_file)
    model2 = get_forecast_data(model2_file)
    live_data = mongo_query()
    print live_data.tail()

    tab = []
    for city in ['denver','ny','chicago','seattle','sf']:
        for cartype in ['uberX','uberXL','uberBLACK','uberSUV']:
            p, ds = create_plots(model1, model2, live_data, city, cartype)
            tab.append(Panel(child=p, title=cartype))
            NEW_PTS.append((city, cartype, ds))
        ALL_PLOTS.append(Tabs(tabs=tab))
        tab = []
    # ps = [vplot(plot) for plot in ALL_PLOTS]

def callback():
    """
    Makes a callback to EC2 Mongo Database every hour to update price data
    """
    print "collecting data again: {}", pd.to_datetime(time.time(), unit='s')
    live_data = mongo_query()
    print live_data.tail()
    for city, cartype, ds in NEW_PTS:
        cartype = cartype.lower()
        single = live_data.query("display_name == @cartype and city == @city")
        # print single.head()
        ds.data['x'] = single['record_time']
        ds.data['y'] = single['avg_price_est']
        ds.trigger('data', ds.data, ds.data)

@app.route("/")
@app.route("/home")
def index():
    return "Hello!"

@app.route("/callback")
def render_plot():
    """
    Renders plot and makes callback every hour
    """
    build_plot()
    # callback()
    session = push_session(curdoc())
    curdoc().add_periodic_callback(callback, 3600000)
    session.show() # open the document in a browser
    session.loop_until_closed() # run forever
    return ""

if __name__ == "__main__":
    app.run(debug=True)

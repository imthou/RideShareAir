# plot_predictions.py

import numpy as np
import pandas as pd

from bokeh.plotting import figure, vplot, output_server, show, push, cursession, ColumnDataSource
from bokeh.models import HoverTool, PanTool, BoxZoomTool, ResizeTool, WheelZoomTool, PreviewSaveTool, ResetTool
from bokeh.models.formatters import DatetimeTickFormatter, PrintfTickFormatter
from bokeh.models.widgets import Panel, Tabs
from bokeh.session import Session
from json import load
from urllib2 import urlopen
from bokeh.embed import autoload_server
from pymongo import MongoClient
import time
import os

ssn = Session(load_from_config=False)
output_server("predictions", session=ssn)

ec2host = os.environ['EC2HOST']
client = MongoClient(host=ec2host,
                        port=27017)
db = client['apidata']
collection = db['uberapi']

model1_file = 'data/model1_w_surgemulti_forecast.csv'
# model2_file = 'data/model2_wo_surgemulti_forecast.csv'
model3_file = 'data/ridgecv_uber_forecast.csv'
model4_file = 'data/uber_arima_forecast.csv'
model5_file = 'data/xgboost_model_forecast.csv'

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
    # monday to monday
    start_date = (pd.to_datetime('2016-05-09') + pd.Timedelta(hours=7)).value // 10**9
    end_date = (pd.to_datetime('2016-05-16') + pd.Timedelta(hours=7)).value // 10**9
    docs = collection.find({'record_time':{'$gte':start_date,
                                    '$lte':end_date}},
                            {'record_time': 1, 'city':1, 'prices':1, '_id':0})
    data = []
    print "organizing data..."
    length = docs.count()
    i = 0
    for doc in docs:
        df = pd.DataFrame(doc['prices'])
        df['avg_price_est'] = (df['low_estimate'] + df['high_estimate']) / 2.
        df['record_time'] = pd.to_datetime(doc['record_time'], unit='s') - pd.Timedelta(hours=7)
        df['city'] = doc['city']
        df['display_name'].replace(['UberBLACK','UberSUV','UberSELECT','uberT','Yellow WAV','ASSIST','PEDAL','For Hire','#UberTAHOE','uberCAB','WarmUpChi'],
                           ['uberBLACK','uberSUV','uberSELECT','uberTAXI','uberWAV','uberASSIST','uberPEDAL','uberTAXI','uberTAHOE','uberTAXI','uberWARMUP'], inplace=True)
        df['display_name'] = df['display_name'].apply(lambda x: x.lower())
        data.append(df[['record_time','city','display_name','avg_price_est']])
        if i % 50 == 0:
            print "progress: {}, {:.2f}%".format(i, float(i)/length * 100.0)
        i += 1
    print "finished collecting docs"
    df = pd.concat(data)
    df = df.set_index('record_time')
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    hourly = df.groupby(['date','hour','city','display_name']).mean().reset_index()
    hourly['record_time'] = pd.to_datetime(hourly['date'].astype(str) + ' ' + hourly['hour'].astype(str) + ":00:00")
    hourly['name'] = 'true values'
    forecast_name = "data/live_forecast_data.csv"
    hourly.to_csv(forecast_name,index=False)
    print "exported live data to {}".format(forecast_name)
    return hourly

# def create_plots(model1, model2, model3, model4, model5, live_data, city, display_name):
def create_plots(model1, model3, model4, model5, live_data, city, display_name):
    """
    Output: Bokeh plot

    Creates individual timeseries plot
    """
    if city != 'chicago':
        model1 = model1.query("city_{} == 1 and display_name_{} == 1".format(city, display_name))
        # model2 = model2.query("city_{} == 1 and display_name_{} == 1".format(city, display_name))
        model3 = model3.query("city_{} == 1 and display_name_{} == 1".format(city, display_name))
        model4 = model4.query("city_{} == 1 and display_name_{} == 1".format(city, display_name))
        model5 = model5.query("city_{} == 1 and display_name_{} == 1".format(city, display_name))
    else:
        model1 = model1.query("city_denver == 0 and city_seattle == 0 and city_sf == 0 and city_ny == 0 and display_name_{} == 1".format(display_name))
        # model2 = model2.query("city_denver == 0 and city_seattle == 0 and city_sf == 0 and city_ny == 0 and display_name_{} == 1".format(display_name))
        model3 = model3.query("city_denver == 0 and city_seattle == 0 and city_sf == 0 and city_ny == 0 and display_name_{} == 1".format(display_name))
        model4 = model4.query("city_denver == 0 and city_seattle == 0 and city_sf == 0 and city_ny == 0 and display_name_{} == 1".format(display_name))
        model5 = model5.query("city_denver == 0 and city_seattle == 0 and city_sf == 0 and city_ny == 0 and display_name_{} == 1".format(display_name))
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

    # source2 = ColumnDataSource(
    #     data=dict(
    #         d=model2['date'].astype(str),
    #         h=model2['hour'],
    #         f=model2['y_forecast'],
    #         n=model2['name']
    #     )
    # )

    source3 = ColumnDataSource(
        data=dict(
            d=model3['date'].astype(str),
            h=model3['hour'],
            f=model3['y_forecast'],
            n=model3['name']
        )
    )

    source4 = ColumnDataSource(
        data=dict(
            d=model4['date'].astype(str),
            h=model4['hour'],
            f=model4['y_forecast'],
            n=model4['name']
        )
    )

    source5 = ColumnDataSource(
        data=dict(
            d=model5['date'].astype(str),
            h=model5['hour'],
            f=model5['y_forecast'],
            n=model5['name']
        )
    )

    source6 = ColumnDataSource(
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
    change_city = {'denver':'Denver','ny':'New York','chicago':'Chicago','seattle':'Seattle','sf':'San Francisco'}
    p = figure(title="Forecast of {} {} Prices - 5/9/16 to 5/15/16".format(change_city[city], display_name),
                    plot_width=1000, plot_height=500, x_axis_type="datetime",
                    tools=[hover, PanTool(), BoxZoomTool(), ResizeTool(), WheelZoomTool(), PreviewSaveTool(), ResetTool()], toolbar_location="left", title_text_font_size="20pt")

    p.line(model1['record_time'], model1['y_forecast'], line_color='blue', line_width=2, legend="Random Forest Regressor", alpha=0.5, source=source1)
    # p.line(model2['record_time'], model2['y_forecast'], line_color='green', line_width=2, legend="RF Model 2 - Without Surge Multiplier", alpha=0.5, source=source2) # line_dash=[4,4]
    p.line(model3['record_time'], model3['y_forecast'], line_color='magenta', line_width=2, legend="Ridge Regression", alpha=0.5, source=source3) # line_dash=[4,4]
    p.line(model4['record_time'], model4['y_forecast'], line_color='gray', line_width=2, legend="ARIMA Model", alpha=0.5, source=source4) # line_dash=[4,4]
    p.line(model5['record_time'], model5['y_forecast'], line_color='green', line_width=2, legend="XGB Regressor", alpha=0.5, source=source5) # line_dash=[4,4]
    # p.xaxis.axis_label = 'Time'
    # p.xaxis.axis_label_text_font_size = "10pt"
    p.yaxis.axis_label = 'Average Price Estimate'
    p.yaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_standoff = 15
    p.xgrid[0].ticker.desired_num_ticks = 20
    xformatter = DatetimeTickFormatter(formats=dict(hours=["%H"]))
    p.xaxis.formatter = xformatter
    p.legend.label_text_font_size = "10pt"

    # add a text renderer to out plot (no data yet)
    r = p.circle(x=live_data['record_time'], y=live_data['avg_price_est'], legend="True Average Prices", source=source6, color='red')
    ds = r.data_source
    return p, ds

def build_plots():
    """
    Builds bokeh plot for each city and cartype
    """
    model1 = get_forecast_data(model1_file)
    # model2 = get_forecast_data(model2_file)
    model3 = get_forecast_data(model3_file)
    model4 = get_forecast_data(model4_file)
    model5 = get_forecast_data(model5_file)
    live_data = mongo_query()
    print live_data.tail()

    all_plots = []
    plots = []
    new_pts = []
    for city in ['denver','ny','chicago','seattle','sf']:
        for cartype in ['uberX','uberXL','uberBLACK','uberSUV']:
            # p, ds = create_plots(model1, model2, model3, model4, model5, live_data, city, cartype)
            p, ds = create_plots(model1, model3, model4, model5, live_data, city, cartype)
            tab = Panel(child=p, title=cartype)
            plots.append(tab)
            new_pts.append((city, cartype, ds))
        tabs = Tabs(tabs=plots)
        all_plots.append(tabs)
        plots = []
    return all_plots, new_pts

def callback(session, new_pts):
    """
    Makes a callback to EC2 Mongo Database every hour to update price data
    """
    print "collecting data again: {}", pd.to_datetime(time.time(), unit='s')
    live_data = mongo_query()
    print live_data.tail()
    for city, cartype, ds in new_pts:
        cartype = cartype.lower()
        single = live_data.query("display_name == @cartype and city == @city")
        # print single.head()
        ds.data['x'] = single['record_time']
        ds.data['y'] = single['avg_price_est']
        session.store_objects(ds)

if __name__ == '__main__':
    # create a plot and style its properties
    all_plots, new_pts = build_plots()
    # plots = np.array(all_plots).flatten()
    vp = vplot(*all_plots)
    ip = load(urlopen('http://jsonip.com'))['ip']
    session = cursession()
    session.publish()
    tag = autoload_server(vp, session, public=True).replace("localhost", ip).replace("localhost", ip)

    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    html = """
    {%% extends "base.html" %%}
    {%% block body %%}
    %s
    {%% endblock %%}
    """ % tag

    with open('templates/model.html', 'w+') as f:
        f.write(html)

    print "plots ready for deployment: {}".format(t)

    while True:
        time.sleep(3600)
        print "pulling data from Mongo EC2 again"
        callback(session, new_pts)

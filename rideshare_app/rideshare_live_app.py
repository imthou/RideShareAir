from flask import request, render_template, Flask, session, url_for
import pandas as pd

app = Flask(__name__)

model1_file = 'data/model1_w_surgemulti_forecast.csv'
lyft_file = 'data/lyft_rf_model_forecast.csv'

def get_forecast_data(forecast_file):
    """
    Loads the forecast data
    """
    forecast = pd.read_csv(forecast_file, parse_dates=['record_time'])
    forecast['date'] = forecast['record_time'].dt.date
    forecast['hour'] = forecast['record_time'].dt.hour
    forecast['dayofweek'] = forecast['record_time'].dt.dayofweek
    name = forecast_file.split('/')[1].split('_')[0]
    forecast['name'] = name
    return forecast

def get_price(city, cartype, dayofweek):
    """
    Input: City, Cartype, Dayofweek

    Output: Average Price for that day, Max Price, Min Price
    """
    print city, cartype, dayofweek
    change_day = {"monday":0,"tuesday":1,"wednesday":2,
    "thursday":3,"friday":4,"saturday":5,"sunday":6}
    if 'lyft' in cartype:
        forecast = get_forecast_data(lyft_file)
        dayofweek = change_day[dayofweek]
        if city != 'chicago':
            results = forecast.query("city_{} == 1 and ride_type_{} == 1 and dayofweek == {}".format(city.lower(), cartype, dayofweek))[['date','hour','y_forecast']]
        else:
            results = forecast.query("city_denver == 0 and city_sf == 0 and city_ny == 0 and city_seattle == 0 and ride_type_{} == 1 and dayofweek == {}".format(cartype, dayofweek))[['date','hour','y_forecast']]
            print results
    else:
        forecast = get_forecast_data(model1_file)
        dayofweek = change_day[dayofweek]
        if city != 'chicago':
            results = forecast.query("city_{} == 1 and display_name_{} == 1 and dayofweek == {}".format(city.lower(), cartype, dayofweek))[['date','hour','y_forecast']]
        else:
            results = forecast.query("city_denver == 0 and city_sf == 0 and city_ny == 0 and city_seattle == 0 and display_name_{} == 1 and dayofweek == {}".format(cartype, dayofweek))[['date','hour','y_forecast']]
    return results

@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")

@app.route("/forecast", methods=['GET','POST'])
def forecast():
    if request.method == "POST":
        city = (request.form['city'])
        cartype = (request.form['cartype'])
        dayofweek = (request.form['dayofweek'])
        results = get_price(city, cartype, dayofweek)
        results.rename(columns={"y_forecast":"price"}, inplace=True)
        results.set_index("date", inplace=True)
        results['price'] = results['price'].round(2)
        min_price = results['price'].min()
        max_price = results['price'].max()
        results_color = results.copy()
        results_color.reset_index(inplace=True)
        results_color.rename(columns=lambda x: x.capitalize(), inplace=True)
        style_results = results_color.style.highlight_max(axis=0, subset=['Price'], color='red').highlight_min(axis=0, subset=['Price'], color='green')
        # style_results.data.set_index("date", inplace=True)
        results_html = style_results.render()
        change_city = {'denver':'Denver','ny':'New York','chicago':'Chicago','seattle':'Seattle','sf':'San Francisco'}
        return render_template("results.html", results=results_html, cartype=cartype, city=change_city[city], min_price=min_price, max_price=max_price)
    return render_template("forecast.html")

@app.route("/model")
def model():
    return render_template("model.html")

@app.route("/lyft_model")
def lyft_model():
    return render_template("lyft_model.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

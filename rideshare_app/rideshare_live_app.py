from flask import request, render_template, Flask, session, url_for
import pandas as pd

app = Flask(__name__)

model1_file = 'data/model1_w_surgemulti_forecast.csv'

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
    change_day = {"monday":0,"tuesday":1,"wednesday":2,
    "thursday":3,"friday":4,"saturday":5,"sunday":6}
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
        return render_template("results.html", results=results.to_html(), cartype=cartype, city=city.capitalize())
    return render_template("forecast.html")

@app.route("/model")
def model():
    return render_template("model.html")

if __name__ == "__main__":
    app.run(debug=True)

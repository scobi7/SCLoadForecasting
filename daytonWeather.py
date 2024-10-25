import requests
import pandas as pd
import time

# OpenWeatherMap API key
API_KEY = '0db393daa611466cfcbb885cb06e8424'

# Coordinates for Dayton, Ohio
lat = 39.7589
lon = -84.1916

# Generate timestamps from 2015-01-01 to 2018-01-03 (Hourly)
start_date = "2015-01-01"
end_date = "2018-01-03"
timestamps = pd.date_range(start=start_date, end=end_date, freq='h')

# Convert to UNIX timestamp (int64)
timestamps = (timestamps.astype(int) // 10**9)

# Store weather data
weather_data = []

for ts in timestamps:
    url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={time}&appid={API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Extract relevant data (temperature, humidity, etc.)
        weather_entry = {
            'timestamp': ts,
            'temp': data['current']['temp'],
            'humidity': data['current']['humidity'],
            'wind_speed': data['current']['wind_speed'],
            'weather': data['current']['weather'][0]['description'],
        }
        weather_data.append(weather_entry)
    else:
        print(f"Error fetching data for {ts}")

    # Avoid hitting the API rate limit
    time.sleep(1)

# Convert to DataFrame and save to CSV
df_weather = pd.DataFrame(weather_data)
df_weather.to_csv('dayton_weather_2015_2018.csv', index=False)

print("Weather data saved to CSV!")

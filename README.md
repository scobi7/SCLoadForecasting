SCLoadForecasting
SCLoadForecasting is a project that uses machine learning to forecast power load based on hourly weather data for Dayton, Ohio. This project aims to help utilities and energy providers better manage energy demand by predicting load based on various weather factors.

Table of Contents
Overview
Project Structure
Installation
Usage
Data Sources
Contributing
Overview
This project predicts hourly power usage (load) based on weather conditions like temperature, precipitation, humidity, and wind speed. By analyzing historical data, we can forecast future energy needs, helping optimize grid management and reduce operational costs.

Key Features
Hourly power load forecasting based on weather data
Integration of temperature, humidity, wind speed, and precipitation as predictive features
CSV-based input and output for easy data handling and model evaluation
Project Structure
bash
Copy code
SCLoadForecasting/
├── data/                       # Folder for data files (ignored in Git)
├── LoadForecasting.py          # Main script for data processing and forecasting
├── combinedDaytonData_fill.csv # Combined dataset with filled weather data
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file to exclude unnecessary files

Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/SCLoadForecasting.git
cd SCLoadForecasting
Set Up a Virtual Environment (Optional but Recommended):

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Prepare Data: Ensure that the weather data is combined with load data in combinedDaytonData_fill.csv.
Run Forecasting Script:
To train the model and generate forecasts, run:
bash
Copy code
python LoadForecasting.py
View Results: The script will output forecast results, which can be analyzed or visualized.
Data Sources
The data used in this project includes:

Historical weather data (temperature, humidity, wind speed, and precipitation) for Dayton, Ohio.
Historical power load data for Dayton.
Note: The data/ folder is excluded from version control for privacy and storage management.

Load Forecasting project of 
find data, filter, and prep for feeding. 

Think of how I want to format the data.

CSV file combining weather and power usage.

Date, time, temperature, weather, power outages - > target data (power usage) 1000-10000 #TODO
 
testing remote connection

data is from dayton ohio, powered by AES

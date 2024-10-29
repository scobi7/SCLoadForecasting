# SCLoadForecasting

SCLoadForecasting is a machine learning project designed to forecast hourly power usage (load) based on weather data for Dayton, Ohio. This project provides a predictive model that can help utilities and energy providers manage demand and optimize grid operations by anticipating load based on weather conditions. I hope to integrate this model into the UCSC microgrid for accurate load forecasting.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Contributing](#contributing)

## Overview

SCLoadForecasting analyzes historical weather data—temperature, humidity, wind speed, and precipitation—to predict power load on an hourly basis. This information is valuable for energy providers to balance supply with demand, reduce costs, and manage resources more efficiently.

### Key Features
- **Hourly Load Forecasting**: Predicts energy usage for each hour based on past weather data through an SNN model.
- **Comprehensive Weather Integration**: Uses weather variables like temperature, precipitation, humidity, and wind speed as predictors.
- **Data**: The data is based on Dayton, Ohio's AES powerplant. The date range is from 12/31/2015 to 1/02/2028 and is hourly

## Project Steps

- **Data Loading**: Scrape data from AES's power usage in Dayton Ohio. Then scrape the weather of Dayton area. Combine into one CSV file
- **Data Format**: datetime,temperature,precipitation,humidity,wind_speed,dayton_mw
- **Define a build the model**: Define an effecient SNN model for time series load forecasting. RNNs and LSTMs may be a good option.
- **Train your data**: Create a training and testing data set. 80% of the data can be training where you run and train your model. Then you can test your results against the actual results from the testing data set. 







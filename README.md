# kerasTuner-MinuteStockData
Hyperparameter tuner using the keras-tuner library in relation to stock minute data

## Latest version/ Added features

- Version 1.0.0
  - Works similar to kerasTuner-DayStockData but instead of day intervals its minutes. Main difference relates to the data pre-processing.

## Key Information

### Libraries

- tensorflow-[keras](https://keras.io/) used for machine learning
- [keras-tuner](https://keras.io/keras_tuner/) used for finding best hyperparameters
- [yfinance]([https://pandas-datareader.readthedocs.io/en/latest/#](https://pypi.org/project/yfinance/)) used to read from Yahoo Fnance
- [matplotlib](https://matplotlib.org/) used to graph data
- [scikit-learn](https://scikit-learn.org/stable/) used to scale data

### Historical Data

- Historical data is retrived from [Yahoo Finance](https://uk.finance.yahoo.com/)

### Key Global Variables

Description of the parameters needed to be inputed are present in both main code and the body code.
- pastDays: if timeIntervals is set to 1m then 7d is the max that can be used.

### Whilst running

- Code may take awhile to run depending on the max hyperprameters that are given. Changing the verbose in the createModel function (line 121) to 1-2 will output a graphic that will show if the code is still running.

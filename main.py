from hpTunerMSE_minute import stockHpTuner

"""Main script to show how to utilise the hpTunerMSE_minute script"""

hpTuner = stockHpTuner(
    # (Mandatory parameter) ticker needs to be specified to determine the stock data that will be tuned.
    ticker = "AMD",
    # (optional) features that can be used ["High", "Low", "Open", "Close", "Adj Close", "Volume"]. Default value removes high and low and close.
    features = ["Open", "Adj Close", "Volume"],
    #(optional but change recomended. read the README file for more info) warning increasing how far back to take data will increase time taken to run.
    #valiid intervals: 1d, 5d, 7d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    pastDays = "7d",
    #(optional. read the README file for more info)
    #valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    timeInterval = "1m",
    # (optional) shorter amount of prediction days used will allow higher accuracy when predicting close future minutes. Defualt value is 3
    predictionTimes = 3,
    # (optional) value closer to one will give more data to the train set and less to the test set.
    percentageSplit = 0.98,
    # (optional but change recomended depending on input data) warning increasing amount will increase time taken to run. 
    # Recomended values 0 - 5. Default value given is 3.
    maxLayers = 3,
    # (optional but change recomended depending on input data) warning increasing amount will increase time taken to run.
    # Recomended values (length of the output values) - (length of the input values). Default value given is 512.
    maxUnits = 512,
    # (optional but change recomended depending on input data) warning decreasing amount will increase time taken to run.
    # Recomended values depends on the maxUnits used. Default value given is 32.
    unitDecriment = 32,
    # (optional but change recomended depending on input data) warning increasing amount will increase time taken to run.
    # Recomended values 0.1 - 0.8. Default value given is 0.5.
    maxDropout = 0.5,
    # (optional but change recomended depending on input data) warning decreasing amount will increase time taken to run.
    # Recomended values depends on the maxDropout used. Default value given is 0.1.
    dropoutDecriment = 0.1,
    # (optional) Recomended values 1 - (length of the input values). Default value 128.
    batchSize = 128,
    # (optional) warning increasing amount will increase time taken to run. Default value 32.
    searchMaxEpochs = 32,
    # (optional) warning increasing amount will increase time taken to run. Default value 5.
    errorCheckLoopAmount = 5,
    # (optional) warning increasing amount will increase time taken to run. Default value 32.
    predictEpochs = 32)

#Fetches and formats train data then sets objects parameters to that data.
trainX, trainY, testX, actualPrices, scaler = hpTuner.getData()
hpTuner.trainX = trainX
hpTuner.trainY = trainY
#Tuning occures to obtain best hyperparameters using the given data given.
bestHp, tuner = hpTuner.hyperPerameterTweaking(trainX, trainY, testX, actualPrices)
bestModel = tuner.hypermodel.build(bestHp)
#Fits the a model and stores MSE history to then check the epoch that has the lowest MSE.
bestEpochTest = bestModel.fit(trainX, trainY, epochs = hpTuner.searchMaxEpochs)
history = bestEpochTest.history["mse"]
bestEpoch = history.index(min(history)) + 1
#Refits the model using the best epoch loop found previously and uses this model and test data to make predictions.
bestModel.fit(trainX, trainY, epochs = bestEpoch)
predictedPrices = bestModel.predict(testX)
predictedPrices = scaler.inverse_transform(predictedPrices)
#Uses the predicted and actual data to map the feature in the first index as a visual representation of acuracy.
hpTuner.modelAccuracyMapping(actualPrices, predictedPrices)
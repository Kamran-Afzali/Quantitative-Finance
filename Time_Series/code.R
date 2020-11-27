#https://github.com/andrew-couch/Tidy-Tuesday/blob/master/TidyTuesdayForecasting.Rmd


library(PerformanceAnalytics)
library(quantmod)
library(tidyverse)
library(modeldata)
library(forecast)


BTC <- getSymbols("BTC-USD", src = "yahoo", from = "2013-01-01", to = "2020-11-01", auto.assign = FALSE)

Op(BTC)
Hi(BTC)
Lo(BTC)
Cl(BTC)
Vo(BTC)
Ad(BTC)



plot(dailyReturn(BTC))

plot(weeklyReturn(BTC))

ts=ts(weeklyReturn(BTC))

auto.arima(ts)

logts=log10(ts)
auto.arima(logts)


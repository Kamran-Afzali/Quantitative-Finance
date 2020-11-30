#https://github.com/andrew-couch/Tidy-Tuesday/blob/master/TidyTuesdayForecasting.Rmd
#https://github.com/andrew-couch/Tidy-Tuesday/blob/master/TidyTuesdayTidyForecast.Rmd


library(PerformanceAnalytics)
library(quantmod)
library(tidyverse)
library(modeldata)
library(forecast)
library(tidymodels)
library(modeltime)
library(timetk)
library(lubridate)


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


train_data <- training(initial_time_split(ts, prop = .8))
test_data <- testing(initial_time_split(ts, prop = .8))



arima_model <- arima_reg() %>% 
  set_engine("auto_arima") %>% 
  fit(daily_change~date, data = train_data)
prophet_model <- prophet_reg() %>% 
  set_engine("prophet") %>% 
  fit(daily_change~date, data = train_data)
tslm_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  fit(daily_change~as.numeric(date) + factor(month(date, label = TRUE)), data = train_data)
arima_boosted_model <- arima_boost(learn_rate = .015, min_n = 2) %>% 
  set_engine("auto_arima_xgboost") %>% 
  fit(daily_change~date + as.numeric(date) + factor(month(date, label = TRUE)), data = train_data)
forecast_table <- modeltime_table(
  arima_model,
  prophet_model,
  tslm_model,
  arima_boosted_model
)

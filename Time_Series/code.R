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

ts=as.data.frame(weeklyReturn(BTC))
ts$date=as.Date (row.names(ts)) 

train_data <- training(initial_time_split(ts, prop = .8))
test_data <- testing(initial_time_split(ts, prop = .8))

train_data %>% mutate(type = "train") %>% 
  bind_rows(test_data %>% mutate(type = "test")) %>% 
  ggplot(aes(x = date, y =weekly.returns, color = type)) + 
  geom_line()

arima_model <- arima_reg() %>% 
  set_engine("auto_arima") %>% 
  fit(weekly.returns~date, data = train_data)


prophet_model <- prophet_reg() %>% 
  set_engine("prophet") %>% 
  fit(weekly.returns~date, data = train_data)


tslm_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  fit(weekly.returns~as.numeric(date) + factor(month(date, label = TRUE)), data = train_data)


arima_boosted_model <- arima_boost(learn_rate = .015, min_n = 2) %>% 
  set_engine("auto_arima_xgboost") %>% 
  fit(weekly.returns~date + as.numeric(date) + factor(month(date, label = TRUE)), data = train_data)


forecast_table <- modeltime_table(
  arima_model,
  prophet_model,
  tslm_model,
  arima_boosted_model
)



forecast_table <- modeltime_table(
  arima_model,
  prophet_model,
  tslm_model,
  arima_boosted_model
)

forecast_table %>% 
  modeltime_calibrate(test_data) %>% 
  modeltime_accuracy()


forecast_table %>% 
  modeltime_calibrate(test_data) %>% 
  modeltime_forecast(actual_data = test_data) %>% 
  plot_modeltime_forecast()


forecast_table %>% 
  modeltime_refit(df) %>% 
  modeltime_forecast(h = 7, actual_data = df) %>% 
  plot_modeltime_forecast()

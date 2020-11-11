rm(list=ls())
install.packages("quantmod")
install.packages("ggplot2")
install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)
library(quantmod)
library(ggplot2)
library(tidyverse)
#https://bookdown.org/kochiuyu/Technical-Analysis-with-R/charting-with-indicators.html
#https://lamfo-unb.github.io/2017/07/22/intro-stock-analysis-1/
spy <- getSymbols("SPY", src = "yahoo", from = "2013-01-01", to = "2017-06-01", auto.assign = FALSE)



spy%>Op()
spy%>Hi()
spy%>Lo()
spy%>Cl()
spy%>Vo()
spy%>Ad()

chartSeries(spy,
            type="line",
            theme=chartTheme('white'))

chartSeries(spy,
            type="bar",
            theme=chartTheme('white'))
SMA(Ad(spy),n=20)
EMA(Ad(spy),n=20)
BBands(Ad(spy),s.d=2)
MACD(Ad(spy),12,26,9,EMA)
RSI(Ad(spy),n=5)

chartSeries(spy,
            subset='2013::2016',
            theme=chartTheme('white'))
addMACD(fast=12,slow=26,signal=9,type="EMA")

dailyReturn(spy)
plot(dailyReturn(spy))


weeklyReturn(spy)
plot(weeklyReturn(spy))


monthlyReturn(spy)
plot(monthlyReturn(spy))


quarterlyReturn(spy)
plot(quarterlyReturn(spy))


yearlyReturn(spy)
plot(yearlyReturn(spy))


signal1=1*(RSI(Ad(spy))< 30)
trade1 <- Lag(signal1)
trade1[is.na(trade1)]=0
ret1 <- dailyReturn(spy)*trade1
sum(ret1)
charts.PerformanceSummary(ret1)


signal2=1*(EMA(Ad(spy),n=10)>EMA(Ad(spy),n=50))
trade2 <- Lag(signal2)
trade2[is.na(trade2)]=0
ret2 <- dailyReturn(spy)*trade2
sum(ret2)
charts.PerformanceSummary(ret2)

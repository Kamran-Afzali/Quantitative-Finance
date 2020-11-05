rm(list=ls())
install.packages("quantmod")
install.packages("ggplot2")
install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)
library(quantmod)
library(ggplot2)
library(tidyverse)
#https://bookdown.org/kochiuyu/Technical-Analysis-with-R/charting-with-indicators.html
spy <- getSymbols("SPY", src = "yahoo", from = "2013-01-01", to = "2017-06-01", auto.assign = FALSE)

ggplot(spy, aes(x = index(spy), y = spy[,6])) + 
  geom_line(color = "darkblue") +
  ggtitle("Spy prices series") + xlab("Date") + 
  ylab("Price") + theme(plot.title = element_text(hjust = 0.5)) + 
  scale_x_date(date_labels = "%b %y", date_breaks = "6 months")

spy%>%Ad()

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



day <-14
price <- Cl(spy)
signal <- c()                    #initialize vector
rsi <- RSI(price, day)     #rsi is the lag of RSI
signal [1:day+1] <- 0            #0 because no signal until day+1

for (i in (day+1): length(price)){
  if (rsi[i] < 30){             #buy if rsi < 30
    signal[i] <- 1
  }else {                       #no trade all if rsi > 30
    signal[i] <- 0
  }
}
signal<-reclass(signal,Cl(spy))
trade1 <- Lag(signal)

#construct a new variable ret1
ret1 <- dailyReturn(spy)*trade1
names(ret1) <- 'Naive'
charts.PerformanceSummary(ret1)


charts.PerformanceSummary(ret1, main="Naive v.s. RSI")

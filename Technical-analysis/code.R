rm(list=ls())
install.packages("quantmod")
install.packages("ggplot2")
library(quantmod)
library(ggplot2)
#https://bookdown.org/kochiuyu/Technical-Analysis-with-R/charting-with-indicators.html
spy <- getSymbols("SPY", src = "yahoo", from = "2013-01-01", to = "2017-06-01", auto.assign = FALSE)

ggplot(spy, aes(x = index(spy), y = spy[,6])) + 
  geom_line(color = "darkblue") +
  ggtitle("Spy prices series") + xlab("Date") + 
  ylab("Price") + theme(plot.title = element_text(hjust = 0.5)) + 
  scale_x_date(date_labels = "%b %y", date_breaks = "6 months")

rm(list=ls())

install.packages("quantmod")
install.packages("PerformanceAnalytics")
install.packages("modeldata")
install.packages("forecast")

library(PerformanceAnalytics)
library(quantmod)
library(tidyverse)
library(modeldata)
library(forecast)

#https://bookdown.org/kochiuyu/Technical-Analysis-with-R/charting-with-indicators.html
#https://lamfo-unb.github.io/2017/07/22/intro-stock-analysis-1/
#https://github.com/andrew-couch/Tidy-Tuesday/blob/master/TidyTuesdayForecasting.Rmd

spy <- getSymbols("SPY", src = "yahoo", from = Sys.Date()-365, to = Sys.Date(), auto.assign = FALSE)

Sys.Date()-365

head(spy$SPY.Close)
head(Lag(spy$SPY.Close))
chartSeries(spy,
            type="line",
            theme=chartTheme('white'))

chartSeries(spy,
            type="bar",
            theme=chartTheme('white'))




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

SMA(Cl(spy),n=20)
EMA(Cl(spy),n=20)
chartSeries(spy,
            subset='2013::2016',
            theme=chartTheme('white'))
addMACD(fast=12,slow=26,signal=9,type="EMA")


signal1=1*(RSI(Cl(spy))< 30)
trade1 <- Lag(signal1)
trade1[is.na(trade1)]=0
ret1 <- dailyReturn(spy)*trade1
sum(ret1)
charts.PerformanceSummary(ret1)

RSI(Cl(spy),n=5)

signal2=1*(EMA(Cl(spy),n=12)>EMA(Cl(spy),n=26))
trade2 <- Lag(signal2)
trade2[is.na(trade2)]=0
ret2 <- dailyReturn(spy)*trade2
sum(ret2)
charts.PerformanceSummary(ret2)

signal2=1*(EMA(Cl(spy),n=24)>EMA(Cl(spy),n=52))
trade2 <- Lag(signal2)
trade2[is.na(trade2)]=0
ret2 <- dailyReturn(spy)*trade2
sum(ret2)
charts.PerformanceSummary(ret2)

signal3=1*(MACD(Cl(spy), nSig = 18)$signal>0)
trade3 <- Lag(signal3)
trade3[is.na(trade3)]=0
ret3 <- dailyReturn(spy)*trade3
sum(ret3)
charts.PerformanceSummary(ret3)

summary(MACD(Cl(spy), nSig = 9)$signal)
###############################################################################################
spy <- getSymbols("IHI", src = "yahoo", from = "2012-07-19", to = "2019-07-19", auto.assign = FALSE)
buy_trd_crt=(1*(RSI(Cl(spy))< 30)%>%Lag()%>% replace(is.na(.), 0)+
    1*(EMA(Cl(spy),n=12)>EMA(Cl(spy),n=26))%>%Lag()%>% replace(is.na(.), 0)+
    1*(EMA(Cl(spy),n=24)>EMA(Cl(spy),n=52))%>%Lag()%>% replace(is.na(.), 0)+
    1*(MACD(Cl(spy))$signal>0.4)%>%Lag()%>% replace(is.na(.), 0)+
    1*(MACD(Cl(spy), nSig = 18)$signal>0.4)%>%Lag()%>% replace(is.na(.), 0))
table(buy_trd_crt)
sell_trd_crt=( 1*(RSI(Cl(spy))> 70)%>%Lag()%>% replace(is.na(.), 0)+
               1*(EMA(Cl(spy),n=12)<EMA(Cl(spy),n=26))%>%Lag()%>% replace(is.na(.), 0)+
               1*(EMA(Cl(spy),n=24)<EMA(Cl(spy),n=52))%>%Lag()%>% replace(is.na(.), 0)+
               1*(MACD(Cl(spy))$signal<0)%>%Lag()%>% replace(is.na(.), 0)+
               1*(MACD(Cl(spy), nSig = 18)$signal<0)%>%Lag()%>% replace(is.na(.), 0))
table(sell_trd_crt)


sum(dailyReturn(spy))
length(dailyReturn(spy))

ret <- dailyReturn(spy)*(1*(buy_trd_crt>0))+ dailyReturn(spy)*(1*!(sell_trd_crt>3))
sum(ret)

ret <- dailyReturn(spy)*(1*(buy_trd_crt>0))
sum(ret)

ret <-  dailyReturn(spy)*(1*!(sell_trd_crt>3))
sum(ret)


sum((trd_crt>0))


sum(!(dailyReturn(spy)*(1*(buy_trd_crt>0))+ dailyReturn(spy)*(1*!(sell_trd_crt>3))==0))
##############################################################################################


sigtab<- function(tkr){
  spy <- getSymbols(tkr, src = "yahoo", from = Sys.Date()-100, to = Sys.Date(), auto.assign = FALSE)
  signal_RSI_B  =1*(RSI(Cl(spy))< 30)%>%Lag()%>% replace(is.na(.), 0)
  signal_EMA1_B =1*(EMA(Cl(spy),n=12)>EMA(Cl(spy),n=26))%>%Lag()%>% replace(is.na(.), 0)
  signal_EMA2_B =1*(EMA(Cl(spy),n=24)>EMA(Cl(spy),n=52))%>%Lag()%>% replace(is.na(.), 0)
  signal_MACD1_B=1*(MACD(Cl(spy))$signal>0.3)%>%Lag()%>% replace(is.na(.), 0)
  signal_MACD2_B=1*(MACD(Cl(spy), nSig = 18)$signal>0.3)%>%Lag()%>% replace(is.na(.), 0)
  signal_B      =signal_RSI_B+signal_EMA1_B+signal_EMA2_B+signal_MACD1_B+signal_MACD2_B
  signal_RSI_S  =1*(RSI(Cl(spy))> 70)%>%Lag()%>% replace(is.na(.), 0)
  signal_EMA1_S =1*(EMA(Cl(spy),n=12)<EMA(Cl(spy),n=26))%>%Lag()%>% replace(is.na(.), 0)
  signal_EMA2_S =1*(EMA(Cl(spy),n=24)<EMA(Cl(spy),n=52))%>%Lag()%>% replace(is.na(.), 0)
  signal_MACD1_S=1*(MACD(Cl(spy))$signal<0)%>%Lag()%>% replace(is.na(.), 0)
  signal_MACD2_S=1*(MACD(Cl(spy), nSig = 18)$signal<0)%>%Lag()%>% replace(is.na(.), 0)
  signal_S      =signal_RSI_S+signal_EMA1_S+signal_EMA2_S+signal_MACD1_S+signal_MACD2_S
  dailyreturn   =(dailyReturn(spy)%>%Lag()%>% replace(is.na(.), 0))*100
  weeklyreturn  =(SMA(dailyReturn(spy),n=5)%>%Lag()%>% replace(is.na(.), 0))*100
  
  
  Sig_Tab=cbind(signal_RSI_B  
                ,signal_EMA1_B 
                ,signal_EMA2_B 
                ,signal_MACD1_B
                ,signal_MACD2_B
                ,signal_B      
                ,signal_RSI_S  
                ,signal_EMA1_S 
                ,signal_EMA2_S 
                ,signal_MACD1_S
                ,signal_MACD2_S
                ,signal_S      
                ,dailyreturn   
                ,weeklyreturn)
  
  colnames(Sig_Tab)=cbind( "signal_RSI_B"
                           ,"signal_EMA1_B" 
                           ,"signal_EMA2_B" 
                           ,"signal_MACD1_B"
                           ,"signal_MACD2_B"
                           ,"signal_B"      
                           ,"signal_RSI_S"  
                           ,"signal_EMA1_S" 
                           ,"signal_EMA2_S" 
                           ,"signal_MACD1_S"
                           ,"signal_MACD2_S"
                           ,"signal_S"      
                           ,"dailyreturn"   
                           ,"weeklyreturn")
  Sig_Tab=as.data.frame(Sig_Tab)
  Sig_Tab$stock=tkr
  sig_day=tail(Sig_Tab,1)
  return(list(tab=Sig_Tab,day=sig_day))
}


tkrs=unique(c("RSI.TO","LSPD.TO","ATE.TO","EDGE.TO","ZEO.TO","ZRE.TO","RCI-B.TO","BTO.TO","ABX.TO","NWH-UN.TO","SJR-B.TO","IHI","XEG.TO","XMA.TO","XIN.TO","XMD.TO","XIT.TO","SU.TO","MFC.TO","TRP.TO","T.TO","BEPC.TO","KL.TO","CGL.TO","SKYY","RNW.TO","PSI","ICLN","ARKK","ETHQ.TO","RSI.TO","LSPD.TO","ATE.TO","LIFE.TO","MCHI","XCH.TO","IHF","ZEO.TO","XGD.TO","ZRE.TO","HACK","RCI-B.TO","BTO.TO","ABX.TO","NWH-UN.TO","XTR.TO","XEG.TO","XMA.TO","XIN.TO","XMD.TO","XIT.TO","BIP-UN.TO","AP-UN.TO","RTH","TAN","PBD","EMQQ","KGRN","VDY.TO","ROBO","BLDP.TO","AC.TO","DGRO","BCE.TO","ATZ.TO","ARRY","ZCH.TO","REI-UN.TO","BB.TO","GNOM","ARKG","VGRO.TO","SCHD","WCN.TO","RBA.TO","VRE.TO","XDSR.TO","BEP-UN.TO","XEQT.TO","XQQ.TO","VFV.TO","CNR.TO","XUH.TO","XUU.TO","PSI","IGV","PSJ","SOXX","IHI","IVW","XST.TO","XUT.TO","CPX.TO","HXS.TO","HXQ.TO","XHC.TO"))

TKRS=map(tkrs,sigtab)
day_tab=c()
for (i in 1:length(TKRS)) {
  x=TKRS[[i]]$day
  day_tab=rbind(x,day_tab)
}

 url=paste("https://finance.yahoo.com/quote/",day_tab$stock,sep = "")
 urls <- paste0("<a href='",url,"'>",url,"</a>")
 urls
# ```{r}
# library(dplyr)
# library(knitr)
# library(kableExtra)
# 
# day_tab %>% 
#   mutate(urlls = cell_spec(stock, "html", link = url)) %>%
#   kable("html", escape = FALSE) %>%
#   kable_styling(bootstrap_options = c("hover", "condensed"))
# ```
# 
# ```{r cars}
# dt <- day_tab
# dt$url <- urls
# # dt <- dt %>%
# #   select(url, everything())
# DT::datatable(
#   dt, escape = FALSE
# )
#```

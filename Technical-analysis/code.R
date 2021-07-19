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

SMA(Ad(spy),n=20)
EMA(Ad(spy),n=20)
chartSeries(spy,
            subset='2013::2016',
            theme=chartTheme('white'))
addMACD(fast=12,slow=26,signal=9,type="EMA")


signal1=1*(RSI(Ad(spy))< 30)
trade1 <- Lag(signal1)
trade1[is.na(trade1)]=0
ret1 <- dailyReturn(spy)*trade1
sum(ret1)
charts.PerformanceSummary(ret1)

RSI(Ad(spy),n=5)

signal2=1*(EMA(Ad(spy),n=12)>EMA(Ad(spy),n=26))
trade2 <- Lag(signal2)
trade2[is.na(trade2)]=0
ret2 <- dailyReturn(spy)*trade2
sum(ret2)
charts.PerformanceSummary(ret2)

signal2=1*(EMA(Ad(spy),n=24)>EMA(Ad(spy),n=52))
trade2 <- Lag(signal2)
trade2[is.na(trade2)]=0
ret2 <- dailyReturn(spy)*trade2
sum(ret2)
charts.PerformanceSummary(ret2)

signal3=1*(MACD(Ad(spy), nSig = 18)$signal>0)
trade3 <- Lag(signal3)
trade3[is.na(trade3)]=0
ret3 <- dailyReturn(spy)*trade3
sum(ret3)
charts.PerformanceSummary(ret3)

signal_RSI=1*(RSI(Cl(spy))< 30)%>%Lag()%>% replace(is.na(.), 0)
signal_EMA1=1*(EMA(Cl(spy),n=12)>EMA(Cl(spy),n=26))%>%Lag()%>% replace(is.na(.), 0)
signal_EMA2=1*(EMA(Cl(spy),n=24)>EMA(Cl(spy),n=52))%>%Lag()%>% replace(is.na(.), 0)
signal_MACD1=1*(MACD(Cl(spy))$signal>0)%>%Lag()%>% replace(is.na(.), 0)
signal_MACD2=1*(MACD(Ad(spy), nSig = 18)$signal>0)%>%Lag()%>% replace(is.na(.), 0)



ETHQ.TO
CTS.V
RSI.TO
GDNP.V
LSPD.TO
ATE.TO
EDGE.TO
ZEO.TO
ZRE.TO
RCI-B.TO
BTO.TO
ABX.TO
NWH-UN.TO
SJR-B.TO
IHI
XEG.TO
XMA.TO
XIN.TO
XMD.TO
XIT.TO
SU.TO
MFC.TO
TRP.TO
T.TO
BEPC.TO
KL.TO
CGL.TO
APHA.TO
SKYY
RNW.TO
PSI
ICLN
HEO.V
ABT.TO
ARKK
ETHQ.TO
CTS.V
RSI.TO
GDNP.V
LSPD.TO
ATE.TO
IPA.V
LIFE.TO
MCHI
XCH.TO
IHF
ZEO.TO
XGD.TO
ZRE.TO
HACK
RCI-B.TO
BTO.TO
ABX.TO
NWH-UN.TO
XTR.TO
XEG.TO
XMA.TO
XIN.TO
XMD.TO
XIT.TO
BIP-UN.TO
AP-UN.TO
RTH
TAN
PBD
EMQQ
KGRN
VDY.TO
ROBO
BLDP.TO
AC.TO
DGRO
BCE.TO
ATZ.TO
ARRY
ZCH.TO
REI-UN.TO
BB.TO
ABT.TO
GNOM
ARKG
VGRO.TO
SCHD
WCN.TO
RBA.TO
VRE.TO
XDSR.TO
BEP-UN.TO
XEQT.TO
XQQ.TO
VFV.TO
CNR.TO
XUH.TO
XUU.TO
PSI
IGV
PSJ
SOXX
IHI
IVW
XST.TO
XUT.TO
CPX.TO
HXS.TO
HXQ.TO
XHC.TO


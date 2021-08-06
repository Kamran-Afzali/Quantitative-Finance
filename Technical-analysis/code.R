rm(list=ls())

# install.packages("quantmod")
# install.packages("PerformanceAnalytics")
# install.packages("modeldata")
# install.packages("forecast")

library(PerformanceAnalytics)
library(quantmod)
library(tidyverse)
library(modeldata)
library(forecast)
library(finreportr)
#https://bookdown.org/kochiuyu/Technical-Analysis-with-R/charting-with-indicators.html
#https://lamfo-unb.github.io/2017/07/22/intro-stock-analysis-1/
#https://github.com/andrew-couch/Tidy-Tuesday/blob/master/TidyTuesdayForecasting.Rmd
#https://github.com/andrew-couch/Tidy-Tuesday/blob/master/Season%202/Scripts/TidyTuesdayAutoplot.Rmd
#https://business-science.github.io/timetk/reference/tk_augment_lags.html
#https://business-science.github.io/timetk/reference/tk_augment_timeseries.html
#https://business-science.github.io/timetk/articles/TK03_Forecasting_Using_Time_Series_Signature.html

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
spy <- getSymbols("spy", src = "yahoo", from = "2012-07-19", to = "2019-07-19", auto.assign = FALSE)
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

1*(BBands(Cl(spy))$pctB<0)%>%Lag()%>% replace(is.na(.), 0)
1*(BBands(Cl(spy))$pctB>1)%>%Lag()%>% replace(is.na(.), 0)
volatility(spy,  mean0=TRUE)%>%Lag()%>% replace(is.na(.), 0)
ADX(spy)$ADX%>%Lag()%>% replace(is.na(.), 0)
sum(dailyReturn(spy))
length(dailyReturn(spy))

ret <- dailyReturn(spy)*(1*(buy_trd_crt>0))+ dailyReturn(spy)*(1*!(sell_trd_crt>3))
sum(ret)

ret <- dailyReturn(spy)*(1*(buy_trd_crt>0))
sum(ret)

ret <-  dailyReturn(spy)*(1*!(sell_trd_crt>3))
sum(ret)




action=c()
action[1]=0
for (i in 1:length(sell_trd_crt)) {
  if (action[i]==0 & buy_trd_crt [i]>sell_trd_crt [i])
  { action[i]=1 }
  else if  (action[i]==1 & sell_trd_crt [i]>buy_trd_crt [i])
  { action[i]=0 }
  action[i+1]=action[i]
  }
action=action[1:length(action)-1]
sum(action)

actiondf=as.data.frame(cbind(buy_trd_crt,sell_trd_crt,action))

ret <-  dailyReturn(spy)*action
sum(ret)
sum(dailyReturn(spy))

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
  signal_BB_B=1*(BBands(Cl(spy))$pctB<0)%>%Lag()%>% replace(is.na(.), 0)
  signal_B      =signal_RSI_B+signal_EMA1_B+signal_EMA2_B+signal_MACD1_B+signal_MACD2_B+signal_BB_B
  signal_RSI_S  =1*(RSI(Cl(spy))> 70)%>%Lag()%>% replace(is.na(.), 0)
  signal_EMA1_S =1*(EMA(Cl(spy),n=12)<EMA(Cl(spy),n=26))%>%Lag()%>% replace(is.na(.), 0)
  signal_EMA2_S =1*(EMA(Cl(spy),n=24)<EMA(Cl(spy),n=52))%>%Lag()%>% replace(is.na(.), 0)
  signal_MACD1_S=1*(MACD(Cl(spy))$signal<0)%>%Lag()%>% replace(is.na(.), 0)
  signal_MACD2_S=1*(MACD(Cl(spy), nSig = 18)$signal<0)%>%Lag()%>% replace(is.na(.), 0)
  signal_BB_S=1*(BBands(Cl(spy))$pctB>1)%>%Lag()%>% replace(is.na(.), 0)
  
  signal_S      =signal_RSI_S+signal_EMA1_S+signal_EMA2_S+signal_MACD1_S+signal_MACD2_S+signal_BB_S
  
  dailyreturn   =(dailyReturn(spy)%>%Lag()%>% replace(is.na(.), 0))*100
  weeklyreturn  =(SMA(dailyReturn(spy),n=5)%>%Lag()%>% replace(is.na(.), 0))*100

  volatility=volatility(spy,  mean0=TRUE)%>%Lag()%>% replace(is.na(.), 0)
  ADX=ADX(spy)$ADX%>%Lag()%>% replace(is.na(.), 0)
  
  
  Sig_Tab=cbind( signal_RSI_B  
                ,signal_EMA1_B 
                ,signal_EMA2_B 
                ,signal_MACD1_B
                ,signal_MACD2_B
                ,signal_BB_B
                ,signal_B      
                ,signal_RSI_S  
                ,signal_EMA1_S 
                ,signal_EMA2_S 
                ,signal_MACD1_S
                ,signal_MACD2_S
                ,signal_BB_S
                ,signal_S      
                ,dailyreturn   
                ,weeklyreturn
                ,volatility
                ,ADX)
  
  colnames(Sig_Tab)=cbind( "signal_RSI_B"
                           ,"signal_EMA1_B" 
                           ,"signal_EMA2_B" 
                           ,"signal_MACD1_B"
                           ,"signal_MACD2_B"
                           ,"signal_BB_B"
                           ,"signal_B"      
                           ,"signal_RSI_S"  
                           ,"signal_EMA1_S" 
                           ,"signal_EMA2_S" 
                           ,"signal_MACD1_S"
                           ,"signal_MACD2_S"
                           ,"signal_BB_S"
                           ,"signal_S"      
                           ,"dailyreturn"   
                           ,"weeklyreturn"
                           ,"Vol"
                           ,"ADX")
  Sig_Tab=as.data.frame(Sig_Tab)
  Sig_Tab$stock=tkr
  sig_day=tail(Sig_Tab,1)
  sig_day_change=sig_day
  sig_day_change[,]=c(1*(Sig_Tab[nrow(Sig_Tab),1:ncol(sig_day)-1] - Sig_Tab[nrow(Sig_Tab)-1,1:ncol(sig_day)-1]),tkr)
  return(list(tab=Sig_Tab,day=sig_day,change=sig_day_change))
}


tkrs=unique(c("RSI.TO","LSPD.TO","ATE.TO","EDGE.TO","ZEO.TO","ZRE.TO","RCI-B.TO","BTO.TO","ABX.TO","NWH-UN.TO","SJR-B.TO","IHI","XEG.TO","XMA.TO","XIN.TO","XMD.TO","XIT.TO","SU.TO","MFC.TO","TRP.TO","T.TO","BEPC.TO","KL.TO","CGL.TO","SKYY","RNW.TO","PSI","ICLN","ARKK","ETHQ.TO","RSI.TO","LSPD.TO","ATE.TO","LIFE.TO","MCHI","XCH.TO","IHF","ZEO.TO","XGD.TO","ZRE.TO","HACK","RCI-B.TO","BTO.TO","ABX.TO","NWH-UN.TO","XTR.TO","XEG.TO","XMA.TO","XIN.TO","XMD.TO","XIT.TO","BIP-UN.TO","AP-UN.TO","RTH","TAN","PBD","EMQQ","KGRN","VDY.TO","ROBO","BLDP.TO","AC.TO","DGRO","BCE.TO","ATZ.TO","ARRY","ZCH.TO","REI-UN.TO","BB.TO","GNOM","ARKG","VGRO.TO","SCHD","WCN.TO","RBA.TO","VRE.TO","XDSR.TO","BEP-UN.TO","XEQT.TO","XQQ.TO","VFV.TO","CNR.TO","XUH.TO","XUU.TO","PSI","IGV","PSJ","SOXX","IHI","IVW","XST.TO","XUT.TO","CPX.TO","HXS.TO","HXQ.TO","XHC.TO"))

TKRS=map(tkrs,sigtab)
day_tab=c()
for (i in 1:length(TKRS)) {
  x=TKRS[[i]]$day
  day_tab=rbind(x,day_tab)
}


change_tab=c()
for (i in 1:length(TKRS)) {
  x=TKRS[[i]]$change
  change_tab=rbind(x,change_tab)
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

 
 
 AAPL <- getFin('AAPL')
 help("quantmod-defunct")
 AnnualReports("TSLA")
 AnnualReports("BABA", foreign = TRUE)
 CompanyInfo("GOOG")

 ################################################################################################
 library(tidymodels)
 library(stacks)
 library(finetune)
 library(vip)
 library(tidyposterior)
 library(modeldata)
 library(workflowsets)
 
 
 spy <- getSymbols("spy", src = "yahoo", from = Sys.Date()-1000, to = Sys.Date(), auto.assign = FALSE)
 signal_RSI_B  =1*(RSI(Cl(spy))< 30)%>%Lag()%>% replace(is.na(.), 0)
 signal_EMA1_B =1*(EMA(Cl(spy),n=12)>EMA(Cl(spy),n=26))%>%Lag()%>% replace(is.na(.), 0)
 signal_EMA2_B =1*(EMA(Cl(spy),n=24)>EMA(Cl(spy),n=52))%>%Lag()%>% replace(is.na(.), 0)
 signal_MACD1_B=1*(MACD(Cl(spy))$signal>0.3)%>%Lag()%>% replace(is.na(.), 0)
 signal_MACD2_B=1*(MACD(Cl(spy), nSig = 18)$signal>0.3)%>%Lag()%>% replace(is.na(.), 0)
 signal_BB_B=1*(BBands(Cl(spy))$pctB<0)%>%Lag()%>% replace(is.na(.), 0)
 signal_B      =signal_RSI_B+signal_EMA1_B+signal_EMA2_B+signal_MACD1_B+signal_MACD2_B+signal_BB_B
 signal_RSI_S  =1*(RSI(Cl(spy))> 70)%>%Lag()%>% replace(is.na(.), 0)
 signal_EMA1_S =1*(EMA(Cl(spy),n=12)<EMA(Cl(spy),n=26))%>%Lag()%>% replace(is.na(.), 0)
 signal_EMA2_S =1*(EMA(Cl(spy),n=24)<EMA(Cl(spy),n=52))%>%Lag()%>% replace(is.na(.), 0)
 signal_MACD1_S=1*(MACD(Cl(spy))$signal<0)%>%Lag()%>% replace(is.na(.), 0)
 signal_MACD2_S=1*(MACD(Cl(spy), nSig = 18)$signal<0)%>%Lag()%>% replace(is.na(.), 0)
 signal_BB_S=1*(BBands(Cl(spy))$pctB>1)%>%Lag()%>% replace(is.na(.), 0)
 signal_S      =signal_RSI_S+signal_EMA1_S+signal_EMA2_S+signal_MACD1_S+signal_MACD2_S+signal_BB_S
 dailyreturn   =(dailyReturn(spy)%>%Lag()%>% replace(is.na(.), 0))*100
 weeklyreturn  =(SMA(dailyReturn(spy),n=5)%>%Lag()%>% replace(is.na(.), 0))*100
 volatility=volatility(spy,  mean0=TRUE)%>%Lag()%>% replace(is.na(.), 0)
 ADX=ADX(spy)$ADX%>%Lag()%>% replace(is.na(.), 0)
 
 spy <- getSymbols("spy", src = "yahoo", from = Sys.Date()-1000, to = Sys.Date(), auto.assign = FALSE)
 esd=dailyReturn(spy)
 spy=spy[-c(which(esd>mean(esd)+(2*sd(esd))|esd<mean(esd)-(2*sd(esd)))),]
 data=as.data.frame(cbind(
 (1*(RSI(Cl(spy))< 30)),
 1*(EMA(Cl(spy),n=12)>EMA(Cl(spy),n=26)),
 1*(EMA(Cl(spy),n=24)>EMA(Cl(spy),n=52)),
 1*(MACD(Cl(spy))$signal>0.3),
 1*(MACD(Cl(spy), nSig = 18)$signal>0.3),
 1*(BBands(Cl(spy))$pctB<0),
 1*(RSI(Cl(spy))> 70),
 1*(EMA(Cl(spy),n=12)<EMA(Cl(spy),n=26)),
 1*(EMA(Cl(spy),n=24)<EMA(Cl(spy),n=52)),
 1*(MACD(Cl(spy))$signal<0),
 1*(MACD(Cl(spy), nSig = 18)$signal<0),
 1*(BBands(Cl(spy))$pctB>1),
 RSI(Cl(spy)),
 MACD(Cl(spy))$signal,
 MACD(Cl(spy), nSig = 18)$signal,
 BBands(Cl(spy))$pctB,
 volatility(spy,  mean0=TRUE),
 ADX(spy)$DIp,
 ADX(spy)$DIn,
 ADX(spy)$ADX,
 SMA(dailyReturn(spy),n=5)>mean(SMA(dailyReturn(spy),n=5),na.rm=T)+(0.5*sd(SMA(dailyReturn(spy),n=5),na.rm=T))))

 
 colnames(data)=c("signal_RSI_B","signal_EMA1_B","signal_EMA2_B","signal_MACD1_B","signal_MACD2_B","signal_BB_B","signal_RSI_S","signal_EMA1_S","signal_EMA2_S","signal_MACD1_S","signal_MACD2_S","signal_BB_S","RSI","MACD1","MACD2","BB","volatility","ADX1","ADX2","ADX3","signal_weeklyreturn")  
 
 data= data%>%
   mutate(sell=signal_RSI_S+signal_EMA1_S+signal_EMA2_S+signal_MACD1_S+signal_MACD2_S+signal_BB_S,
               Buy=signal_RSI_B+signal_EMA1_B+signal_EMA2_B+signal_MACD1_B+signal_MACD2_B+signal_BB_B)%>%drop_na()
 
 #Lag
 
 data.t=tail(data,10)
 data=data%>%Lag(5)
 #%>%mutate_at(vars(starts_with("signal_")), funs(as.factor))
 data$signal_weeklyreturn=as.factor( data$signal_weeklyreturn)
 set.seed(1)
 class_split <- initial_split(data, strata = "signal_weeklyreturn")
 class_train <- training(class_split)
 test_data <- testing(class_split)
 class_k_folds <- vfold_cv(class_train)
 

 

 
 class_rec <- recipe(signal_weeklyreturn~., data = class_train) %>%
   step_center(all_numeric_predictors())  %>%
   step_scale(all_numeric_predictors()) %>%
   #step_corr(all_numeric_predictors()) %>% 
   #step_lincomb(all_numeric_predictors()) %>% 
   themis::step_smote (signal_weeklyreturn)
 
 train_preped <- prep(class_rec) %>%
   bake(new_data = NULL)
 
 test_preped <-  prep(class_rec) %>%
   bake(new_data = test_data)
 
 elastic_class <- logistic_reg(mixture = tune(), penalty = tune()) %>% 
   set_mode("classification") %>% 
   set_engine("glmnet")
 xgboost_class <- boost_tree(learn_rate = tune(), trees = tune()) %>% 
   set_mode("classification") %>% 
   set_engine("xgboost")
 randomForest_class <- rand_forest(trees = tune()) %>% 
   set_mode("classification") %>% 
   set_engine("ranger")
 

 classification_metrics <- metric_set(roc_auc)
 model_control <- control_stack_grid()

 classification_set <- workflow_set(
   preproc = list(regular = class_rec),
   models = list(elastic = elastic_class, xgboost = xgboost_class, randomForest = randomForest_class),
   cross = TRUE )

 
 classification_set <- classification_set %>% 
   workflow_map("tune_sim_anneal", resamples = class_k_folds, metrics = classification_metrics)
 autoplot(classification_set)

 
 autoplot(classification_set, rank_metric = "roc_auc", id = "regular_elastic")
 rank_results(classification_set, rank_metric = "roc_auc") %>% 
   filter(.metric == "roc_auc")
 classification_set %>% 
   extract_workflow_set_result("regular_elastic") %>% 
   show_best("roc_auc", n = 1)
 classification_set %>% 
   extract_workflow_set_result("regular_randomForest") %>% 
   show_best("roc_auc", n = 1)
 
 
 xgb_best=classification_set %>% 
   extract_workflow_set_result("regular_xgboost") %>% 
   show_best("roc_auc", n = 1)

 
 
 final_xgb <- finalize_model(
   xgboost_class,
   xgb_best
 )
 
 final_xgb
 
 final_xgb %>%
   set_mode("classification") %>% 
   set_engine("xgboost")%>%
   fit(signal_weeklyreturn ~ .,
       data = train_preped
   ) %>%
   vip()
 
 mod_pred=final_xgb %>%
   set_mode("classification") %>% 
   set_engine("xgboost")%>%
   fit(signal_weeklyreturn ~ .,
       data = train_preped
   ) %>% predict(test_preped)%>% 
   bind_cols(test_preped %>% select(signal_weeklyreturn))
 
mod_pred%>% yardstick::accuracy(truth = signal_weeklyreturn, .pred_class)%>%bind_rows(mod_pred%>% yardstick::sens(truth = signal_weeklyreturn, .pred_class))%>%
   bind_rows(mod_pred%>% yardstick::spec(truth = signal_weeklyreturn, .pred_class))%>%bind_rows(mod_pred%>% yardstick::f_meas(truth = signal_weeklyreturn, .pred_class))
  

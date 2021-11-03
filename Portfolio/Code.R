library(tidyquant) # To download the data
library(plotly) # To create interactive charts
library(timetk) # To manipulate the data series
library(forcats)
library(tidyr)
#https://seekingalpha.com/etfs-and-funds/etf-tables/sectors?utm_source=google&utm_medium=cpc&utm_campaign=14049528666&utm_term=127926794296%5Eaud-1457157706959:dsa-1427142718946%5Eb%5E547566878395%5E%5E%5Eg&gclid=Cj0KCQjw5oiMBhDtARIsAJi0qk2LjR58Nfps9hx7OgGrL_XmycmlH96YxiapMt-b5as3aFIQfQ5ggoIaAlOwEALw_wcB
#https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/Betas.html
tick <- c('IXC', 'IXG', 'IXN', 'IXJ', 'IXP','RXI','EXI','MXI','KXI','JXI')

price_data <- tq_get(tick,
                     from = '2010-01-01',
                     to = '2021-11-01',
                     get = 'stock.prices')

log_ret_tidy <- price_data %>%
  group_by(symbol) %>%
  tq_transmute(select = adjusted,
               mutate_fun = periodReturn,
               period = 'daily',
               col_rename = 'ret',
               type = 'log')

head(log_ret_tidy)

log_ret_xts <- log_ret_tidy %>%
  spread(symbol, value = ret) %>%
  tk_xts()

head(log_ret_xts)

summary(log_ret_xts)

log_ret_xts=log_ret_xts%>%as.data.frame()%>%drop_na()

summary(log_ret_xts)

head(log_ret_xts)


mean_ret <- colMeans(log_ret_xts,na.rm = T)
print(round(mean_ret, 5))


cov_mat <- cov(log_ret_xts) * 252

print(round(cov_mat,4))


wts <- runif(n = length(tick))
print(wts)

print(sum(wts))



wts <- wts/sum(wts)
print(wts)



port_returns <- (sum(wts * mean_ret) + 1)^252 - 1


port_risk <- sqrt(t(wts) %*% (cov_mat %*% wts))
print(port_risk)


sharpe_ratio <- port_returns/port_risk
print(sharpe_ratio)



num_port <- 10000

# Creating a matrix to store the weights

all_wts <- matrix(nrow = num_port,
                  ncol = length(tick))

# Creating an empty vector to store
# Portfolio returns

port_returns <- vector('numeric', length = num_port)

# Creating an empty vector to store
# Portfolio Standard deviation

port_risk <- vector('numeric', length = num_port)

# Creating an empty vector to store
# Portfolio Sharpe Ratio

sharpe_ratio <- vector('numeric', length = num_port)



for (i in seq_along(port_returns)) {
  
  wts <- runif(length(tick))
  wts <- wts/sum(wts)
  
  # Storing weight in the matrix
  all_wts[i,] <- wts
  
  # Portfolio returns
  
  port_ret <- sum(wts * mean_ret)
  port_ret <- ((port_ret + 1)^252) - 1
  
  # Storing Portfolio Returns values
  port_returns[i] <- port_ret
  
  
  # Creating and storing portfolio risk
  port_sd <- sqrt(t(wts) %*% (cov_mat  %*% wts))
  port_risk[i] <- port_sd
  
  # Creating and storing Portfolio Sharpe Ratios
  # Assuming 0% Risk free rate
  
  sr <- port_ret/port_sd
  sharpe_ratio[i] <- sr
  
}




portfolio_values <- tibble(Return = port_returns,
                           Risk = port_risk,
                           SharpeRatio = sharpe_ratio)


# Converting matrix to a tibble and changing column names
all_wts <- tk_tbl(all_wts)
## Warning in tk_tbl.data.frame(as.data.frame(data), preserve_index,
## rename_index, : Warning: No index to preserve. Object otherwise converted
## to tibble successfully.
colnames(all_wts) <- colnames(log_ret_xts)

# Combing all the values together
portfolio_values <- tk_tbl(cbind(all_wts, portfolio_values))
head(portfolio_values)


min_var <- portfolio_values[which.min(portfolio_values$Risk),]
max_sr <- portfolio_values[which.max(portfolio_values$SharpeRatio),]

p <- min_var %>%
  gather(EXI:RXI, key = Asset,
         value = Weights) %>%
  mutate(Asset = as.factor(Asset)) %>%
  ggplot(aes(x = fct_reorder(Asset,Weights), y = Weights, fill = Asset)) +
  geom_bar(stat = 'identity') +
  theme_minimal() +
  labs(x = 'Assets', y = 'Weights', title = "Minimum Variance Portfolio Weights") +
  scale_y_continuous(labels = scales::percent) 

ggplotly(p)


p <- max_sr %>%
  gather(EXI:RXI, key = Asset,
         value = Weights) %>%
  mutate(Asset = as.factor(Asset)) %>%
  ggplot(aes(x = fct_reorder(Asset,Weights), y = Weights, fill = Asset)) +
  geom_bar(stat = 'identity') +
  theme_minimal() +
  labs(x = 'Assets', y = 'Weights', title = "Tangency Portfolio Weights") +
  scale_y_continuous(labels = scales::percent) 

ggplotly(p)



p <- portfolio_values %>%
  ggplot(aes(x = Risk, y = Return, color = SharpeRatio)) +
  geom_point() +
  theme_classic() +
  scale_y_continuous(labels = scales::percent) +
  scale_x_continuous(labels = scales::percent) +
  labs(x = 'Annualized Risk',
       y = 'Annualized Returns',
       title = "Portfolio Optimization & Efficient Frontier") +
  geom_point(aes(x = Risk,
                 y = Return), data = min_var, color = 'red') +
  geom_point(aes(x = Risk,
                 y = Return), data = max_sr, color = 'red')
p
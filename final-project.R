library(readxl)   
library(dplyr)    
library(ggplot2)
library(usmap)

#Data cleaning
# Convert Private to a binary variable
College$Private <- ifelse(College$Private == "Yes", 1, 0)

# Handle missing values
sum(is.na(College)) # Check for missing values
College <- na.omit(College) # Remove rows with missing values

# Check for outliers in Grad.Rate
summary(College$Grad.Rate)
College[College$Grad.Rate > 100, ]
College <- College[College$Grad.Rate <= 100, ] 
# Standardize continuous variables
# Variables to standardize
scale_vars <- c("Outstate", "Expend", "S.F.Ratio", "Top10perc", "Top25perc",
  "Apps", "Accept", "Enroll", "F.Undergrad", "P.Undergrad",
  "Room.Board", "Books", "Personal"
)

# Apply standardization
College[scale_vars] <- scale(College[scale_vars])

#HIST Grad.Rate
hist(College$Grad.Rate, 
     breaks = 20, 
     probability = TRUE, 
     col = "steelblue", 
     main = "Distribution of Graduation Rates (%)", 
     xlab = "Graduation Rate (%)", 
     border = "black", 
     xlim = c(0, 120))

curve(dnorm(x, mean = mean(College$Grad.Rate, na.rm = TRUE), 
            sd = sd(College$Grad.Rate, na.rm = TRUE)), 
      col = "red", 
      lwd = 2, 
      add = TRUE)


#Transformation of Grad.Rate
# Convert Graduation Rate to proportion
College$Grad.Rate.Prop <- College$Grad.Rate / 100
epsilon <- 0.001
College$Grad.Rate.Prop <- pmax(epsilon, pmin(College$Grad.Rate.Prop, 1 - epsilon))
# Apply logit transformation
College$Grad.Rate.Logit <- log(College$Grad.Rate.Prop / (1 - College$Grad.Rate.Prop))

#HIST Grad.Rate.Logit
mean_logit <- mean(College$Grad.Rate.Logit, na.rm = TRUE)
sd_logit <- sd(College$Grad.Rate.Logit, na.rm = TRUE)

hist(College$Grad.Rate.Logit,
     breaks = 30,                  
     probability = TRUE,          
     main = "Distribution of Logit-Transformed Graduation Rates", 
     xlab = "Logit(Graduation Rate)", 
     col = "steelblue",            
     border = "black")            

curve(dnorm(x, mean = mean_logit, sd = sd_logit),
      col = "red",                 
      lwd = 2,                     
      lty = 2,                     
      add = TRUE)                  


library(ggplot2)
#Boxplot 
ggplot(College, aes(x = factor(Private, labels = c("Public", "Private")), y = Grad.Rate)) +
  geom_boxplot(fill = c("purple", "yellow"), alpha = 0.7) +
  labs(
    title = NULL,
    x = "Institution Type",
    y = "Graduation Rate (%)"
  ) +
  theme_minimal()


library(ggcorrplot)
library(dplyr) 
table_summary <- College %>%
  group_by(Private) %>%
  summarise(
    Mean_Grad_Rate = mean(Grad.Rate, na.rm = TRUE),
    Proportion = n() / nrow(College)
  )

kable(
  table_summary,
  col.names = c("Type", "Mean Graduation Rate", "Proportion"),
  caption = "Mean Graduation Rate and Proportion by College Type"
)

# correlation matrix for all numeric variables
cor_matrix <- cor(College[, sapply(College, is.numeric)], use = "complete.obs")

# heatmap
ggcorrplot(cor_matrix, 
           method = "square", 
           lab = FALSE,
           colors = c("red", "white", "blue"), 
           title = "Heatmap",
           hc.order = TRUE) +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#MAP
state_grad_rate <- College %>%
  group_by(state) %>%
  summarise(Mean_Grad_Rate = mean(Grad.Rate, na.rm = TRUE))

plot_usmap(data = state_grad_rate, values = "Mean_Grad_Rate", regions = "states") +
  scale_fill_continuous(name = "Graduation Rate (%)", low = "yellow", high = "red") +
  labs(title = "Graduation Rate by State in the U.S.") +
  theme(legend.position = "right")


# Create data frame
df <- data.frame(
  State = c("New York", "Pennsylvania", "Ohio", "Illinois", "Virginia", 
            "Massachusetts", "North Carolina", "California", "Texas", "Indiana"),
  Colleges = c(64, 60, 34, 33, 33, 32, 31, 30, 30, 27)
)
# Create bar chart
ggplot(df, aes(x = reorder(State, Colleges), y = Colleges)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 States with the Highest Number of Colleges",
       x = "State",
       y = "Number of Colleges") +
  theme_minimal()




#####
library(knitr)
library(ggplot2)
library(mice)
library(patchwork)
library(tidyr)
library(rjags)
library(R2jags)
library(coda)
library(lme4)
library(readr)
# Center the continuous predictors
College$Expend <- as.vector(scale(College$Expend, center = TRUE, scale = FALSE))
College$Outstate <- as.vector(scale(College$Outstate, center = TRUE, scale = FALSE))
College$S.F.Ratio <- as.vector(scale(College$S.F.Ratio, center = TRUE, scale = FALSE))
College$perc.alumni <- as.vector(scale(College$perc.alumni, center = TRUE, scale = FALSE))
College$Top10perc <- as.vector(scale(College$Top10perc, center = TRUE, scale = FALSE))
College$Terminal <- as.vector(scale(College$Terminal, center = TRUE, scale = FALSE))
College$F.Undergrad <- as.vector(scale(College$F.Undergrad, center = TRUE, scale = FALSE))
College$Books <- as.vector(scale(College$Books, center = TRUE, scale = FALSE))
College$Personal <- as.vector(scale(College$Personal, center = TRUE, scale = FALSE))
College$Enroll <- as.vector(scale(College$Enroll, center = TRUE, scale = FALSE))

# data1 for JAGS 
jags_data1 <- list(
  y = College$Grad.Rate.Logit,  # Logit-transformed graduation rate
  state = as.numeric(as.factor(College$state)),  # State as grouping variable
  N = nrow(College),  # Number of colleges
  J = length(unique(College$state))  # Number of unique states (random effect)
)

model_code1 <- "
model {
  for (i in 1:N) {
    # Likelihood
    y[i] ~ dnorm(mu[i], tau) # Normal likelihood with precision tau
    mu[i] <- beta0 + u_state[state[i]] # Random intercept for states

    #posterior predictive samples for y, helping assess model fit.
    yrep[i] ~ dnorm(mu[i], tau) # Posterior predictions for y
  }
  
  # Random effects for states
  for (j in 1:J) {
    u_state[j] ~ dnorm(0, tau_u)
  }
  
  # Prior for intercept
  beta0 ~ dnorm(0, 0.001)
  
  # Priors for variances
  tau ~ dgamma(0.001, 0.001)
  sigma <- 1 / sqrt(tau)

  tau_u ~ dgamma(0.001, 0.001) # Random effect precision
  sigma_u <- 1 / sqrt(tau_u) # Standard deviation for state-level variation
}
"

params1 <- c("beta0", "sigma", "sigma_u")

set.seed(123)
jags_model1 <- jags(
  data = jags_data1,
  parameters.to.save = params1,
  model.file = textConnection(model_code1),
  n.chains = 1, # Number of chains
  n.iter = 10000, # Total iterations
  n.burnin = 1000, # Burn-in iterations
  n.thin = 10, # Thinning factor
  DIC = TRUE # Calculate DIC for model comparison
)


# data2 for JAGS 
jags_data2 <- list(
  y = College$Grad.Rate.Logit,   #outcome
  x1 = College$Enroll,     #predictor 1
  x2 = College$Outstate,   #predictor 2
  x3 = College$S.F.Ratio,  #predictor 3
  state = as.numeric(as.factor(College$state)),  # State as grouping variable
  N = nrow(College),   #number of college
  J = length(unique(College$state))    # Number of unique states (random effect)
)


model_code2 <- "
model {
  for (i in 1:N) {
    # Likelihood
    y[i] ~ dnorm(mu[i], tau) # Normal likelihood with precision tau
    mu[i] <- beta0 + beta1 * x1[i] + beta2 * x2[i] + beta3 * x3[i] 
             + u_state[state[i]]

    # Posterior predictive distribution
    yrep[i] ~ dnorm(mu[i], tau) # Posterior predictions for y
  }
  
  # Random effects for states
  for (j in 1:J) {
    u_state[j] ~ dnorm(0, tau_u)
  }

  # Priors for fixed effects
  beta0 ~ dnorm(0, 0.001)
  beta1 ~ dnorm(0, 0.001)
  beta2 ~ dnorm(0, 0.001)
  beta3 ~ dnorm(0, 0.001)

  # Priors for variances
  tau ~ dgamma(0.001, 0.001)
  sigma <- 1 / sqrt(tau)

  tau_u ~ dgamma(0.001, 0.001) # Random effect precision
  sigma_u <- 1 / sqrt(tau_u) # Standard deviation for state-level variation
}
"

# Parameters to monitor
params2 <- c("beta0", "beta1", "beta2", "beta3", "sigma", "sigma_u")

# Run the JAGS model2
set.seed(123) 
jags_model2 <- jags(
data = jags_data2,
parameters.to.save = params2,
model.file = textConnection(model_code2),
n.chains = 1, # Number of chains
n.iter = 10000, # Total iterations
n.burnin = 1000, # Burn-in iterations
n.thin = 10, # Thinning factor
DIC = TRUE # Calculate DIC for model comparison
)


# Data for JAGS Model 3

# Prepare data for JAGS
jags_data3 <- list(
  y = College$Grad.Rate.Logit,   # Logit-transformed graduation rate
  x1 = College$Enroll,           # Predictor 1: Number of new students enrolled
  x2 = College$Outstate,         # Predictor 2: Out-of-state tuition
  x3 = College$S.F.Ratio,        # Predictor 3: Student-to-faculty ratio
  x4 = College$perc.alumni,      # Predictor 4: Percentage of alumni donations
  x5 = College$Top10perc,        # Predictor 5: Percentage of top 10% students
  x6 = College$Terminal,         # Predictor 6: Percentage of faculty with terminal degrees
  x7 = as.numeric(College$Private), # Predictor 7: Private (1 = Private, 0 = Public)
  state = as.numeric(as.factor(College$state)),  # State as grouping variable
  N = nrow(College),   # Number of colleges
  J = length(unique(College$state))  # Number of unique states (random effect)
)

# Define JAGS Model
model_code3 <- "
model {
  for (i in 1:N) {
    # Likelihood
    y[i] ~ dnorm(mu[i], tau) # Normal likelihood with precision tau
    mu[i] <- beta0 + beta1 * x1[i] + beta2 * x2[i] + beta3 * x3[i] +
             beta4 * x4[i] + beta5 * x5[i] + beta6 * x6[i] + beta7 * x7[i] +
             u_state[state[i]]

    # Posterior predictive distribution
    yrep[i] ~ dnorm(mu[i], tau) #Posterior predictions for y
  }
  
  # Random effects for states
  for (j in 1:J) {
    u_state[j] ~ dnorm(0, tau_u)
  }

  # Priors for fixed effects
  beta0 ~ dnorm(0, 0.001)
  beta1 ~ dnorm(0, 0.001)
  beta2 ~ dnorm(0, 0.001)
  beta3 ~ dnorm(0, 0.001)
  beta4 ~ dnorm(0, 0.001)
  beta5 ~ dnorm(0, 0.001)
  beta6 ~ dnorm(0, 0.001)
  beta7 ~ dnorm(0, 0.001) 

  # Priors for variances
  tau ~ dgamma(0.001, 0.001)  
  sigma <- 1 / sqrt(tau)      

  tau_u ~ dgamma(0.001, 0.001) 
  sigma_u <- 1 / sqrt(tau_u) 
}
"

#parameters to monitor posterior distributions
params3 <- c("beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "beta6",
             "beta7", "sigma", "sigma_u")

#JAGS Model 3
set.seed(123)
jags_model3 <- jags(
  data = jags_data3,
  parameters.to.save = params3,
  model.file = textConnection(model_code3),
  n.chains = 1, # Number of chains
  n.iter = 10000, # Total iterations
  n.burnin = 1000, # Burn-in iterations
  n.thin = 10, # Thinning factor
  DIC = TRUE # Calculate DIC for model comparison
)


# data for JAGS 4
jags_data4 <- list(
  y = College$Grad.Rate.Logit,   
  x1 = College$Enroll,           
  x2 = College$Outstate,         
  x3 = College$S.F.Ratio,        
  x4 = College$perc.alumni,      
  x5 = College$Top10perc,        
  x6 = College$Terminal,         
  x7 = as.numeric(College$Private),  
  N = nrow(College)   
)

# Define JAGS Model (Fixed Effects Only)
model_code4 <- "
model {
  for (i in 1:N) {
    # Likelihood
    y[i] ~ dnorm(mu[i], tau) 
    mu[i] <- beta0 + beta1 * x1[i] + beta2 * x2[i] + beta3 * x3[i] +
             beta4 * x4[i] + beta5 * x5[i] + beta6 * x6[i] + beta7 * x7[i]

    # Posterior predictive distribution
    yrep[i] ~ dnorm(mu[i], tau) # Posterior predictions for y
  }

  # Priors for fixed effects
  beta0 ~ dnorm(0, 0.001)
  beta1 ~ dnorm(0, 0.001)
  beta2 ~ dnorm(0, 0.001)
  beta3 ~ dnorm(0, 0.001)
  beta4 ~ dnorm(0, 0.001)
  beta5 ~ dnorm(0, 0.001)
  beta6 ~ dnorm(0, 0.001)
  beta7 ~ dnorm(0, 0.001) 

  # Prior for variance
  tau ~ dgamma(0.001, 0.001)
  sigma <- 1 / sqrt(tau)
}
"

# Parameters to monitor
params4 <- c("beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "beta6", 
             "beta7", "sigma")

# Run JAGS Model
set.seed(123)
jags_model4 <- jags(
  data = jags_data4,
  parameters.to.save = params4,
  model.file = textConnection(model_code4),
  n.chains = 1, # Number of chains
  n.iter = 10000, # Total iterations
  n.burnin = 1000, # Burn-in iterations
  n.thin = 10, # Thinning factor
  DIC = TRUE # Calculate DIC for model comparison
)


# DIC
dic_values <- c(
model1 = jags_model1$BUGSoutput$DIC,
model2 = jags_model2$BUGSoutput$DIC,
model3 = jags_model3$BUGSoutput$DIC,
model4 = jags_model4$BUGSoutput$DIC)
dic_table <- data.frame( Model = names(dic_values), 
                         DIC = as.numeric(dic_values))
kable(dic_table, caption = "DIC Values")

#model3
print(jags_model3)



#DIAGNOSTIC
# Define parameters to monitor
params3diag <- c("beta0", "beta1", "beta2", "beta3", "beta4", "beta5",
             "beta6", "beta7", "sigma", "sigma_u", "yrep", "mu", "u_state")

# Run JAGS Model 3
set.seed(123)
jags_model3diag <- jags(
  data = jags_data3,
  parameters.to.save = params3diag,
  model.file = textConnection(model_code3),
  n.chains = 1,  # Number of chains
  n.iter = 10000,  # Total iterations
  n.burnin = 1000,  # Burn-in iterations
  n.thin = 10,  # Thinning factor
  DIC = TRUE  # Calculate DIC for model comparison
)

# TRACEPOLOTS FOR CONVERGENCE
mcmc_samples <- as.mcmc(jags_model3diag$BUGSoutput$sims.matrix)
par(mfrow=c(2,4))
traceplot(mcmc_samples[, c("beta0", "beta1", "beta2", "beta3",
                           "beta4", "beta5", "beta6", "beta7")])
# EFFECTIVE SAMPLE SIZE (ESS)
head(effectiveSize(mcmc_samples), 10) # Show the first 10 rows

 #POSTERIOR PREDICTIVE CHECK
# Extract posterior predictive samples
posterior_pred <- jags_model3diag$BUGSoutput$sims.list$yrep
observed <- jags_data3$y
# histogram for observed data
hist(observed, breaks = 30, col = rgb(0, 0, 1, 0.5), prob = TRUE,
     main = "Posterior Predictive",
     xlab = "Observed Graduation Rates (Logit)")
lines(density(as.vector(posterior_pred)), col = "red", lwd = 2, lty=2)
legend("topright", legend = c("Observed", "Posterior Predictive"), 
       col = c("blue", "red"), lwd = 2, bty = "n")



#DIAGNOSTIC RESIDUALS
fitted_values <- jags_model3diag$BUGSoutput$sims.list$mu  #predicted values
residuals <- observed - colMeans(fitted_values)           

# Residuals vs. Predicted values
par(mfrow = c(1, 2))
plot(colMeans(fitted_values), residuals,
     main = "Residuals vs. Fitted Values",
     xlab = "Predicted Values",
     ylab = "Residuals",
     pch = 16, col = "blue")
abline(h = 0, col = "red", lwd = 2)

# Q-Q Plot 
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red", lwd=2)
par(mfrow = c(1, 1))

# Hist residuals
hist(residuals, breaks = 30, col = "lightblue",
     main = "Histogram of Residuals",
     xlab = "Residuals", prob = TRUE)
lines(density(residuals), col = "red", lwd = 2)

# Autocorrelation
acf(residuals, main = "ACF of Residuals", col = "blue")
         


# RANDOM EFFECTS
# Hist
random_effects <- apply(jags_model3diag$BUGSoutput$sims.list$u_state, 2 ,mean)
#This object contains the MCMC chains for random effects (u_state) generated 
#during the Bayesian estimation. It is an array in which the rows represent the 
#iterations of the MCMC chain and the columns represent the groups ( states).

hist(random_effects, 
     breaks = 20, 
     col = "blue",  
     prob = TRUE,  
     main = "Histogram and Density of Random Effects",
     xlab = "Random Effects (State-Level)")

# Q-Q plot
qqnorm(random_effects)  # Q-Q plot
qqline(random_effects, col = "red", lwd = 2)  


# Caterpillar
# Extract random effects means and credible intervals
random_effects <- apply(jags_model3diag$BUGSoutput$sims.list$u_state, 2 ,mean)
lower_bounds <- apply(jags_model3diag$BUGSoutput$sims.list$u_state, 2,
                      function(x) quantile(x, 0.025)) # 2.5% quantile 
upper_bounds <- apply(jags_model3diag$BUGSoutput$sims.list$u_state, 2,
                      function(x) quantile(x, 0.975)) # 97.5% quantile

random_effects_df <- data.frame( 
  State = 1:length(random_effects), 
  Effect = random_effects,
  Lower = lower_bounds,
  Upper = upper_bounds
)

# Order the states by effect size
random_effects_df <- random_effects_df[order(random_effects_df$Effect), ]
random_effects_df$State <- 1:nrow(random_effects_df)  

library(ggplot2)
ggplot(random_effects_df, aes(x = State, y = Effect)) +
geom_point(color = "blue") +
geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2, color = "gray") + 
  labs(
    title = "Caterpillar Plot of State-Level Random Effects",
    x = "State (Ordered by Effect Size)",
    y = "Random Effect"
    ) + 
  theme_minimal() + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14), 
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
)

#Bayesian Posterior Predictive Checks at the State Level
y_pred <- apply(jags_model3diag$BUGSoutput$sims.list$yrep, 2, mean)
state_means <- aggregate(y_pred, by = list(College$state), FUN = mean)
colnames(state_means) <- c("State", "Pred_GradRate")

actual_means <- aggregate(College$Grad.Rate.Logit, by = list(College$state), FUN = mean)
colnames(actual_means) <- c("State", "Actual_GradRate")

comparison <- merge(state_means, actual_means, by = "State")
ggplot(comparison, aes(x = Actual_GradRate, y = Pred_GradRate, label = State)) +
  geom_point() +
  geom_text(vjust = -0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Predicted vs. Observed Graduation Rates by State", x = "Observed", y = "Predicted") +
  theme_minimal()

#FREQUENTIST / BAYESIAN
# Frequentist mixed-effects
library(lme4)
frequentist_model <- lmer(Grad.Rate.Logit ~ Enroll + Outstate + S.F.Ratio + 
                            perc.alumni + Top10perc + Terminal + Private + 
                            (1 | state), data = College)
summary(frequentist_model)

# JAGS " (Model 3)
print("Bayesian mixed-effects model summary:") 
print(jags_model3)    
          


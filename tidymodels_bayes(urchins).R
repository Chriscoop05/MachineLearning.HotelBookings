library(tidymodels)
library(tidyverse)
library(caret)
library(ranger)
library(mlbench)
library(glmnet)
library(vip)
library(broom.mixed)
library(dotwhisker) 




#Width is size at end of experiment
#initial volume is volume at the beginning

urchins <-
  read_csv("https://tidymodels.org/start/models/urchins.csv") %>% 
  setNames(c("food_regime", "initial_volume", "width")) %>% 
  mutate(food_regime = factor(food_regime, levels = c("Initial", "Low", "High")))



ggplot(urchins,
       aes(x = initial_volume, 
           y = width, 
           group = food_regime, 
           col = food_regime)) + 
  geom_point() + 
  geom_smooth(method = lm, se = FALSE) +
  scale_color_viridis_d(option = "plasma", end = .7)

#Urchins that were larger in volume at the start
#end with wider sutures but slopes are different per regime
#Thus effects vary


#To analyze the effect the initial volume + food regime has on suture
#we will use a two-way ANOVA
urchin.linear.mod = 
  linear_reg() %>% 
    set_engine("lm")


#Fit the model
urchin.linear.fit = 
  urchin.linear.mod %>% 
  fit(width ~ initial_volume*food_regime, data = urchins)


tidy(urchin.linear.fit) %>% 
  dwplot(dot_args = list(size = 2, color ='black'),
         whisker_args = list(color = 'black'),
         vline = geom_vline(xintercept = 0,
                            color ='grey50',
                            linetype =2))




urchins.new.points = expand.grid(initial_volume = 20,
                                 food_regime = c("Initial",
                                                 "Low",
                                                 "High"))


urchin.mean.new.pred.size = predict(urchin.linear.fit,
                                    new_data = urchins.new.points)

urchin.mean.new.pred.size



urchin.confint.new.pred.size = predict(urchin.linear.fit,
                                       new_data = urchins.new.points,
                                       type = "conf_int")

#Combine the new points data, predicted points, and conf interval
urchin.plot.predictions = 
  cbind(urchins.new.points,
        urchin.mean.new.pred.size,
        urchin.confint.new.pred.size)

ggplot(urchin.plot.predictions,
       aes(x = food_regime))+
    geom_point(aes( y = .pred))+
    geom_errorbar(aes(ymin = .pred_lower,
                      ymax = .pred_upper),
                  width = .2)+
    labs(y = "urchin size")+
    ggtitle("Food Regime Urchin Width Predictions")+
    theme(
      plot.title = element_text(hjust = 0.5)
    )








#Model this same problem using a bayesian approach where we establish priors for 
#the range of all possible values before being observed. The priors will
#mirror a distribution of the shape of a t-distribution with df = 1


urchin.prior.dist = rstanarm::student_t(df = 1)

set.seed(123)
urchin.bayes.mod = 
  linear_reg() %>% 
  set_engine("stan",
             prior_intercept = urchin.prior.dist,
             prior = urchin.prior.dist)

#fit the bayes model
urchin.bayes.fit = 
  urchin.bayes.mod %>% 
  fit(width ~ initial_volume*food_regime,
      data = urchins)

tidy(urchin.bayes.fit, conf.int = T)




urchin.bayes.plot = 
  cbind(urchins.new.points,
        predict(urchin.bayes.fit,
                new_data = urchins.new.points),
        predict(urchin.bayes.fit,
                new_data = urchins.new.points,
                type = "conf_int")) 



ggplot(urchin.bayes.plot,
       aes(x = food_regime))+
    geom_point(aes(y = .pred))+
    geom_errorbar(aes(ymin = .pred_lower,
                      ymax = .pred_upper),
                  width = 0.2)+
    labs(y = "urchin size")+
    ggtitle("Bayesian Model with t(1) Prior of Urchin Width")+
    theme(
      plot.title = element_text(hjust =0.5)
    )

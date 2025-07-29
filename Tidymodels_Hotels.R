library(tidymodels)
library(tidyverse)
library(caret)
library(ranger)
library(mlbench)
library(glmnet)
library(vip)


hotels <- 
  read_csv("https://tidymodels.org/start/case-study/hotels.csv") %>%
  mutate(across(where(is.character), as.factor))


#Build model to predict which hotel stays include children/babies


hotels %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))
  

#Children only .08 of the sample so we stratify the dataset
set.seed(123)
hotel.splits = initial_split(hotels, 
                             strata = children)
hotel.other = training(hotel.splits)
hotel.test = testing(hotel.splits)


#Check to see if both the 'other' and test sets have appropriate
#stratified sets 

#The other data set will be used to create a validation set
#to measure model performance and create a training set
#to fit the model
hotel.other %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))


hotel.test %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))


set.seed(123)
hotel.validation.set = validation_split(hotel.other,
                                        strata = children,
                                        prop = 0.80)









#Penalized Logistic Regression
hotel.logistic.model = 
  logistic_reg(penalty = tune(),
               mixture = 1) %>% 
  set_engine("glmnet")



hotel.holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter", 
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")


hotel.logistic.recipe =
  recipe(children ~., data = hotel.other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date, holidays = hotel.holidays) %>% 
  step_rm(arrival_date) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())


#Create workflow
hotel.logistic.workflow = 
  workflow() %>% 
  add_model(hotel.logistic.model) %>% 
  add_recipe(hotel.logistic.recipe)





#Tune the model
hotel.logistic.grid = expand.grid(penalty = 10^seq(-4,-1,length.out =30))

hotel.logistic.tune.results = 
  hotel.logistic.workflow %>% 
  tune_grid(hotel.validation.set,
            grid = hotel.logistic.grid,
            control = control_grid(save_pred = T),
            metrics = metric_set(roc_auc, accuracy))



#Plot the ROC curve
hotel.logistic.roc.plot = 
  hotel.logistic.tune.results %>% 
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  ggplot(aes(x = penalty, y = mean))+
    geom_point()+
    geom_line()+
    ylab("Area under the ROC")+
    scale_x_log10(labels = scales::label_number())

hotel.logistic.roc.plot


#We see from the plot model performance is better at smaller penalty
#This suggests majority of predictors are important





#Get the top performing model
hotel.logistic.top.models = 
  hotel.logistic.tune.results %>% 
  show_best(metric = "roc_auc",
            n = 15) %>% 
  arrange(penalty)




#Get Best Model
hotel.logistic.best.model = 
  hotel.logistic.tune.results %>% 
  collect_metrics() %>% 
  arrange(penalty) %>% 
  slice(12)


hotel.logistic.best.model


hotel.logistic.best.model.auc = 
  hotel.logistic.tune.results %>% 
  collect_predictions(parameters = hotel.logistic.best.model) %>% 
  roc_curve(children,.pred_children) %>% 
  mutate(model = "Logistic Regression")

autoplot(hotel.logistic.best.model.auc)
#Solid Results but can be improved













# Second Model - Tree Based Ensemble
#We can use parallel processing to use multiple cores
#here we use single validation and thus do not need to
#If used CV for resampling, recommended to use parallel processing
cores = parallel::detectCores()
cores



hotel.rf.model = 
  rand_forest(mtry = tune(),
              min_n = tune(),
              trees = 1000) %>% 
  set_engine("ranger",
             #num.threads is where you can implement parallel processing
             num.threads = cores) %>% 
  set_mode("classification")


#Create RF recipe
#RF doesn't require dummy or normalized predictor variables
hotel.rf.recipe = 
    recipe(children ~.,
           data = hotel.other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date) %>% 
  step_rm(arrival_date)


#Create new RF workflow

hotel.rf.workflow = 
  workflow() %>% 
  add_model(hotel.rf.model) %>% 
  add_recipe(hotel.rf.recipe)




#Tune and train the model
set.seed(123)
hotel.rf.tune = 
  hotel.rf.workflow %>% 
  tune_grid(hotel.validation.set,
            grid = 25,
            control = control_grid(save_pred = T),
            metrics = metric_set(roc_auc, accuracy))



#Top  5 models from the tuning 
#All models here outperform every logistic model
hotel.rf.tune %>% 
  show_best(metric = "roc_auc")


#Shows accuracy/roc_auc for each tuned parameter
autoplot(hotel.rf.tune)


#Get best model
hotel.rf.best = 
  hotel.rf.tune %>% 
  select_best(metric = "roc_auc")

hotel.rf.best



hotel.rf.auc = 
  hotel.rf.tune %>% 
  collect_predictions(parameters = hotel.rf.best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model="Random Forest")

rbind(
  hotel.logistic.best.model.auc,
  hotel.rf.auc
) %>% 
  ggplot(aes(x = 1-specificity,
             y = sensitivity,
             col = model))+
    geom_path(lwd = 1.5, alpha =0.8)+
    geom_abline(lty = 3)+
    coord_equal()+
    scale_color_viridis_d(option = 'plasma',
                          end = 0.6)









#Fit Final model
hotel.final.model = 
  rand_forest(mtry = 6,
              min_n = 8,
              trees = 1000) %>% 
  set_engine('ranger',
             num.threads = cores,
             importance = "impurity") %>% 
  set_mode("classification")

#Update workflow with final model
hotel.final.workflow = 
  hotel.rf.workflow %>% 
  update_model(hotel.final.model)



#last fit
set.seed(123)
hotel.final.rf.fit = 
  hotel.final.workflow %>% 
  last_fit(hotel.splits)


#ROC_AUC value is pretty close to what we saw when we 
#tuned the random forest model with the validation set
#good news meaning our training estimate wasn't far off

hotel.final.rf.fit %>% 
  collect_metrics()




#Get variable importance
hotel.final.rf.fit %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 20)




#Generate last ROC curve 
hotel.final.rf.fit %>% 
  collect_predictions() %>% 
  roc_curve(children,
            .pred_children) %>% 
  autoplot()

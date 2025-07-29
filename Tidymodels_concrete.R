library(tidymodels)
library(tidyverse)
library(caret)
library(ranger)
library(mlbench)
library(glmnet)
library(vip)
library(broom.mixed)
library(dotwhisker) 
library(janitor)
library(skimr)
library(kknn)
library(xgboost)

concrete = modeldata::concrete

skim(concrete)

#No missing values 
#1030 rows, 9 columns (all numeric)

#compressive_strength is the outcome variable 



concrete = 
  concrete %>% 
  group_by(across(-compressive_strength)) %>% 
  summarise(compressive_strength = mean(compressive_strength),
            .groups = "drop")


nrow(concrete)




#Split the data into training and testing
set.seed(123)
concrete.split = initial_split(concrete, strata = compressive_strength)
concrete.train = training(concrete.split)
concrete.test = testing(concrete.split)

set.seed(123)
concrete.folds = 
  vfold_cv(
    concrete.train,
    strata = compressive_strength,
    repeats = 5
  )




#Create two recipes - one centered and scaled in preprocessing
#the other using quadratic interactions 

concrete.normalized.rec = 
  recipe(compressive_strength ~ ., data = concrete.train) %>% 
  step_normalize(all_predictors()) 

concrete.poly.recipe = 
  concrete.normalized.rec %>% 
  step_poly(all_predictors()) %>% 
  step_interact(~ all_predictors():all_predictors())
  


 concrete.linear = 
  linear_reg(penalty = tune(),
             mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("regression")


concrete.rf = 
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")


concrete.xgb = 
  boost_tree(
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune(),
    min_n = tune(),
    sample_size = tune(),
    trees = tune()
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")


concrete.knn = 
  nearest_neighbor(
    neighbors = tune(),
    dist_power = tune(),
    weight_func = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")












#Workflows 
concrete.rf.workflow = 
  workflow() %>% 
  add_model(concrete.rf) %>% 
  add_recipe(concrete.normalized.rec)
  



concrete.linear.workflow = 
  workflow() %>% 
  add_model(concrete.linear) %>% 
  add_recipe(concrete.poly.recipe)


concrete.knn.workflow =
  workflow() %>% 
  add_model(concrete.knn) %>% 
  add_recipe(concrete.normalized.rec)



concrete.xgb.workflow = 
  workflow() %>% 
  add_model(concrete.xgb) %>% 
  add_recipe(concrete.normalized.rec)






### Random Forest #####



#Tuning and evaluating the models

concrete.tune.rf.results = 
  concrete.rf.workflow %>% 
  tune_grid(
   resamples = concrete.folds,
   grid = 25,
   metrics = metric_set(rmse),
   control = control_grid(save_pred = T)
   
 )


concrete.tune.rf.results %>% 
  show_best(metric = "rmse",
            n = 15)


#Best rf model is from an mtry of 6 and min_n of 3 which gives a 5.01 rmse

concrete.rf.top.models = 
  concrete.tune.rf.results %>% 
  show_best(metric = "rmse",
            n = 15)

concrete.rf.top.models


#Get Best Model
concrete.rf.best.model =
  concrete.tune.rf.results %>% 
  slice(1)









##### Linear ######


concrete.tune.linear.results = 
  concrete.linear.workflow %>% 
  tune_grid(
    resamples = concrete.folds,
    grid = 25,
    metrics = metric_set(rmse),
    control = control_grid(save_pred = T)
    
  )





concrete.linear.top.models = 
  concrete.tune.linear.results %>% 
  show_best(metric = "rmse",
            n = 15)

concrete.linear.top.models


#Best linear model is when penalty is .0133 and mixture is 0.524


concrete.linear.best.model =
  concrete.linear.top.models %>% 
  slice(1)








##### knn ### 

#Tuning and evaluating the models

concrete.tune.knn.results = 
  concrete.knn.workflow %>% 
  tune_grid(
    resamples = concrete.folds,
    grid = 25,
    metrics = metric_set(rmse),
    control = control_grid(save_pred = T)
    
  )



concrete.knn.top.models = 
  concrete.tune.knn.results %>% 
  show_best(metric = "rmse",
            n = 15)

concrete.knn.top.models


#Get Best Model
concrete.knn.best.model =
  concrete.tune.knn.results %>% 
  slice(1)










#### XGBoost ####


concrete.tune.xgb.results = 
  concrete.xgb.workflow %>% 
  tune_grid(
    resamples = concrete.folds,
    grid = 25,
    metrics = metric_set(rmse),
    control = control_grid(save_pred = T)
    
  )





concrete.xgb.top.models = 
  concrete.tune.xgb.results %>% 
  show_best(metric = "rmse",
            n = 15)

concrete.xgb.top.models


#Best xgboost model and best overall has rmse of 3.91


concrete.xgb.best.model =
  concrete.xgb.top.models %>% 
  head(1)















#Make final models and final workflows with tuned parameters

concrete.xgb.top.models

#Final XGBoost
concrete.final.xgb.model = 
  boost_tree(
    tree_depth = 5,
    learn_rate = .0294,
    loss_reduction = 1.11e-8,
    min_n = 7,
    sample_size = .657,
    trees = 1483
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")


concrete.linear.top.models


concrete.final.linear.model = 
  linear_reg(
    penalty = 1.33e-2,
    mixture = 0.524
  ) %>% 
  set_engine("glmnet") %>% 
  set_mode("regression")
  
  

  
concrete.final.rf.model = 
  rand_forest(
    mtry = 6,
    min_n = 3,
    trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")
  

  
  
  
concrete.final.knn.model = 
  nearest_neighbor(
    neighbors = 15,
    dist_power = 1.95,
    weight_func = "inv") %>% 
  set_engine("kknn") %>% 
  set_mode("regression")






concrete.final.linear.workflow. = 
  workflow() %>% 
  add_model(concrete.final.linear.model) %>% 
  add_recipe(concrete.poly.recipe)


concrete.final.knn.workflow = 
  workflow() %>% 
  add_model(concrete.final.knn.model) %>% 
  add_recipe(concrete.normalized.rec)

concrete.final.xgb.workflow = 
  workflow() %>% 
  add_model(concrete.final.xgb.model) %>% 
  add_recipe(concrete.normalized.rec)

concrete.final.rf.workflow = 
  workflow() %>% 
  add_model(concrete.final.rf.model) %>% 
  add_recipe(concrete.normalized.rec)


#Fit the model

set.seed(123)
concrete.final.linear.fit = 
  concrete.final.linear.workflow %>% 
  last_fit(concrete.split)


set.seed(123)
concrete.final.rf.fit = 
  concrete.final.rf.workflow %>% 
  last_fit(concrete.split)


set.seed(123)
concrete.final.knn.fit = 
  concrete.final.knn.workflow %>% 
  last_fit(concrete.split)


set.seed(123)
concrete.final.xgb.fit = 
  concrete.final.xgb.workflow %>% 
  last_fit(concrete.split)



#fit metrics
concrete.final.xgb.metrics =  
  concrete.final.xgb.fit %>% 
  collect_metrics() %>% 
  mutate(model = "xgb")

concrete.final.linear.metrics =  
  concrete.final.linear.fit %>% 
  collect_metrics() %>% 
  mutate(model = "linear")

concrete.final.rf.metrics = 
  concrete.final.rf.fit %>% 
  collect_metrics() %>% 
  mutate(model = "rf")

concrete.final.knn.metrics =
  concrete.final.knn.fit %>% 
  collect_metrics() %>% 
  mutate(model = "knn")




concrete.all.model.metrics = 
  rbind(concrete.final.xgb.fit,
        concrete.final.linear.fit,
        concrete.final.rf.fit,
        concrete.final.knn.fit)

#Extract the metrics for each model

concrete.xgb.fit.metrics =   
  concrete.all.model.metrics$.metrics[[1]] %>% 
  mutate(model = "xgb")

concrete.linear.fit.metrics =
  concrete.all.model.metrics$.metrics[[2]] %>% 
  mutate(model = "linear")

concrete.rf.fit.metrics =
  concrete.all.model.metrics$.metrics[[3]] %>% 
  mutate(model = "rf")

concrete.knn.fit.metrics = 
  concrete.all.model.metrics$.metrics[[4]] %>% 
  mutate(model = "knn")

concrete.all.rmse = 
  rbind(
    concrete.xgb.fit.metrics,
    concrete.linear.fit.metrics,
    concrete.rf.fit.metrics,
    concrete.knn.fit.metrics
  )

concrete.all.rmse %>% 
  filter(.metric == "rmse") %>% 
  ggplot(aes(x = model,
             y = .estimate,
             group = model,
             fill = model))+
    geom_col()+
    ggtitle("RMSE Of Models")


concrete.all.rmse %>% 
  filter(.metric == "rsq") %>% 
  ggplot(aes(x = model,
             y = .estimate,
             group = model,
             fill = model))+
  geom_col()+
  ggtitle("RSQ Of Models")

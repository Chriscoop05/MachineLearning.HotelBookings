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


titanic.survived =  read_csv("C:/Users/chris/Downloads/titanic/gender_submission.csv")
titanic.train = read_csv("C:/Users/chris/Downloads/titanic/train.csv")
titanic.test = read_csv("C:/Users/chris/Downloads/titanic/test.csv")


titanic.test = 
  titanic.test %>% 
  left_join(titanic.survived, 
             by = "PassengerId") 


titanic.test = titanic.test %>% 
  select(PassengerId,Survived,Pclass:Embarked)



skim(titanic.train)
#891 unique names
#Cabin has 687 na values
#Embarked has 2 na values
#Age has 177 missing values

titanic.train = titanic.train %>% 
  clean_names() %>% 
  mutate(survived = as.factor(survived))








#Pre Processing
titanic.recipe = 
  titanic.train %>% 
  recipe(survived~.) %>% 
  step_select(
    -c(
      name,
      ticket,
      passenger_id
    )
  ) %>% 
  #Exclude all unique values
  step_mutate(
    cabin = as.factor(
      str_sub(
        cabin,
        start = 1,
        end = 1
      )
    )
  ) %>%  
  step_impute_median(
    all_numeric_predictors()
  ) %>% 
  step_normalize(
    all_numeric_predictors()
  ) %>% 
  step_corr(
    all_numeric_predictors()
  ) %>% 
  step_unknown(
    all_nominal_predictors()
  ) %>% 
  step_nzv(
    all_numeric_predictors()
  ) %>% 
  step_dummy(
    all_nominal_predictors()
  )





#Create model
titanic.rf.model = 
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = 1000
) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")



#Create workflow

titanic.rf.workflow =
  workflow() %>% 
  add_recipe(titanic.recipe) %>% 
  add_model(titanic.rf.model)




#Create folds for resampling
set.seed(123)
titanic.split.folds = 
  initial_split(titanic.train,
                strata = all.vars(titanic.recipe))



set.seed(123)
titanic.folds = vfold_cv(
  training(titanic.split.folds),
  v = 10,
  strata = all.vars(titanic.recipe)
)



#Tune and train
titanic.rf.tune = 
  titanic.rf.workflow %>% 
  tune_grid(
    resamples = titanic.folds,
    grid = 15,
    control = control_grid(save_pred = T),
    metrics = metric_set(roc_auc,
                         accuracy)
    
  )










autoplot(titanic.rf.tune)
#You can see model is optimized at around mtry = 15,min_n = 32




#Select the best model based off roc_auc

titanic.rf.best = 
  titanic.rf.tune %>% 
  select_best(metric = "roc_auc")


#Best model mtry = 15, min_n = 34
titanic.rf.best



titanic.rf.auc = 
  titanic.rf.tune %>% 
  collect_predictions(parameters = titanic.rf.best) %>% 
  roc_curve(survived, .pred_0) %>% 
  mutate(model = "Random Forest")


autoplot(titanic.rf.auc)


#Finalize and Fit the final model
titanic.rf.final.model= 
  rand_forest(
    mtry = 15,
    min_n = 34
  ) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")


titanic.final.rf.workflow =
  titanic.rf.workflow %>% 
  update_model(titanic.rf.final.model)
           
  
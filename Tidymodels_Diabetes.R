library(tidymodels)
library(tidyverse)
library(caret)
library(ranger)
library(mlbench)

data("PimaIndiansDiabetes")

df.diabetes = PimaIndiansDiabetes

#Amount of 0's in each column of the dataset
colSums(df.diabetes == 0, na.rm =T)

#Distribution of values in the datasets
hist_triceps = ggplot(df.diabetes,aes(x = triceps))+
    geom_histogram()

hist_insulin = ggplot(df.diabetes,aes(x = insulin))+
    geom_histogram()

hist_pressure = ggplot(df.diabetes,aes(x = pressure))+
  geom_histogram()

hist_glucose = ggplot(df.diabetes,aes(x = glucose))+
  geom_histogram()

hist_mass = ggplot(df.diabetes,aes(x = mass))+
  geom_histogram()

gridExtra::grid.arrange(hist_triceps,
                        hist_insulin,
                        hist_pressure,
                        hist_glucose,
                        hist_mass)




#Change all 0's in the select columns to NA
df.clean.diabetes = df.diabetes %>% 
  mutate_at(vars(triceps,glucose,pressure,insulin,mass),
            function(.var){
              if_else((.var == 0),
                      as.numeric(NA),
                      .var)
            })

#Sum na's in each column to check 
colSums(is.na(df.clean.diabetes))







#Split the data into train/test
set.seed(123)
df.split.diabetes = initial_split(df.clean.diabetes,
                               prop= .75)

df.train.diabetes = training(df.split.diabetes)
df.test.diabetes = testing(df.split.diabetes)


#Create the cross validation folds right now
df.cv.diabetes = vfold_cv(df.train.diabetes)







#Create a recipe
df.recipe.diabetes = 
  recipe(diabetes ~ pregnant + glucose + pressure+
                    triceps + insulin + mass +pedigree +
                    age,
         data = df.train.diabetes) %>% 
  step_normalize(all_numeric()) %>% 
  step_impute_knn(all_predictors())






#Random Forest Model
df.diabetes.rf.model = 
  rand_forest() %>% 
  set_args(mtry = tune()) %>% 
  set_engine("ranger", importance ="impurity") %>% 
  set_mode("classification")


#Logistic Regression
df.diabetes.logistic.model =
  logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")





#Put it together in a workflow
df.diabetes.workflow = 
  workflow() %>% 
  add_recipe(df.recipe.diabetes) %>% 
  add_model(df.diabetes.rf.model)


##Tune parameters

#Specify which mtry values to  try
df.diabaetes.rf.grid = expand.grid(mtry = c(3,4,5))

#Extract Results
df.diabetes.rf.tune.results = df.diabetes.workflow %>% 
  tune_grid(
    resamples = df.cv.diabetes,
    grid = df.diabaetes.rf.grid,
    metrics = metric_set(accuracy, roc_auc)
  )


#Collect the results of the tune
df.diabetes.rf.tune.metrics = df.diabetes.rf.tune.results %>% 
  collect_metrics()

ggplot(df.diabetes.rf.tune.metrics,
       aes(x = mtry, y = mean,
           group = .metric, color = .metric))+
    geom_line()+
    geom_point()+
    theme_classic()+
    ggtitle("Tuned Model Performance Metrics")+
    ggrepel::geom_label_repel(data= df.diabetes.rf.tune.metrics,
                              aes(label = round(mean,4)),
                              box.padding = 0,
                              point.padding = 0,
                              nudge_y = 0.005)

#We see that mtry = 3 yields the best roc_auc and accuracy







#Finalize the workflow with the tuned parameters
df.diabetes.param.final = df.diabetes.rf.tune.results %>% 
  select_best(metric = 'accuracy')
df.diabetes.param.final



df.diabetes.workflow = df.diabetes.workflow %>% 
  finalize_workflow(df.diabetes.param.final)




#Evaluate the model on the test set
df.diabetes.rf.fit = df.diabetes.workflow %>% 
  last_fit(df.split.diabetes)

df.test.diabetes.performance = df.diabetes.rf.fit %>% 
  collect_metrics() %>% 
  mutate(model_type = "rf")




#Generate predictions
df.test.diabetes.predictions = df.diabetes.rf.fit %>% 
  collect_predictions

df.test.diabetes.predictions %>% 
  conf_mat(truth = diabetes,
           estimate = .pred_class)


#Distributions of predicted probability 
df.test.diabetes.predictions %>% 
  ggplot()+
  geom_density(aes(x = .pred_pos, fill = diabetes),
               alpha = 0.25)

df.test.predictions = df.diabetes.rf.fit %>% 
  pull(.predictions)
df.test.predictions





df.diabetes.final.model = fit(df.diabetes.workflow,
                              df.clean.diabetes)

df.diabetes.final.model



df.test.diabetes.predictions








#Variable Importance
df.ranger_obj_diabetes = pull_workflow_fit(df.diabetes.final.model)$fit

df.var.importance.diabetes = df.ranger_obj_diabetes$variable.importance

#Uncomment to transform var importance data into dataframe
#df.var.importance.diabetes = as.data.frame(df.var.importance.diabetes) %>% 
#  mutate(variables = c("pregnant",'glucose',
#                       "pressure", "triceps",
#                       "insulin", "mass",
#                       "pedigree", "age")) %>% 
#  rename("values" = "df.var.importance.diabetes")


ggplot(df.var.importance.diabetes, aes(x = variables, y = values))+
    geom_col()+
    theme(axis.text.x = element_text(angle = 330),
          panel.background = element_rect(fill = 'snow2'),
          panel.grid = element_blank(),
          plot.title = element_text(hjust = 0.5,
                                    size = 14,
                                    color = 'black',
                                    face = 'bold'))+
    ggtitle("Random Forest Classification Variable Importance")






########################
#Make a new logistic workflow for the logistic classification model to the workflow now
df.logistic.diabetes.workflow = workflow() %>% 
  add_model(df.diabetes.logistic.model) %>% 
  add_recipe(df.recipe.diabetes)


df.logistic.diabetes.fit = df.logistic.diabetes.workflow %>% 
  last_fit(df.split.diabetes)


df.logistic.test.performance = df.logistic.diabetes.fit %>% 
  collect_metrics() %>% 
  mutate(model_type = "logistic")

df.diabetes.all.performance = rbind(df.logistic.test.performance,
      df.test.diabetes.performance)

ggplot(df.diabetes.all.performance,
       aes(x = .metric, y = .estimate,
           group = model_type, fill = model_type))+
    geom_col(position = position_dodge(0.9))+
    ggtitle("Model Performance")+
    xlab("Metrics")+
    ylab("Estimate")+
    theme(plot.title = element_text(hjust = 0.5),
          plot.background = element_rect(fill ='snow2'),
          panel.background = element_rect(fill = 'snow4'),
          panel.grid = element_blank())+
    scale_fill_brewer(palette = "RdBu")

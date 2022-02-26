#------------
# Author: Hector Cozar Gordo
# Data: 2022_02_18
# Purpose: Introduce the two logic variables with 5% NAs.
# Input: Train, test dataset.
# Output: Submissions.
# Iteration: Numeric + Categorical (<1000) + grid search
#------------

#---- Libraries
library(dplyr)       # Data Manipulation.
library(data.table)  # Fast Data Manipulation.
library(inspectdf)   # EDAs automatic
library(ranger)      # fast randomForest
library(tictoc)      # Measure execution time
library(magrittr)    # Piping mode
library(ggplot2)     # Very nice charts 
library(forcats)     # Manage factor variables
library(tibble)      # Compact dataframe
# library(tidytable)
# library(poorman)

#---- Data Loading.
datTrainOri    <- as.data.frame(fread("./data/train.csv", nThread = 3))
datTrainOrilab <- fread("./data/train_labels.csv", nThread = 3, data.table = FALSE )
datTestOri     <- fread("./data/test.csv", nThread = 3, data.table = FALSE)
names(datTrainOrilab) 
names(datTrainOri)
datTrainOrilab %>% head(n = 10)

# #---- Confirm "target" distribution
# datTrainOrilab %>%
#   count(status_group) %>%
#   arrange(-n)

#              status_group     n
# 1              functional 32259
# 2          non functional 22824
# 3 functional needs repair  4317

#-- Dplyr - another way
datTrainOritarget <- datTrainOri  %>%
  left_join(datTrainOrilab)

#----- We use just NUMERICAL columns
#-- Now what I have is: train + target. Ready to model...
datTrainOritargetnum <- 
  datTrainOritarget %>%
  select(where(is.numeric), status_group) %>%
  mutate(status_group = as.factor(status_group))

#----- Add new variables....
# Veamos cardinalidad de cada variable categorica.
# datTrainOri %>%
#   select(where(is.character)) %>%
#   count()
#-- Function to get number of levels
myfun <- function(x) {
  num_levels <- length(unique(x))
  return(num_levels)
}
#-- Just character columns
datTrainOrichar <- datTrainOri %>%
  select(where(is.character)) 

#-- Count levels in character columns.
levels_val <- as.data.frame(apply(datTrainOrichar, 2, myfun))
names(levels_val)[1] <- "num_levels"
levels_val$vars <- rownames(levels_val)
rownames(levels_val) <- NULL
levels_val %<>% arrange(num_levels)

#-- We will use levels up to 125 (lga)
lev_god <- levels_val %>%
  filter( num_levels <= 1000) %>%
  #-- This filter is to remove "recorded_by"
  filter( num_levels > 1) %>%
  #-- This selects just var column
  select(vars) %>%
  #-- Convert to vector
  pull(vars)

# Dataframe with just the columns I want.
# datTrainOrigod <- datTrainOri[ , lev_god ]

#-- With dplyr.
datTrainOrichargod <- datTrainOri %>%
  # select(!!sym(lev_god))
  select(lev_god)

#-- Now put together numeric + the dataframe with the character columns.
#-- And locate target (status_group) at the end of the dataframe.
datTrainOrinumchargod <- cbind(datTrainOritargetnum, datTrainOrichargod) %>%
  relocate( status_group, .after = lga)

#----- To include Logic variables : permit / public_meeting
# # Barplot of column types
# x <- inspect_types(datTrainOri)
# show_plot(x)
#Number NAs Test
sum(is.na(datTestOri$permit))*100/nrow(datTestOri)
sum(is.na(datTestOri$public_meeting))*100/nrow(datTestOri)
#1-- Pongo junto train y test (original) 
datAll <- rbind(datTrainOri, datTestOri)
#2-- Selecciono las logicas
datLogic <- datAll %>%
  select(where(is.logical)) %>%
  mutate(across(where(is.logical), as.numeric))
#-- Selecciono las 31 variables que tengo hasta ahora
col_gd <- names(datTrainOrinumchargod)
col_gd <-  col_gd[ col_gd != "status_group"]
datGd <- datAll[, col_gd]
datGd <- cbind(datGd, datLogic)
#--- Apply library/algorithm to impute NAs.
library(missRanger)
datGdImputed <- missRanger(datGd, pmm.k = 3, num.trees = 100)
#-- Separo train / test imputados
datGdImpTrain <- datGdImputed[1:nrow(datTrainOri), ]
datGdImpTest  <- datGdImputed[(nrow(datTrainOri) + 1):nrow(datGdImputed),]
#-- Manera alternativa
datGdImputed %<>%
  mutate(miindice = 1:nrow(datGdImputed))

#-- Train
datGdTrain <- datGdImputed %>%
               filter(miindice <= nrow(datTrainOri)) %>%
               select(-miindice)
#-- Test
datGdTest <- datGdImputed %>%
               filter(miindice > nrow(datTrainOri)) %>% 
               select(-miindice)
datGdTrain %<>%              
  mutate( date_recorded = datTrainOri$date_recorded) %>%
  mutate( fe_antiguedad = max(date_recorded) - date_recorded) %>%
  mutate( fe_distancia = sqrt(longitude^2 + latitude^2)) %>%
  select(-date_recorded)

datGdTest %<>%              
  mutate( date_recorded = datTestOri$date_recorded) %>%
  mutate( fe_antiguedad = max(date_recorded) - date_recorded) %>%
  mutate( fe_distancia = sqrt(longitude^2 + latitude^2)) %>%
  select(-date_recorded)
  
  
#-- Pego status_group al nuevo Train..               
datGdTrain$status_group <- as.factor(datTrainOrilab$status_group)               



#---------------- MODEL --------------
#--- Apply an algorithm - randomForest -> ranger!.
#--  In ranger, "target" must be a factor!.
#--- Tune model with gridsearch
mytrain <- datGdTrain

#---- Get the best model from the gridsearch
mymodel <- ranger(
  status_group ~ .,
  data       = mytrain,
  num.trees  = 500,
  mtry       = 6,
  # To get variable importance
  importance = "impurity" 
)

#--- Estimated model error.
error_val <- mymodel$prediction.error
error_val
accu_val <- 1 - error_val
accu_val

#--- Variable importance
varImp <- mymodel$variable.importance %>%
  as.data.frame()
varImp   %<>%
  mutate( vars = rownames(varImp)) %>%
  arrange(-.)

rownames(varImp) <- NULL
names(varImp)[1] <- "Importance"
varImp %<>%
  arrange(-Importance)

#--- Trampa
# varImp$Importance[1] <- 20000

#--- Chart Importance
ggplot(varImp, aes(x = fct_reorder(vars, Importance), y = Importance, alpha = Importance)) +
  geom_col( fill = "darkred") +
  coord_flip() +
  labs(
    title = "Relative Variable Importance",
    subtitle = paste("Accuracy: ", round(100*accu_val,2), "%", sep = ""), 
    x = "Variables",
    y = "Relative Importance",
    caption = paste("model num vars: ", ncol(mytrain) ,sep = "") 
  ) +
  theme_bw()
ggsave(paste("./charts/Distancia_Antiguedad_Logical_Variable_Importance_", ncol(mytrain), ".png", sep = ""))

# # Highligting most important one.
# library(ggcharts)
# bar_chart(
#   varImp,
#   vars,
#   Importance,
#   top_n = 10,
#   highlight = "longitude"
# )

#--- Confusion Matrix.
#-- Son las predicciones sobre el conjunto de train vs. los valores reales.
table(mymodel$predictions, datTrainOrinumchargod$status_group)
mymodel$confusion.matrix

#----- Submission 
#-- Prediction
pred_val <- predict( mymodel, data = datGdTest)$predictions
head(pred_val)

#-- Prepare submission
sub_df <- data.frame(
  id = datTestOri$id,
  status_group = pred_val
)
#-- Save submission
fwrite(sub_df, "./submissions/ranger_vars_35_acc_081_.csv", nThread = 3 )

#---- END OF FILE ------------
#-- Results:
# 00 - ranger - 11 vars numeric - 71% local -> 0.7140 plataforma
# 01 - ranger - 31 vars num/cat 1000 - 81% local -> 0.8160 plataforma
# 02 - ranger - 35 vars num/cat 10000 - 81.60% local -> 0.8129 plataforma
# 03 - ranger - 31 vars num/cat 1000 - ntrees: 500 - mtry = 6 - 0.8127 local -> 0.8164 plataforma
# 04 - ranger - 33 vars num/cat/logic 1000 - ntrees: 500 - mtry = 6 - 0.8130 local -> 0.8193 plataforma
# 04 - ranger - 34 vars num/cat/logic/antiguedad 1000 - ntrees: 500 - mtry = 6 - 0.8151 local -> 0.8197 plataforma
# 04 - ranger - 35 vars num/cat/logic/antiguedad/distancia 1000 - ntrees: 500 - mtry = 6 - 0.8163 local -> 0.8206 plataforma

# datTrainOri %>%
#   select(where(is.character)) %>%
#   summarize(lev = n())

lev_res <- datTrainOri %>%
  select(where(is.character)) %>%
  mutate(across( funder:waterpoint_type_group, myfun)) %>%
  unique() %>%
  as.data.frame() %>%
  t() %>%
  as.data.frame() %>%
  rownames_to_column( var = "vars") %>%
  rename( res = 2) %>%
  # remove_rownames() %>%
  arrange(res)



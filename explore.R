rm(list=ls())
library(data.table)

# load datasets 
air_reserve <- fread('data/air_reserve.csv')
air_store_info <- fread('data/air_store_info.csv')
air_visit_data <- fread('data/air_visit_data.csv')
date_info <- fread('data/date_info.csv')
hpg_reserve <- fread('data/hpg_reserve.csv')
hpg_store_info <- fread('data/hpg_store_info.csv')
store_id_relation <- fread('data/store_id_relation.csv')
sample_submission <- fread('data/sample_submission.csv')

# look into the structure
str(air_reserve)
str(air_store_info)
str(air_visit_data)
str(date_info)
str(hpg_reserve)
str(hpg_store_info)
str(store_id_relation)
str(sample_submission)

# Is there any NA fields?
colSums(is.na(air_reserve))
colSums(is.na(air_store_info))
colSums(is.na(air_visit_data))
colSums(is.na(date_info))
colSums(is.na(hpg_reserve))
colSums(is.na(hpg_store_info))
colSums(is.na(store_id_relation))

# summary
summary(air_reserve)
summary(air_store_info)
summary(air_visit_data)
summary(date_info)
summary(hpg_reserve)
summary(hpg_store_info)
summary(store_id_relation)
summary(sample_submission)
        
# TODO: check imbalancement
# TODO: find correlation between attributes


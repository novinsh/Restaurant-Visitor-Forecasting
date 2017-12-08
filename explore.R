rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

#### load datasets ####
air_reserve <- fread('data/air_reserve.csv')
air_store_info <- fread('data/air_store_info.csv')
air_visit_data <- fread('data/air_visit_data.csv')
date_info <- fread('data/date_info.csv')
hpg_reserve <- fread('data/hpg_reserve.csv')
hpg_store_info <- fread('data/hpg_store_info.csv')
store_id_relation <- fread('data/store_id_relation.csv')
sample_submission <- fread('data/sample_submission.csv')

#### look into the structure ####
str(air_reserve)
str(air_store_info)
str(air_visit_data)
str(date_info)
str(hpg_reserve)
str(hpg_store_info)
str(store_id_relation)
str(sample_submission)

#### Is there any NA fields? ####
colSums(is.na(air_reserve))
colSums(is.na(air_store_info))
colSums(is.na(air_visit_data))
colSums(is.na(date_info))
colSums(is.na(hpg_reserve))
colSums(is.na(hpg_store_info))
colSums(is.na(store_id_relation))

#### match number of restaurant records in each table ####
# according to the datasets document each dataset has a key value
# which is store/restaurant id. find porportion of each store records
# in each data set.

air_stores <- air_reserve$air_store_id
air_stores_u <- unique(air_stores)
hpg_stores <- hpg_reserve$hpg_store_id
hpg_stores_u <- unique(hpg_stores)
length(air_stores)
length(air_stores_u)
length(hpg_stores)
length(hpg_stores_u)

# air store ids in store_info
table(air_stores %in% air_store_info$air_store_id)["TRUE"]
table(air_stores_u %in% air_store_info$air_store_id)["TRUE"]
table(air_stores %in% hpg_store_info$hpg_store_id)["TRUE"]
table(air_stores_u %in% hpg_store_info$hpg_store_id)["TRUE"]

# air store ids in air visit_data
table(air_stores %in% air_visit_data$air_store_id)["TRUE"]
table(air_stores_u %in% air_visit_data$air_store_id)["TRUE"]

# air store ids in store_id_relation
table(air_stores %in% store_id_relation$air_store_id)["TRUE"]
table(air_stores_u %in% store_id_relation$air_store_id)["TRUE"]

# if possible: convert id of a hpg to its corresponding air 
hpg_to_air <- function(hpg_id) {
  return (store_id_relation$air_store_id[hpg_id == store_id_relation$hpg_store_id])
}

# if possible: convert id of an air to its corresponding hpg
ait_to_hpg <- function(air_id) {
  return (store_id_relation$hpg_store_id[air_id == store_id_relation$air_store_id])
}

hpg_in_air <- unlist(lapply(hpg_stores_u, hpg_to_air))

# proportion of unique hpg stores to air 
(length(hpg_in_air)/length(hpg_stores_u)) * 100

length(hpg_in_air)
table(hpg_in_air %in% air_reserve$air_store_id)["TRUE"]
table(hpg_in_air %in% air_store_info$air_store_id)["TRUE"]
table(hpg_in_air %in% air_visit_data$air_store_id)["TRUE"]

#### air reserve ####
# convert datetime fields
air_reserve$visit_datetime = as.Date(air_reserve$visit_datetime)
air_reserve$reserve_datetime = as.Date(air_reserve$reserve_datetime)

# number of restaurants
length(unique(air_reserve$air_store_id))

# frequency table of the restaurants records and num of visitors
data.frame(table(air_reserve$reserve_visitors))
data.frame(table(t$Freq)) # freq of freq
data.frame(table(air_reserve$air_store_id))

t <- data.frame(table(air_reserve$air_store_id))
# distribution of each attribute
ggplot(t, aes(x=reorder(as.numeric(t$Var1), t$Freq), y=t$Freq)) + 
  geom_bar(stat="identity") +
  theme(axis.text.x=element_blank()) +
  xlab("Restaurant ID") +
  ylab("Number of records") +
  ggtitle("Distribution of Restaurants")

hist(air_reserve$visit_datetime, "weeks", format = "%d %b")
hist(air_reserve$reserve_datetime, "weeks", format = "%d %b")
# hist(air_reserve$reserve_visitors)
qplot(air_reserve$reserve_visitors,
      geom="histogram",
      binwidth = 1,  
      main = "Histogram of number of visitors", 
      xlab = "reserve_visitors",
      fill=I("blue"))

summary(air_reserve)

#### air_store_info ####
summary(air_store_info)

#### air_visit_data ####
# convert datetime fields
air_visit_data$visit_date = as.Date(air_visit_data$visit_date)
hist(air_visit_data$visit_date, "weeks", format = "%d %b")
hist(air_visit_data$visitors)
summary(air_visit_data)

#### date_info ####
date_info$calendar_date = as.Date(date_info$calendar_date)
date_info$day_of_week = factor(date_info$day_of_week)
date_info$holiday_flg = factor(date_info$holiday_flg)
hist(date_info$calendar_date, "weeks", format = "%d %b")
hist(as.numeric(date_info$day_of_week))
hist(as.numeric(date_info$holiday_flg)-1)
summary(date_info)

#### hpg_reserve ####
summary(hpg_reserve)

#### hpg_store_info ####
summary(hpg_store_info)

#### store_id_relation ####
summary(store_id_relation)

#### all summary ####
summary(air_reserve)
summary(air_store_info)
summary(air_visit_data)
summary(date_info)
summary(hpg_reserve)
summary(hpg_store_info)
summary(store_id_relation)
summary(sample_submission)

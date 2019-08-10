'data/manhattan_Train.csv'
here::here('data', 'manhattan_Train.csv')

man_data <- readr::read_csv(
    here::here('data', 'manhattan_Train.csv')
)

View(man_data)

library(rsample)

set.seed(2019)
man_split <- initial_split(
    data=man_data,
    prop=0.8,
    strata='TotalValue'
)
man_split
class(man_split)
str(man_split, max.level=1)

man_train <- training(man_split)
man_test <- testing(man_split)

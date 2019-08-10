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

library(recipes)


base_formula <- TotalValue ~ FireService + 
    ZoneDist1 + ZoneDist2 + 
    Class + OwnerType + 
    LotArea + BldgArea + ComArea + ResArea + 
    OfficeArea + RetailArea + 
    NumFloors + UnitsTotal + UnitsRes

mod1 <- lm(base_formula, data=man_train)
mod1
summary(mod1)

library(coefplot)
coefplot(mod1, sort='magnitude')
coefplot(mod1, sort='magnitude', lwdOuter=0.5)

favecolors <- c('blue', 'red', 'blue', 'green', 'blue', 
                'red', 'blue', 'green', 'red', 
                'blue', 'blue', 'green', 'red', 'blue',
                'yellow', 'blue', 'yellow', 'yellow')
favecolors

model.matrix( ~ favecolors)

base_recipe <- recipe(base_formula, data=man_train)
base_recipe

man_recipe <- base_recipe %>% 
    step_zv(all_predictors()) %>% 
    step_knnimpute(everything(), neighbors=5) %>% 
    step_log(TotalValue, base=2) %>% 
    # step_normalize(all_numeric(), -TotalValue) %>% 
    step_center(all_numeric(), -TotalValue) %>% 
    step_scale(all_numeric(), -TotalValue) %>% 
    step_other(all_nominal(), threshold=0.2) %>% 
    step_dummy(all_nominal(), one_hot=TRUE)
man_recipe


man_prepped <- man_recipe %>% 
    prep()
man_prepped

man_x_train <- man_prepped %>% 
    bake(all_predictors(), 
         new_data=man_train,
         composition='dgCMatrix')
head(man_x_train)

man_y_train <- man_prepped %>% 
    bake(all_outcomes(),
         new_data=man_train,
         composition='matrix')
head(man_y_train)

man_x_test <- man_prepped %>% 
    bake(all_predictors(),
         new_data=man_test,
         composition='matrix')
man_y_test <- man_prepped %>% 
    bake(all_outcomes(),
         new_data=man_test,
         composition='matrix')

library(glmnet)

mod2 <- glmnet(x=man_x_train, y=man_y_train,
               family='gaussian', alpha=1,
               standardize=FALSE)
plot(mod2, xvar='lambda')
plot(mod2, xvar='lambda', label=TRUE)

coefpath(mod2)

mod3 <- cv.glmnet(x=man_x_train, y=man_y_train,
                  family='gaussian', alpha=1,
                  standardize=FALSE,
                  nfolds=5)
plot(mod3)
mod3$lambda.min
mod3$lambda.1se

coefpath(mod3)
coefplot(mod3, sort='magnitude', lambda='lambda.min',
         intercept=FALSE)
coefplot(mod3, sort='magnitude', lambda='lambda.1se',
         intercept=FALSE)
coefplot(mod3, sort='magnitude', lambda='lambda.1se',
         intercept=FALSE) + 
    xlim(-0.01, 0.01)


mod4 <- cv.glmnet(x=man_x_train, y=man_y_train,
                  family='gaussian', alpha=0,
                  standardize=FALSE,
                  nfolds=5)

coefpath(mod4)

sd(man_train$BldgArea)
coefplot(mod3, sort='magnitude', lambda='lambda.1se',
         intercept=FALSE, plot=FALSE)
sd(man_train$BldgArea)
2^.43
sd(man_train$BldgArea) * 2^.43

preds3 <- predict(mod3, newx=man_x_test, s='lambda.1se')
head(2^preds3)

set.seed(19723)
sample(10)
sample(10)


library(xgboost)

xg_train <- xgb.DMatrix(
    data=man_x_train,
    label=man_y_train
)
# you shouldn't do this, but we will for time
xg_val <- xgb.DMatrix(
    data=man_x_test,
    label=man_y_test
)

xg_train

mod5 <- xgb.train(
    data=xg_train,
    nrounds=1
)
mod5

mod5 %>% 
    xgb.plot.multi.trees()

mod6 <- xgb.train(
    data=xg_train,
    nrounds=1,
    eval_metric='rmse',
    watchlist=list(train=xg_train, validate=xg_val)
)

sqrt(mod3$cvm[which(mod3$lambda == mod3$lambda.min)])

mod7 <- xgb.train(
    data=xg_train,
    nrounds=100,
    eval_metric='rmse',
    watchlist=list(train=xg_train, validate=xg_val)
)

mod8 <- xgb.train(
    data=xg_train,
    nrounds=500,
    eval_metric='rmse',
    watchlist=list(train=xg_train, validate=xg_val)
)

mod8$evaluation_log %>% 
    dygraphs::dygraph()
mod8$evaluation_log %>% 
    .[validate_rmse == min(validate_rmse)]

mod9 <- xgb.train(
    data=xg_train,
    nrounds=16,
    eval_metric='rmse',
    watchlist=list(train=xg_train, validate=xg_val)
)

mod10 <- xgb.train(
    data=xg_train,
    nrounds=500,
    eval_metric='rmse',
    watchlist=list(train=xg_train, validate=xg_val),
    early_stopping_rounds=70
)

?xgb.train

mod11 <- xgb.train(
    data=xg_train,
    nrounds=500,
    eval_metric='rmse',
    watchlist=list(train=xg_train, validate=xg_val),
    early_stopping_rounds=70,
    max_depth=3
)

mod12 <- xgb.train(
    data=xg_train,
    nrounds=500,
    eval_metric='rmse',
    watchlist=list(train=xg_train, validate=xg_val),
    early_stopping_rounds=70,
    max_depth=9
)

search_grid <- tibble(
    index=1:20,
    nrounds=sample(10:300, size=20, replace=TRUE),
    max_depth=sample(2:8, size=20, replace=TRUE)
)
search_grid

library(purrr)

param_tuner <- search_grid %>% 
    tidyr::nest(-index) %>% 
    mutate(
        Model=map(
            data,
            ~ xgb.train(
                data=xg_train,
                nrounds=.x$nrounds,
                max_depth=.x$max_depth,
                watchlist=list(
                    train=xg_train,
                    validate=xg_val
                ),
                eval_metric='rmse'
            )
        )
    )

param_tuner
param_tuner$Model[[1]]$evaluation_log


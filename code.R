##########################################################
# CREATING SETS
##########################################################

#loading or installing the libraries.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(gridExtra)
library(ggthemes)
library(stringr)
library(lubridate)

#downloading the data and defining the data frame.

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

#creating the first partition, edx (90%) and validation(10%).
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set.
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set.
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#creating the second partition, edx_train (90% of edx) and edx_test (10% of edx).
set.seed(1, sample.kind="Rounding")
edx_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-edx_index,]
temp_1 <- edx[edx_index,]
edx_test <- temp_1 %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

#removing all the objects that we don't need.
rm(dl, ratings, movies, test_index, temp, movielens, removed,edx_index,temp_1)

#creating RMSE function to evaluate the predictions.
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#checking the dimensions of the datasets.
dimensions <- c("observations","variables")
edx_train_d <- c(nrow(edx_train),ncol(edx_train))
edx_test_d <- c(nrow(edx_test),ncol(edx_test))
edx_d <- c(nrow(edx),ncol(edx))
validation_d <- c(nrow(validation),ncol(validation))
data.frame(dimensions,edx_d,edx_train_d,edx_test_d,validation_d)
head(edx_train) %>% grid.table()

##########################################################
# EXPLORATORY DATA ANALYSIS
##########################################################

#1. USER COMPONENT.

#are all users the same in terms of rating? (the same theme is going to be applied to all the plots)
edx_train %>% 
  group_by(userId) %>%
  summarise(mean_rating_by_user = mean(rating)) %>%
  ggplot(aes(mean_rating_by_user)) +
  geom_histogram(bins = 30, color = "white", fill = "black") +
  ggtitle("Distribution of mean ratings by users") +
  xlab("Mean rating") +
  ylab("Frequency") +
  theme_clean()

#do all users rate the same number of times?
edx_train %>%
  group_by(userId) %>%
  summarise(number_of_ratings = n()) %>%
  ggplot(aes(number_of_ratings)) +
  geom_histogram(bins = 30, color = "white", fill = "black") +
  xlim(0,1000) +
  xlab("Number of ratings") +
  ylab("Frequency") +
  ggtitle("Distribution of number of ratings by users") +
  theme_clean()

#2. MOVIE COMPONENT.

#are all movies the same in terms of rating?
p1 <- edx_train %>% 
  group_by(movieId) %>%
  summarise(mean_rating_per_movie = mean(rating)) %>%
  ggplot(aes(mean_rating_per_movie)) +
  geom_histogram(bins = 30, color = "white", fill = "black") +
  ggtitle("Distribution of mean ratings by movies") +
  xlab("Mean rating") +
  ylab("Frequency") +
  theme_clean()

#are all movies rated the same number of times?

p2 <- edx_train %>% 
  group_by(movieId) %>%
  summarise(number_of_ratings = n()) %>%
  ggplot(aes(number_of_ratings)) +
  geom_histogram(bins = 30, color = "white", fill = "black") +
  xlim(0,5000) +
  ylim(0,2000) +
  ggtitle("Distribution of number of ratings by movies") +
  xlab("Number of ratings") +
  ylab("frequency")+
  theme_clean()

#joining both plots.
grid.arrange(p1,p2,nrow = 1)

#watching the importance of considering how many times the movies have been rated.
options(digits = 3)
edx_train %>% #top rated movies without any filter related to the number of ratings.
  select(title,movieId,rating) %>%
  group_by(title) %>%
  summarise(mean_rating = mean(rating),
            number_of_ratings = n()) %>%
  top_n(10,mean_rating) %>%
  arrange(desc(mean_rating)) %>%
  grid.table()
edx_train %>% ##top rated movies filtering out movies that have not been rated by more than 100 people.
  select(title,movieId,rating) %>%
  group_by(title) %>%
  summarise(mean_rating = mean(rating),
            number_of_ratings = n()) %>%
  filter(number_of_ratings > 100) %>%
  top_n(10,mean_rating) %>%
  arrange(desc(mean_rating)) %>%
  grid.table()

#3. YEAR OF RELEASE COMPONENT.

#extracting the year of release out of the title variable.
edx_train <- edx_train %>% 
  mutate(movie_year = str_extract(edx_train$title,"\\([0-9]{4}\\)")) %>%
  mutate(movie_year = str_remove(movie_year,"^\\(")) %>%
  mutate(movie_year = as.factor(str_remove(movie_year,"\\)")))

#are the mean ratings independent of the year of release?
edx_train %>%
  group_by(movie_year) %>%
  summarise(mean_rating = mean(rating)) %>% 
  ggplot(aes(movie_year,mean_rating)) +
  geom_point() +
  geom_hline(yintercept = mean_rating, color = "gray", size = 1) +
  ylim(0,5) +
  ggtitle("Average ratings by movie's release year") +
  xlab("Release year") +
  ylab("Mean rating") +
  annotate(geom="text", x=30, y=3.3, label="Overall mean rating", color="gray")+
  theme_clean() +
  theme(axis.text.x = element_text(angle = 90,hjust = 1,size = 7))

#testing with ANOVA the significance of the difference between ratings done before and after 1980.
edx_train <- edx_train %>% #creating he factor.
  mutate(movie_year = as.character(movie_year)) %>%
  mutate(movie_year = as.numeric(movie_year)) %>%
  mutate(factor_year_release = ifelse(movie_year > 1980,"recent","classic")) %>% 
  mutate(factor_year_release = factor(factor_year_release, levels = c("recent","classic")))

options(digits = 2)
summary(aov(rating ~ factor_year_release, data = edx_train)) #ANOVA.

#creating the factor in the remaining datasets.
#edx_test
edx_test <- edx_test %>% 
  mutate(movie_year = str_extract(edx_test$title,"\\([0-9]{4}\\)")) %>%
  mutate(movie_year = str_remove(movie_year,"^\\(")) %>%
  mutate(movie_year = as.factor(str_remove(movie_year,"\\)")))

edx_test <- edx_test %>%
  mutate(movie_year = as.character(movie_year)) %>%
  mutate(movie_year = as.numeric(movie_year)) %>%
  mutate(factor_year_release = ifelse(movie_year > 1980,"recent","classic")) %>% 
  mutate(factor_year_release = factor(factor_year_release, levels = c("recent","classic")))
#edx
edx <- edx %>% 
  mutate(movie_year = str_extract(edx$title,"\\([0-9]{4}\\)")) %>%
  mutate(movie_year = str_remove(movie_year,"^\\(")) %>%
  mutate(movie_year = as.factor(str_remove(movie_year,"\\)")))

edx <- edx %>%
  mutate(movie_year = as.character(movie_year)) %>%
  mutate(movie_year = as.numeric(movie_year)) %>%
  mutate(factor_year_release = ifelse(movie_year > 1980,"recent","classic")) %>% 
  mutate(factor_year_release = factor(factor_year_release, levels = c("recent","classic")))
#validation
validation <- validation %>% 
  mutate(movie_year = str_extract(validation$title,"\\([0-9]{4}\\)")) %>%
  mutate(movie_year = str_remove(movie_year,"^\\(")) %>%
  mutate(movie_year = as.factor(str_remove(movie_year,"\\)")))

validation <- validation %>%
  mutate(movie_year = as.character(movie_year)) %>%
  mutate(movie_year = as.numeric(movie_year)) %>%
  mutate(factor_year_release = ifelse(movie_year > 1980,"recent","classic")) %>% 
  mutate(factor_year_release = factor(factor_year_release, levels = c("recent","classic")))

#3. DAY, MONTH OR YEAR OF THE RATING COMPONENT.

#Are the ratings different depending on the day of the week in which they are made?
edx_train <- edx_train %>% mutate(week_day = wday(as_datetime(timestamp))) #creating the "day" variable.
edx_train %>%
  group_by(week_day) %>%
  summarise(mean_rating = mean(rating)) %>% 
  ggplot(aes(week_day,mean_rating)) +
  geom_point(size = 3) +
  geom_hline(yintercept = mean_rating, color = "gray", size = 1) +
  ylim(0,5) +
  ggtitle("Average ratings by day of the week") +
  xlab("Day of the week") +
  ylab("Mean rating") +
  annotate(geom="text", x=2, y=3.2, label="Overall mean rating", color="gray")+
  theme_clean()

#Are the ratings different depending on the month in which they are made?
edx_train <- edx_train %>% mutate(month = month(as_datetime(timestamp))) #creating the "month" variable.
edx_train %>%
  group_by(month) %>%
  summarise(mean_rating = mean(rating)) %>% 
  ggplot(aes(month,mean_rating)) +
  geom_point(size = 3) +
  geom_hline(yintercept = mean_rating, color = "gray", size = 1) +
  ylim(0,5) +
  xlim(0,13) +
  ggtitle("Average ratings by month") +
  xlab("Month") +
  ylab("Mean rating") +
  annotate(geom="text", x=4, y=3.2, label="Overall mean rating", color="gray")+
  theme_clean()

#Are the ratings different depending on the year in which they are made?
edx_train <- edx_train %>% mutate(year = year(as_datetime(timestamp))) #creating the "month" variable.
edx_train %>%
  group_by(year) %>%
  summarise(mean_rating = mean(rating)) %>% 
  ggplot(aes(year,mean_rating)) +
  geom_point(size = 3) +
  geom_hline(yintercept = mean_rating, color = "gray", size = 1) +
  ylim(0,5) +
  ggtitle("Average ratings by year") +
  xlab("Year") +
  ylab("Mean rating") +
  annotate(geom="text", x=2000, y=3.2, label="Overall mean rating", color="gray")+
  theme_clean()

#removing the variables as they are not going to be used.
edx_train <- edx_train[,1:8]

#4. GENRES COMPONENT.

#generating all the genres variables.
edx_train <- edx_train %>% mutate(action = as.factor(like(edx_train$genres, "Action")),
                                  animation = as.factor(like(edx_train$genres, "Animation")),
                                  adventure = as.factor(like(edx_train$genres, "Adventure")),
                                  childrens = as.factor(like(edx_train$genres, "Childern's")),
                                  comedy = as.factor(like(edx_train$genres, "Comedy")),
                                  crime = as.factor(like(edx_train$genres, "Crime")),
                                  documentary = as.factor(like(edx_train$genres, "Documentary")),
                                  drama = as.factor(like(edx_train$genres, "Drama")),
                                  fantasy = as.factor(like(edx_train$genres, "Fantasy")),
                                  film_noir = as.factor(like(edx_train$genres, "Film-noir")),
                                  Horror = as.factor(like(edx_train$genres, "Horror")),
                                  musical = as.factor(like(edx_train$genres, "Musical")),
                                  mystery = as.factor(like(edx_train$genres, "Mystery")),
                                  romance = as.factor(like(edx_train$genres, "Romance")),
                                  sci_fi = as.factor(like(edx_train$genres, "Sci-Fi")),
                                  thriller = as.factor(like(edx_train$genres, "Thriller")),
                                  war = as.factor(like(edx_train$genres, "War")),
                                  western = as.factor(like(edx_train$genres, "Western")))

#Are the mean ratings different depending on the genre of the movie? Generating all 18 plots.
pl1 <- edx_train %>%
  group_by(movieId) %>%
  group_by(action) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(action,mean_rating, col = mean_rating)) + 
  geom_label(aes(action,mean_rating, label = mean_rating, size = 1)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Action") +
  theme_clean() +
  theme(legend.position = "none")
pl2 <- edx_train %>%
  group_by(movieId) %>%
  group_by(adventure) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(adventure,mean_rating, col = mean_rating)) + 
  geom_label(aes(adventure,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Adventure") +
  theme_clean() +
  theme(legend.position = "none")
pl3 <- edx_train %>%
  group_by(movieId) %>%
  group_by(childrens) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(childrens,mean_rating, col = mean_rating)) + 
  geom_label(aes(childrens,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Children's") +
  theme_clean() +
  theme(legend.position = "none")
pl4 <- edx_train %>%
  group_by(movieId) %>%
  group_by(comedy) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(comedy,mean_rating, col = mean_rating)) + 
  geom_label(aes(comedy,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Comedy") +
  theme_clean() +
  theme(legend.position = "none")
pl5 <- edx_train %>%
  group_by(movieId) %>%
  group_by(crime) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(crime,mean_rating, col = mean_rating)) + 
  geom_label(aes(crime,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Crime") +
  theme_clean() +
  theme(legend.position = "none")
pl6 <- edx_train %>%
  group_by(movieId) %>%
  group_by(documentary) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(documentary,mean_rating, col = mean_rating)) + 
  geom_label(aes(documentary,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Documentary") +
  theme_clean() +
  theme(legend.position = "none")
pl7 <- edx_train %>%
  group_by(movieId) %>%
  group_by(drama) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(drama,mean_rating, col = mean_rating)) + 
  geom_label(aes(drama,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Drama") +
  theme_clean() +
  theme(legend.position = "none")
pl8 <- edx_train %>%
  group_by(movieId) %>%
  group_by(fantasy) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(fantasy,mean_rating, col = mean_rating)) + 
  geom_label(aes(fantasy,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Fantasy") +
  theme_clean() +
  theme(legend.position = "none")
pl9 <- edx_train %>%
  group_by(movieId) %>%
  group_by(film_noir) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(film_noir,mean_rating, col = mean_rating)) + 
  geom_label(aes(film_noir,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Film noir") +
  theme_clean() +
  theme(legend.position = "none")
pl10 <- edx_train %>%
  group_by(movieId) %>%
  group_by(Horror) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(Horror,mean_rating, col = mean_rating)) + 
  geom_label(aes(Horror,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Horror") +
  theme_clean() +
  theme(legend.position = "none")
pl11 <- edx_train %>%
  group_by(movieId) %>%
  group_by(musical) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(musical,mean_rating, col = mean_rating)) + 
  geom_label(aes(musical,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Musical") +
  theme_clean() +
  theme(legend.position = "none")
pl12 <- edx_train %>%
  group_by(movieId) %>%
  group_by(mystery) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(mystery,mean_rating, col = mean_rating)) + 
  geom_label(aes(mystery,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Mystery") +
  theme_clean() +
  theme(legend.position = "none")
pl13 <- edx_train %>%
  group_by(movieId) %>%
  group_by(romance) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(romance,mean_rating, col = mean_rating)) + 
  geom_label(aes(romance,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Romance") +
  theme_clean() +
  theme(legend.position = "none")
pl14 <- edx_train %>%
  group_by(movieId) %>%
  group_by(sci_fi) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(sci_fi,mean_rating, col = mean_rating)) + 
  geom_label(aes(sci_fi,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Sci Fi") +
  theme_clean() +
  theme(legend.position = "none")
pl15 <- edx_train %>%
  group_by(movieId) %>%
  group_by(thriller) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(thriller,mean_rating, col = mean_rating)) + 
  geom_label(aes(thriller,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Thriller") +
  theme_clean() +
  theme(legend.position = "none")
pl16 <- edx_train %>%
  group_by(movieId) %>%
  group_by(war) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(war,mean_rating, col = mean_rating)) + 
  geom_label(aes(war,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("War") +
  theme_clean() +
  theme(legend.position = "none")
pl17 <- edx_train %>%
  group_by(movieId) %>%
  group_by(western) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(western,mean_rating, col = mean_rating)) + 
  geom_label(aes(western,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Western") +
  theme_clean() +
  theme(legend.position = "none")
pl18 <- edx_train %>%
  group_by(movieId) %>%
  group_by(animation) %>%
  summarise(mean_rating = round(mean(rating),2)) %>%
  ggplot(aes(animation,mean_rating, col = mean_rating)) + 
  geom_label(aes(animation,mean_rating, label = mean_rating, size = 4)) +
  ylim(0,5) +
  xlab("") +
  ylab("Mean rating") +
  ggtitle("Animation") +
  theme_clean() +
  theme(legend.position = "none")

#plotting them together tanks to the grid.arrange function.
grid.arrange(pl1,pl2,pl3,pl4,pl5,pl6,pl7,pl8,pl9,pl10,pl11,pl12,pl13,pl14,pl15,pl16,pl17,pl18, nrow = 3)

#checking with ANOVAS if the genres when appears to be more differences really present significative differences.
summary(aov(rating ~ documentary, data = edx_train))
summary(aov(rating ~ war, data = edx_train))
summary(aov(rating ~ drama, data = edx_train)) 

#adding the genres that results to have significance difference to the other sets.
edx_test <- edx_test %>% mutate(documentary = as.factor(like(edx_test$genres, "Documentary")),
                                drama = as.factor(like(edx_test$genres, "Drama")),
                                war = as.factor(like(edx_test$genres, "War")))
edx <- edx %>% mutate(documentary = as.factor(like(edx$genres, "Documentary")),
                      drama = as.factor(like(edx$genres, "Drama")),
                      war = as.factor(like(edx$genres, "War")))
validation <- validation %>% mutate(documentary = as.factor(like(validation$genres, "Documentary")),
                                    drama = as.factor(like(validation$genres, "Drama")),
                                    war = as.factor(like(validation$genres, "War")))

##########################################################
# CONSTRUCTION OF THE MODEL (ADDING THE COMPONENTS SEQUENTALLY AND CHECKING THE CHANGES ON THE RMSES)
##########################################################

#1. FIRST MODEL: ALL THE PREDICTIONS ARE THE MEAN RATING.

#generating the prediction on the test set.
edx_test <- edx_test %>% mutate(prediction_m_0 = round(mean(edx_train$rating),2))

#computing the RMSE of this approach.
RMSE_m_0 <- RMSE(edx_test$rating,edx_test$prediction_m_0)
round(RMSE_m_0,2)

#2. SECOND MODEL: ADDING THE USER COMPONENT.

#obtaining the optimal lambda to get the user coefficients that minimize RMSE on test data.
lambdas <- seq(0, 20, 0.5)
mean <- mean(edx_train$rating)
user_1 <- edx_train %>% #its called user_1 because user_2 its generated in a following point (with other lambda).
  group_by(userId) %>%
  summarize(s = sum(rating - mean), nu = n())
rmses_1 <- sapply(lambdas, function(l){
  predicted_ratings <- edx_test %>%
    left_join(user_1, by='userId') %>%
    mutate(user_coefficient = s/(nu+l)) %>%
    mutate(prediction_m_1 = mean + user_coefficient) %>%
    pull(prediction_m_1)
  return(RMSE(predicted_ratings, edx_test$rating))
})
data_frame(lambdas,rmses_1) %>%
  ggplot(aes(lambdas,rmses_1)) +
  geom_point() +
  ggtitle("Choosing the optimal lambda") +
  theme_clean()
l_1 <- lambdas[which.min(rmses_1)]
l_1

#getting the prediction by generating the user coefficients with l_1 and adding them to the first model prediction.
edx_test <- edx_test %>%
  left_join(user_1, by='userId') %>%
  mutate(user_coefficient_1 = s/(nu+l_1)) %>%
  mutate(prediction_m_1 = round(mean + user_coefficient_1,2))

#getting the RMSE of this model and comparing it to the previously obtained.
options(digits=3)
RMSE_m_1 <- RMSE(edx_test$rating,edx_test$prediction_m_1)

Models <- c("Model with the constant","Model with the constant and the user effect")
RMSE <- c(RMSE_m_0,RMSE_m_1)
data_frame(Models,RMSE) %>% grid.table()

#3. THIRD MODEL: ADDING THE MOVIE COMPONENT.

#Obtaining the optimal lambda for both components, the user and movie component.
rmses_2 <- sapply(lambdas, function(l){
  user_2 <- edx_train %>%
    group_by(userId) %>%
    summarize(user_coefficient_2 = sum(rating - mean)/(n()+l))
  movies_1 <- edx_train %>%
    left_join(user_2, by="userId") %>%
    group_by(movieId) %>%
    summarize(movie_coefficient_1 = sum(rating - user_coefficient_2 - mean)/(n()+l))
  predicted_ratings <-
    edx_test %>%
    left_join(user_2, by = "userId") %>%
    left_join(movies_1, by = "movieId") %>%
    mutate(prediction_m_2 = mean + user_coefficient_2 + movie_coefficient_1) %>%
    pull(prediction_m_2)
  return(RMSE(predicted_ratings, edx_test$rating))
})
data_frame(lambdas,rmses_2) %>%
  ggplot(aes(lambdas,rmses_2)) +
  geom_point() +
  ggtitle("Choosing the optimal lambda") +
  theme_clean()
l_2 <- lambdas[which.min(rmses_2)]
l_2

#generating the new user coefficient and the movie coefficient with the new optimal lambda (l_2).Creating the new prediction by adding this coefficients to the mean.
user_2 <- edx_train %>%
  group_by(userId) %>%
  summarize(user_coefficient_2 = sum(rating - mean)/(n()+l_2))
movies_1 <- edx_train %>%
  left_join(user_2, by="userId") %>%
  group_by(movieId) %>%
  summarize(movie_coefficient_1 = sum(rating - user_coefficient_2 - mean)/(n()+l_2))
edx_test <- edx_test %>%
  left_join(user_2, by = "userId") %>%
  left_join(movies_1, by = "movieId") %>%
  mutate(prediction_m_2 = round(mean + user_coefficient_2 + movie_coefficient_1,2))

#Obtaining the new RMSE and comparing with the previously obtained.
options(digits=3)
RMSE_m_2 <- RMSE(edx_test$rating,edx_test$prediction_m_2)

Models <- c("Model with the constant","Model with the constant and the user effect","Model with the constant, the user and movie effect")
RMSE <- c(RMSE_m_0,RMSE_m_1,RMSE_m_2)
data_frame(Models,RMSE) %>% grid.table()

#4. FOURTH MODEL: ADDING THE YEAR OF RELEASE COMPONENT.
#The residuals of the previous model are obtained and the coefficient for the factor is estimated with this residuals been the dependent variable.

#generating the prediction and residuals of the third model on the training data.
edx_train <- edx_train %>%
  left_join(user_2, by = "userId") %>%
  left_join(movies_1, by = "movieId") %>%
  mutate(prediction_m_2 = round(mean + user_coefficient_2 + movie_coefficient_1,2),
         residuals = rating - prediction_m_2)

#estimating the factor "year of release", extracting the significant coefficient and generating the new predictions.
linear_model <- summary(lm(residuals ~ 0 + factor_year_release,data = edx_train))
linear_model$coefficients

recent <- linear_model$coefficients[1,1]

edx_test <- edx_test %>% 
  mutate(prediction_m_3 =(ifelse(factor_year_release == "classic",
                                 round(mean + user_coefficient_2 + movie_coefficient_1,2),
                                 round(mean + user_coefficient_2 + movie_coefficient_1 + recent,2))))

#generating the RMSE and comparing it to the preovious models.
options(digits = 3)
RMSE_m_3 <- RMSE(edx_test$rating,edx_test$prediction_m_3)
Models <- c("Model with the constant","Model with the constant and the user effect","Model with the constant, the user and movie effect","Model with the constant, the user the movie and the year of release effect")
RMSE <- c(RMSE_m_0,RMSE_m_1,RMSE_m_2,RMSE_m_3)
data_frame(Models,RMSE) %>% grid.table()
RMSE_m_3 < RMSE_m_2 #the difference is so small and it have to be checked with this question.

#5. FIFTH MODEL: ADDING THE GENRES COMPONENT.

#Same as before, prediction and residuals of the previous are generated on training data.
edx_train <- edx_train %>%
  mutate(prediction_m_3 =(ifelse(factor_year_release == "classic",
                                 round(mean + user_coefficient_2 + movie_coefficient_1,2),
                                 round(mean + user_coefficient_2 + movie_coefficient_1 + recent,2))),
         residuals_2 = rating - prediction_m_3)

#Coefficients are estimated and the significant ones are saved to compute the prediction.
linear_model_2 <- summary(lm(residuals_2 ~ 0 + documentary + drama + war,data = edx_train))
linear_model_2$coefficient
documentary_true <- linear_model_2$coefficients[2,1]
not_documentary <- linear_model_2$coefficients[1,1]
drama_true <- linear_model_2$coefficients[3,1]

#generating the prediction taking into account all significant genres combination
edx_test <- edx_test %>% 
  mutate(prediction_m_4_p1 = ifelse(drama == "TRUE" & documentary == "TRUE", round(mean + user_coefficient_2 + movie_coefficient_1 + drama_true + documentary_true,2), 0),
         prediction_m_4_p2 = ifelse(documentary == "TRUE" & drama == "FALSE", round(mean + user_coefficient_2 + movie_coefficient_1 + documentary_true,2), 0),
         prediction_m_4_p3 = ifelse(documentary == "FALSE" & drama == "FALSE", round(mean + user_coefficient_2 + movie_coefficient_1 + not_documentary,2), 0),
         prediction_m_4_p4 = ifelse(documentary == "FALSE" & drama == "TRUE", round(mean + user_coefficient_2 + movie_coefficient_1 + not_documentary + drama_true,2), 0))

edx_test <- edx_test %>%
  mutate(prediction_m_4 = prediction_m_4_p1 + prediction_m_4_p2 +  prediction_m_4_p3 + prediction_m_4_p4)

#computing the RMSE and checking if it has been improved.
options(digits = 3)
RMSE_m_4 <- RMSE(edx_test$rating,edx_test$prediction_m_4)
Models <- c("Model with the constant","Model with the constant and the user effect","Model with the constant, the user and movie effect","Model with the constant, the user the movie and the year of release effect","Model with the constant, the user the movie, the year of release and the genres effect")
RMSE <- c(RMSE_m_0,RMSE_m_1,RMSE_m_2, RMSE_m_3, RMSE_m_4)
data_frame(Models,RMSE) %>% grid.table()
RMSE_m_4 < RMSE_m_3

##########################################################
# CHECKING THE MODEL PERFORMANCE (COMPUTING THE RMSE OF THE FINAL MODEL ON AN INDEPENDENT DATA SET)
##########################################################

#1. CHECKING THE PERFORMANCE AND DISTRIBUTION OF RESIDUALS.

#generating the predictions of the final model in the validation set.
validation <- validation %>%
  left_join(user_2, by = "userId") %>%
  left_join(movies_1, by = "movieId") %>%
  mutate(prediction_p1 = ifelse(drama == "TRUE" & documentary == "TRUE", round(mean + user_coefficient_2 + movie_coefficient_1 + drama_true + documentary_true,2), 0),
         prediction_p2 = ifelse(documentary == "TRUE" & drama == "FALSE", round(mean + user_coefficient_2 + movie_coefficient_1 + documentary_true,2), 0),
         prediction_p3 = ifelse(documentary == "FALSE" & drama == "FALSE", round(mean + user_coefficient_2 + movie_coefficient_1 + not_documentary,2), 0),
         prediction_p4 = ifelse(documentary == "FALSE" & drama == "TRUE", round(mean + user_coefficient_2 + movie_coefficient_1 + not_documentary + drama_true,2), 0),
         prediction = prediction_p1 + prediction_p2 + prediction_p3 + prediction_p4)

#computing the RMSE.
validation %>%
  filter(!is.na(prediction)) %>%
  summarise(RMSE = round(sqrt(mean((rating - prediction)^2)),4)) %>%
  pull(RMSE)

#checking with two different plots the distribution of the residuals.

Model <- c("Residuals achieve from the final model")
d_1 <- validation %>%
  summarise(residuals = abs(rating - prediction)) %>%
  ggplot(aes(residuals)) +
  geom_density(fill = "gray", col = "gray") +
  theme_clean() + 
  xlab("Residuals") + 
  ylab( "Density") + 
  ggtitle("Distribution of the residuals (density plot)")

d_2 <- validation %>%
  summarise(residuals = abs(rating - prediction)) %>%
  ggplot(aes(Model,residuals)) +
  geom_boxplot() +
  theme_clean() + 
  ylab("Residuals") + 
  ggtitle("Distribution of the residuals (box plot)")
grid.arrange(d_1,d_2,nrow = 1)

#checking the percentage of ratings that falls below a star.
rows_validation <- nrow(validation)
validation %>%
  mutate(residuals = abs(rating - prediction)) %>%
  filter(residuals <= 1) %>%
  summarise(proportion_under_1 = round(n()/rows_validation,2)) %>%
  pull(proportion_under_1)

#checking the percentage of ratings the model correctly predicts if a decision rule (split like and no like with 3 stars) is established.
validation %>%
filter(!is.na(prediction)) %>%
  mutate(like_predictlike = ifelse(rating >= 3 & prediction >= 3, "sucess","fail"),         
         dontlike_predictdontlike = ifelse(rating <= 3 & prediction <= 3, "sucess","fail")) %>%
  summarise(sucess_proportion_1 = mean(like_predictlike == "sucess"),
            sucess_proportion_2 = mean(dontlike_predictdontlike == "sucess"),
            sucess_proportion = round(sum(sucess_proportion_1,sucess_proportion_2),2)) %>%
  pull(sucess_proportion)

#2. CHECKING THE RATINGS THAT HAVE BIG RESIDUALS.

#Generating the set with the ratings that have residuals bigger than 2.
residuals_over_two <- validation %>%
  mutate(residuals = abs(rating - prediction)) %>%
  filter(residuals >= 2)
dim(residuals_over_two)

#checking the number of ratings that users of this set have given.
residuals_over_two  %>%
  group_by(userId) %>%
  mutate(number_of_ratings = n()) %>%
  ggplot(aes(number_of_ratings)) +
  geom_histogram(bins = 30, color = "white", fill = "black") +
  ggtitle("Distribution of number of ratings") +
  xlab("Number of ratings") +
  ylab("Frequency") +
  theme_clean()

##########################################################
# KEEPING ONLY RMSE AND VALIDATION SET WITH PREDICTIONS
##########################################################

#If someone executes this code entirely and only wants to keep the RMSE obtained in the validation set
#and the validation set itself, it could be done executing the following final piece of code.
validation %>%
  filter(!is.na(prediction)) %>%
  summarise(RMSE = round(sqrt(mean((rating - prediction)^2)),4)) %>%
  pull(RMSE)

remove(edx,edx_train,edx_test,linear_model,linear_model_2,movies_1,residuals_over_two,user_1,user_2,documentary_true,drama_true,l_1,l_2,lambdas,mean,mean_rating,Models,not_documentary,recent,RMSE,RMSE_m_0,RMSE_m_1,rmses_1,rmses_2,RMSE_m_2,RMSE_m_3,RMSE_m_4)

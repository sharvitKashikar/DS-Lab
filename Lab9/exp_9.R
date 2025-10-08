#Aim: To load and explore data in R using tibbles, perform descriptive analysis, fit a linear regression model, visualize the regression line, and evaluate model performance using statistical metrics such as R-squared and p-value.

#Load the libraries you will need in R programming
#a) Load the data set as a tibble (readr has a function that does this directly)
#b) Get the descriptive for your data so you can understand what you're dealing with
#c) Explore the data and see if there is any interesting trends to consider
#d) Define a linear model
#e) Plot the regression line
#f) Print the results of the model
#g) How many observations was the regression line run on?
#h) What is the R-squared significant predictor of price
#i) Determine if size is a statistically significant predictor of price
#j) What is the regression equation with this regression model?

library(readr)
library(tibble)
library(dplyr)
library(ggplot2)


#a) Load the data set:

house_data <- tibble(
  Area = c(800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000),
  Price = c(25, 30, 36, 45, 52, 58, 65, 72, 80, 88)
)


print(house_data)

#b) Description of our data

house_data %>%
  summarise(
    Count = n(),
    Mean_Area = mean(Area),
    SD_Area = sd(Area),
    Min_Area = min(Area),
    Max_Area = max(Area),
    Mean_Price = mean(Price),
    SD_Price = sd(Price),
    Min_Price = min(Price),
    Max_Price = max(Price)
  )

#c) Explore data

ggplot(house_data, aes(x = Area, y = Price)) +
  geom_point(color = "blue", size = 3) +
  labs(title = "House Price vs Area",
       x = "Area (sqft)",
       y = "Price (lakh Rupees)") +
  theme_minimal()

#d) Define a linear model

model <- lm(Price ~ Area, data = house_data)


#e) Plot the regression line

ggplot(house_data, aes(x = Area, y = Price)) +
  geom_point(color = "blue", size = 3) +
  geom_smooth(method = "lm", se = FALSE, color = "red", linewidth = 1.2) +
  labs(title = "Linear Regression: Price ~ Area",
       x = "Area (sqft)",
       y = "Price (lakh Rupees)") +
  theme_minimal()


#f) summary of the model

summary(model)

#g) The regression was run on 10 observations

#h) What is the R-squared

print(paste("R squared error = ", summary(model)$r.squared))

#i) Is Size statistically significant predictor?

summary(model)$coefficients

#j) Regression equation

#Price = 2.123 + 0.0285 × Area
#Price = coeff. * area + intercept













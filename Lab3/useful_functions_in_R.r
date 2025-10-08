# Arguments of sample() function
cat("Arguments of sample() function:\n")
args(sample)
cat("\n")

# Sample of students selected
cat("Sample of students selected:\n")
students <- c("Alice", "Bob", "Charlie", "David", "Eve", "Frank",
              "Grace", "Hannah", "Ian", "Julia")
selected_students <- sample(students, size = 3, replace = FALSE)
print(selected_students)
cat("\n")

# Arguments of median() function
cat("Arguments of median() function:\n")
args(median)
cat("\n")

v1 <- c(15, 25, NA, 20, 40)
cat("Median with na.rm = TRUE:\n")
print(median(v1, na.rm = TRUE))
cat("\n")

cat("Median with na.rm = FALSE:\n")
print(median(v1, na.rm = FALSE))
cat("\n")

# Which function example
x <- c(5, 2, 7, 3, 9)
cat("Indices where x > 5:\n")
print(which(x > 5))
cat("\n")

# Frequency table
colors <- c("cyan", "violet", "cyan", "turquoise", "violet", "violet")
cat("Frequency table of colors:\n")
print(table(colors))
cat("\n")

# Subset of iris dataset
cat("Subset of iris dataset (Species == 'setosa' and Sepal.Length > 5):\n")
data(iris)
subset_setosa <- subset(iris, Species == "setosa" & Sepal.Length > 5)
print(subset_setosa)
cat("\n")

# Age categorization using ifelse
age <- c(12, 25, 35, 8, 60)
category <- ifelse(age < 18, "Minor", "Adult")
cat("Age categories:\n")
print(data.frame(age, category))
cat("\n")

# ggplot2 plot of Sepal dimensions
cat("Plotting Sepal dimensions using ggplot2...\n")
library(ggplot2)
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point() +
  labs(title = "Sepal Dimensions by Species",
       x = "Sepal Length",
       y = "Sepal Width") +
  theme_minimal()

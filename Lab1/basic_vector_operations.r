# Aim: To study basic vector operations of R Programming

# init
x <- 10:20
print(x)

# a)
y <- x + 2
print(y)

# b)
z <- y * 3
print(z)

# c)
answer <- (z - 6) / 2

# d)
print(answer)

# e)
answer <- (((x + 2) * 3) - 6) / 2
print(answer)

# f)
# Theory

# g)
# Theory

# h)
a <- seq(from = 1, to = 12, by = 0.5)
b <- seq(1, 10)^3
print(a)
print(b)

# i)
c <- c(1, 2, 2, 2, 3, 4, 5)
value <- 2
count <- sum(c == value)
print(count)

# j)
vector1 <- c(1, 2, 3, 4, 5)
vector2 <- c(3, 4, 5, 6, 7)
common_elements <- intersect(vector1, vector2)
print(common_elements)

# k)
vector3 <- c(1:20, 19:1)
print(vector3)

# l)
string <- "data science"
pattern <- "data"
print(grepl(pattern, string))

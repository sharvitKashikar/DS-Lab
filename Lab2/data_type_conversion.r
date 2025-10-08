# a)
monster <- c(T, T, T, F, T, T, T, T, T, T)
print(monster)
class(monster)

# b)
yugioh <- c("A", "B", "C", "D")
print(yugioh)
class(yugioh)

# c)
combined <- c(monster, yugioh)
print(combined)
class(combined)

# d)
atk <- c(1, 2, 3, 4, 5)
coerce.check <- c(atk, monster)
print(coerce.check)
class(coerce.check)

# e)
a <- 2
b <- "2"

print(as.character(a))
print(as.numeric(b))

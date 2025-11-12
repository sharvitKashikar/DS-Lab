# 1) Put data here (replace with your values)
weight_loss <- c(2.1, 1.8, 3.2, 2.5, 1.9, 2.7, 3.0, 2.4, 1.6, 2.8)

# 2) Basic stats
n <- length(weight_loss) # sample size
mean_x <- mean(weight_loss) # sample mean
sd_x <- sd(weight_loss) # sample standard deviation (uses n-1)
se <- sd_x / sqrt(n) # standard error

cat("n =", n, "\n")
cat("Mean =", mean_x, "kg\n")
cat("SD =", sd_x, "kg\n")
cat("SE =", se, "kg\n\n")

# 3) Manual t-value (HO: mu=0)
mu0 <- 0
t_value <- (mean_x - mu0) / se
df <- n - 1
cat("t-value (manual) =", t_value, " (df=", df, ")\n\n")

# 4) p-value for one-sided test (right tail; Ha: mean > 0)
p_one_sided <- 1 - pt(t_value, df)
cat("One-sided p-value =", p_one_sided, "\n\n")

# 5) Same test using built-in function (verifies result)
tt <- t.test(weight_loss, mu = mu0, alternative = "greater")
print(tt)

# 6) Decision at alpha = 0.05
alpha <- 0.05
if (p_one_sided < alpha) {
  cat("\nDecision: Reject HO evidence the program causes weight loss (mean > 0).\n")
} else {
  cat("\nDecision: Fail to reject HO no significant evidence of weight loss.\n")
}
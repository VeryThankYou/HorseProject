D <- read.delim("horse_data23.txt")

par(mfrow = c(1, 1))
boxplot(D$A ~ D$lameLeg, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)
boxplot(D$S ~ D$lameLeg, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)
boxplot(D$W ~ D$lameLeg, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)

par(mfrow = c(3, 1))

boxplot(D$A ~ D$horse, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)
boxplot(D$S ~ D$horse, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)
boxplot(D$W ~ D$horse, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)


# If horse has a significant effect on the symmetry scores
L_A <- lm(A ~ horse * lameLeg, data = D)
anova(L_A)

L_S <- lm(S ~ horse * lameLeg, data = D)
anova(L_S)

L_W <- lm(W ~ horse * lameLeg, data = D)
anova(L_W)

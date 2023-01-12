D <- read.delim("horse_data23.txt")

par(mfrow = c(1, 1))
boxplot(D$A ~ D$lameLeg, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)
boxplot(D$S ~ D$lameLeg, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)
boxplot(D$W ~ D$lameLeg, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)

par(mfrow = c(3, 1))

boxplot(D$A ~ D$horse, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)
boxplot(D$S ~ D$horse, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)
boxplot(D$W ~ D$horse, col = c("red", "blue", "green", "yellow", "white"), horizontal = TRUE)
print(D[D$horse == c("B5", "B9"), ])
print(var.test(A ~ horse, data = D[D$horse == c("B5", "B9"), ]))
print(var.test(S ~ horse, data = D[D$horse == c("B4", "B6"), ]))
print(var.test(W ~ horse, data = D[D$horse == c("B1", "B7"), ]))
print(var.test(W ~ lameLeg, data = D[D$lameLeg == c("none", "right:fore"), ]))

print(kruskal.test(A ~ horse, data = D))
print(kruskal.test(S ~ horse, data = D))
#print(kruskal.test(W ~ horse, data = D))

table(D$lameLeg, D$horse)
summary(lm(lameLeg ~ horse, data = D))


# If horse has a significant effect on the symmetry scores
L_A <- lm(A ~ horse * lameLeg, data = D)
#print(anova(L_A))

L_S <- lm(S ~ horse * lameLeg, data = D)
#print(anova(L_S))

L_W <- lm(W ~ horse, data = D)
print(anova(L_W))

L_W2 <- lm(W ~ horse + lameLeg, data = D)
print(anova(L_W2))

L_W3 <- lm(W ~ horse * lameLeg, data = D)
print(anova(L_W3))

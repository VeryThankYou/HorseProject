D <- read.delim("horse_data23.txt")
print(D)
par(mfrow = c(1, 3))
boxplot(D[D$lameLeg == "none", "A"], D[D$lameLeg != "none", "A"], col = c("red", "blue"))
boxplot(D[D$lameLeg == "none", "S"], D[D$lameLeg != "none", "S"], col = c("red", "blue"))
boxplot(D[D$lameLeg == "none", "W"], D[D$lameLeg != "none", "W"], col = c("red", "blue"))
legend(1, 95, legend = c("No lameness", "Lame"))

# If horse has a significant effect on the symmetry scores
data.horse <- D
L_A <- lm(A ~ horse * lameLeg, data = data.horse)
anova(L_A)

L_S <- lm(S ~ horse * lameLeg, data = data.horse)
anova(L_S)

L_W <- lm(W ~ horse * lameLeg, data = data.horse)
anova(L_W)

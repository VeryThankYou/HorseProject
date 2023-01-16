D <- read.delim("horse_data23.txt")

qqnorm(D$W)
qqline(D$W)

print(anova(lm(log(W) ~ horse, data = D)))
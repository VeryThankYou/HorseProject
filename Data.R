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


# Data description
table(D$lameLeg, D$horse)

# Symmetry scores vs horse
boxplot(A ~ horse, data =D)
boxplot(S ~ horse, data =D)
boxplot(W ~ horse, data =D)

### OK
par(mfrow = c(1, 1))
par(mar = c(5, 8, 4, 2) + 0.1)
boxplot(D$A ~ D$lameLeg, main = "Boxplot of A and W by LameLeg", ylab = "Value", xlab = "LameLeg", col = "blue", outline = F)
boxplot(D$W ~ D$lameLeg, main = "", ylab = "", xlab = "", col = "red", add = TRUE, outline = F)

# Ikke relevant. Symmetry scores ud fra horse
x <- rep(D$horse, 3)
y <- c(D$A, D$S, D$W)
data_norm <- D %>% mutate_at(c('A'), ~(scale(.) %>% as.vector))
z <- rep(c("A","S","W"), c(85,85,85))
data1 <- data.frame(x, z ,  y)

ggplot(data1, aes(x=x, y=y, fill=z)) + 
  geom_boxplot()

# Ikke relevant. Symmetry scores ud fra horse STANDARDIZED
x <- rep(D$horse, 3)
y <- c(scale(D$A), scale(D$S), scale(D$W))
data_norm <- D %>% mutate_at(c('A'), ~(scale(.) %>% as.vector))
z <- rep(c("A","S","W"), c(85,85,85))
data1 <- data.frame(x, z ,  y)

ggplot(data1, aes(x=x, y=y, fill=z)) + 
  geom_boxplot()
# Each horse difficult to classify if large variance? 


# Data description - Section 2. A/W scores ud fra lameLeg (der er noget at classficiere på)
x <- rep(D$lameLeg, 2)
y <- c(D$A, D$W)
Attributes <- rep(c("A","W"), c(85,85))
data1 <- data.frame(x, Attributes ,  y)

ggplot(data1, aes(x=x, y=y, fill=Attributes)) + 
  geom_boxplot() + scale_y_continuous(limits = c(-0.7, 0.7)) 
# https://r-graph-gallery.com/265-grouped-boxplot-with-ggplot2.html


# Data description - Section 2. PC3/PC4 scores ud fra lameLeg (der er noget at classficiere på)
x <- rep(D$lameLeg, 2)
y <- c(D$pc3, D$pc4)
Attributes <- rep(c("PC3","PC4"), c(85,85))
data1 <- data.frame(x, Attributes ,  y)

ggplot(data1, aes(x=x, y=y, fill=Attributes)) + 
  geom_boxplot() + scale_y_continuous(limits = c(-0.7, 0.7)) 
# https://r-graph-gallery.com/265-grouped-boxplot-with-ggplot2.html

# Data description - Section 2. Combined scores ud fra lameLeg (der er noget at classficiere på)
LameLeg <- rep(D$lameLeg, 4)
Value <- c(D$A, D$W, D$pc3, D$pc4)
Attributes <- rep(c("A","W", "PC3","PC4"), c(85,85,85,85))
data1 <- data.frame(LameLeg, Attributes ,  Value)
data1$Attributes <- factor(data1$Attributes, levels = c("A", "W", "PC3", "PC4"))
plot <- ggplot(data1, aes(x=LameLeg, y=Value, fill=Attributes)) + 
  geom_boxplot() + scale_y_continuous(limits = c(-0.7, 0.7)) 
plot
ggsave(filename = "plot.png", width = 7, height = 4, device='png', dpi=500)

ggsave("test.tiff", units="in", width=5, height=4, dpi=300, compression = 'lzw')
# https://r-graph-gallery.com/265-grouped-boxplot-with-ggplot2.html
tiff("test.tiff", units="in", width=5, height=5, res=300)
# insert ggplot code
dev.off()

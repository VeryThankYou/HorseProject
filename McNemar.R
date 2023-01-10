
names <- c("AW-Both.csv", "AW-PC3PC4.csv", "Baseline-AW.csv", "Baseline-Both.csv", "Baseline-PC3PC4.csv", "PC3PC4-Both.csv")

for(e in names)
{
    D <- read.csv(e, header = T)
    D <- D[, c("int1", "int2")]
    D <- data.matrix(D)
    print(e)
    print(mcnemar.test(D))
}
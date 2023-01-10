
names <- c("AW-Both.csv", "AW-PC3PC4.csv", "Baseline-AW.csv", "Baseline-Both.csv", "Baseline-PC3PC4.csv", "PC3PC4-Both.csv")

for(e in names)
{
    D <- read.csv(e, header = T)
    D <- D[, c("int1", "int2")]
    D <- data.matrix(D)
    print(e)
    print(mcnemar.test(D))
}

print("----- Section 3 -----")
names3 <- c("AW-Both_3.csv", "AW-PC3PC4_3.csv", "Baseline-AW_3.csv", "Baseline-Both_3.csv", "Baseline-PC3PC4_3.csv", "PC3PC4-Both_3.csv")

for(e in names3)
{
    print(e)
    D <- read.csv(e, header = T)
    D <- D[, c("int1", "int2")]
    D <- data.matrix(D)
    
    print(mcnemar.test(D))
}
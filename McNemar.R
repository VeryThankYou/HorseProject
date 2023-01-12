
# Set working directory
setwd("/Users/clarasofiechristiansen/Documents/Github_Desktop/Horseproject")


names <- c("AW-Both.csv", "AW-PC3PC4.csv", "Baseline-AW.csv", "Baseline-Both.csv", "Baseline-PC3PC4.csv", "PC3PC4-Both.csv")

d_section2 <- {}

for(e in names)
{
    D <- read.csv(e, header = T)
    D <- D[, c("int1", "int2")]
    D <- data.matrix(D)
    print(e)
    print(mcnemar.test(D))
    d_section2[e] = mcnemar.test(D)$p.value
}

print("----- Section 3 -----")
names3 <- c("AW_2-AW_3.csv", "PC3PC4_2-PC3PC4_3.csv", "Both_2-Both_3.csv", "Baseline_2-Baseline_3.csv")
d_section3 <- {}

for(e in names3)
{
    print(e)
    D <- read.csv(e, header = T)
    D <- D[, c("int1", "int2")]
    D <- data.matrix(D)
    
    print(mcnemar.test(D))
    d_section3[e] = mcnemar.test(D)$p.value
}


for_export2 <- signif(t(data.frame(d_section2)),3)


write.csv(for_export2, "pvalues_mcnemar_2.csv")

for_export3 <- signif(t(data.frame(d_section3)),3)

write.csv(for_export3, "pvalues_mcnemar_3.csv")


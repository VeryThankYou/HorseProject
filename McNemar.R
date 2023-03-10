


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


# Accuracy (round to 3 significant figures)
acc_dataframe2 <- read.csv("accuracy_section2_horseout.csv", header = T)
acc_dataframe2 <- signif(acc_dataframe2,3)*100
acc_dataframe2[] <- lapply(acc_dataframe2, paste0, '%')
write.csv(acc_dataframe2, "accuracy_procent_section2_horseout.csv")

acc_dataframe3 <- read.csv("accuracy_section3_horseout.csv", header = T)
acc_dataframe3 <- signif(acc_dataframe3,3)*100
acc_dataframe3[] <- lapply(acc_dataframe3, paste0, '%')
write.csv(acc_dataframe3, "accuracy_procent_section3_horseout.csv")


print("----- Section 2-3 collapse -----")
names3collapse <- c("AW_2-AW_3_collapse.csv", "PC3PC4_2-PC3PC4_3_collapse.csv", "Both_2-Both_3_collapse.csv", "Baseline_2-Baseline_3_collapse.csv")
d_section3collapse <- {}

for(e in names3collapse)
{
    print(e)
    D <- read.csv(e, header = T)
    D <- D[, c("int1", "int2")]
    D <- data.matrix(D)
    
    print(mcnemar.test(D))
    d_section3collapse[e] = mcnemar.test(D)$p.value
}


for_export3collapse <- signif(t(data.frame(d_section3collapse)),3)

write.csv(for_export3collapse, "pvalues_mcnemar_3_collapse.csv")


# Accuracy (round to 3 significant figures)
acc_dataframe3collapse <- read.csv("accuracy_section2_horseout_collapse.csv", header = T)
acc_dataframe3collapse <- signif(acc_dataframe3collapse,3)*100
acc_dataframe3collapse[] <- lapply(acc_dataframe3collapse, paste0, '%')
write.csv(acc_dataframe3collapse, "accuracy_procent_section2_horseout_collapse.csv")
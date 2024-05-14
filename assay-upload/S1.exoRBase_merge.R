library(optparse)
library(dplyr)
library(data.table)
library(tibble)
rm(list = ls())

## ========================== functions ==========================

source("./merge_functions.R")

## ========================== binary ==========================

opt <- list()
opt$indir = '../data/'
opt$outdir = '../result_upload/'
opt$item = 'binary'
opt$in_grp1 = 'Benign'
opt$in_grp2 = 'CRC'
opt$out_grp1 = 'Benign'
opt$out_grp2 = 'CRC'

grouAs <- strsplit(opt$in_grp1, ";") %>% unlist()
grouBs <- strsplit(opt$in_grp2, ";") %>% unlist()
  
print(paste0(opt$out_grp1, "_vs_", opt$out_grp2))

if(length(grouAs) == 1){
  full_dataA <- Only1grps(in_dir = opt$indir, grp = grouAs, grp_lab = 0)
}else{
  full_dataA <- Multigrps(in_dir = opt$indir, grp = grouAs, grp_lab = 0)
}

if(length(grouBs) == 1){
  full_dataB <- Only1grps(in_dir = opt$indir, grp = grouBs, grp_lab = 1)
}else{
  full_dataB <- Multigrps(in_dir = opt$indir, grp = grouBs, grp_lab = 1)
}

full_data <- rbind(full_dataA, full_dataB)
print(table(full_data$Group))

## occurrence rate
occ = 0.5
count_dat <-  full_data %>%
  group_by(Group) %>%
  summarise(across(everything(), ~ mean(. != 0, na.rm = TRUE)))  %>%
  t() %>%
  data.frame()
count_dat <- count_dat[-1, ]
count_dat_fil <- count_dat[rowSums(count_dat > occ) > 0, ]

## wilcox test
full_data2 <- full_data[, c("Group", rownames(count_dat_fil))]
gene_num = ncol(full_data2) - 1
genes = colnames(full_data2)[-1]
pvalues = matrix(NA, nrow = gene_num, ncol = 5) %>% data.frame()
colnames(pvalues) = c("protein", "pvalue", "mean_ILD", "mean_nonILD", "fc")
for(i in 1:gene_num){
  gene = genes[i]
  dat <- data.frame(Exp = full_data2[, gene], Group = full_data2$Group)
  dat$Group <- factor(dat$Group)
  dat$Exp <- as.numeric(dat$Exp)
  
  tmp =  dat %>% group_by(Group) %>% summarise_all(mean)
  mean_1 = tmp[match(1,tmp$Group),2] %>% as.numeric()
  mean_0 = tmp[match(0,tmp$Group),2] %>% as.numeric()
  fc = mean_1 /  mean_0
  res = wilcox.test(dat$Exp ~ dat$Group, alternative = c("two.sided"))

  pvalues[i,1] = gene
  pvalues[i,2] = res[["p.value"]]
  pvalues[i,3] = mean_1
  pvalues[i,4] = mean_0
  pvalues[i,5] = fc
}
pvalues$padj = p.adjust(pvalues$pvalue, "BH")
pvalues2 = pvalues %>% filter(padj < 0.05)
print(nrow(pvalues2))
full_data_fin <- full_data2[, c("Group", pvalues2$protein)]
write.csv(full_data_fin, paste0(opt$outdir, "exoRBase_", opt$out_grp1, "_vs_", opt$out_grp2, "_merge_full_data.csv"),
          quote = F, row.names = T)
  
## ========================== multiple ==========================
opt$item = 'multi'
opt$in_grp1 = 'Healthy'
opt$in_grp2 = "Benign"
opt$in_grp3 = "OV"

grouAs <- strsplit(opt$in_grp1, ";") %>% unlist()
grouBs <- strsplit(opt$in_grp2, ";") %>% unlist()
grouCs <- strsplit(opt$in_grp3, ";") %>% unlist()

full_dataA <- Only1grps(in_dir = opt$indir, grp = grouAs, grp_lab = 0)
full_dataB <- Only1grps(in_dir = opt$indir, grp = grouBs, grp_lab = 1)
full_dataC <- Only1grps(in_dir = opt$indir, grp = grouCs, grp_lab = 2)
full_data <- rbind(full_dataA, full_dataB)
full_data <- rbind(full_data, full_dataC)
print(table(full_data$Group))

## occurrence rate
occ = 0.5
count_dat <-  full_data %>%
  group_by(Group) %>%
  summarise(across(everything(), ~ mean(. != 0, na.rm = TRUE)))  %>%
  t() %>%
  data.frame()
count_dat <- count_dat[-1, ]
count_dat_fil <- count_dat[rowSums(count_dat > occ) > 0, ]

## wilcox test
full_data2 <- full_data[, c("Group", rownames(count_dat_fil))]
gene_num = ncol(full_data2) - 1
genes = colnames(full_data2)[-1]
pvalues = matrix(NA, nrow = gene_num, ncol = 5) %>% data.frame()
colnames(pvalues) = c("protein", "pvalue", "mean_ILD", "mean_nonILD", "fc")
for(i in 1:gene_num){
  gene = genes[i]
  dat <- data.frame(Exp = full_data2[, gene], Group = full_data2$Group)
  dat$Group <- factor(dat$Group)
  dat$Exp <- as.numeric(dat$Exp)
  
  tmp =  dat %>% group_by(Group) %>% summarise_all(mean)
  mean_0 = tmp[match(0,tmp$Group),2] %>% as.numeric()
  mean_1 = tmp[match(1,tmp$Group),2] %>% as.numeric()
  mean_2 = tmp[match(2,tmp$Group),2] %>% as.numeric()
  fc = max(mean_0, mean_1, mean_2) / min(mean_0, mean_1, mean_2)
  
  # res <- aov(value ~ group, data = data)
  res <- kruskal.test(Exp ~ Group, data = dat)
  
  pvalues[i,1] = gene
  pvalues[i,2] = res[["p.value"]]
  pvalues[i,3] = max(mean_0, mean_1, mean_2)
  pvalues[i,4] = min(mean_0, mean_1, mean_2)
  pvalues[i,5] = fc
}
pvalues$padj = p.adjust(pvalues$pvalue, "BH")
pvalues2 = pvalues %>% filter(padj < 0.05)
print(nrow(pvalues2))
full_data_fin <- full_data2[, c("Group", pvalues2$protein)]
write.csv(full_data_fin, paste0(opt$outdir, "exoRBase_", "multiple", "_merge_full_data.csv"), 
          quote = F, row.names = T)

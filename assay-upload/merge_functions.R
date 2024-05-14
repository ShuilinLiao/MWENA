Only1grps <- function(in_dir, grp, grp_lab){
  
  anno_dir = "../data/"
  longRNAs_anno <- fread(paste0(anno_dir, "longRNAs_anno.txt"), sep = "\t", header = T) %>% 
    dplyr::select(c("Gene symbol", "Gene type", "Full name"))
  colnames(longRNAs_anno) <- c("GeneSymbol", "GeneType", "FullName") 
  longRNAs_anno <- longRNAs_anno %>% 
    filter(GeneType == "protein coding gene") %>% 
    filter(!is.na(FullName))
  
  grp_long <- paste0(in_dir, grp, '_longRNAs.txt')
  longRNA_grp <- read.table(grp_long, header = TRUE, sep = "\t", row.names = 1) %>% 
    t() %>% data.frame()
  
  int_genes <- intersect(longRNAs_anno$GeneSymbol, colnames(longRNA_grp))
  
  full_data <- longRNA_grp[, int_genes]
  full_data$Group <- grp_lab
  full_data <- full_data[, c("Group", setdiff(names(full_data), "Group"))]
  return(full_data)
}

Multigrps <- function(in_dir, grp, grp_lab, longRNAs_anno = longRNAs_anno){
  anno_dir = "../data/"
  longRNAs_anno <- fread(paste0(anno_dir, "longRNAs_anno.txt"), sep = "\t", header = T) %>% 
    dplyr::select(c("Gene symbol", "Gene type", "Full name"))
  colnames(longRNAs_anno) <- c("GeneSymbol", "GeneType", "FullName") 
  longRNAs_anno <- longRNAs_anno %>% 
    filter(GeneType == "protein coding gene") %>% 
    filter(!is.na(FullName))
  
  longRNA_grp <- read.table(paste0(in_dir, grp[1], '_longRNAs.txt'), header = TRUE, sep = "\t", row.names = 1)
  
  for(i in 2:length(grp)){
    tmp_long_dat <- read.table(paste0(in_dir, grp[i], '_longRNAs.txt'), header = TRUE, sep = "\t", row.names = 1)
    longRNA_grp <- cbind(longRNA_grp, tmp_long_dat)
  }
  
  longRNA_grp <- longRNA_grp %>% t() %>% data.frame()
  int_genes <- intersect(longRNAs_anno$GeneSymbol, colnames(longRNA_grp))
  full_data <- longRNA_grp[, int_genes]
  full_data$Group <- grp_lab
  full_data <- full_data[, c("Group", setdiff(names(full_data), "Group"))]
  return(full_data)
}
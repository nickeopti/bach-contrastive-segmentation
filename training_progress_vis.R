library(tidyverse)

setwd("~/Documents/bach")
data <- read_csv("concat.csv", col_types = "ffdd")

plot <- function(data) {
  data %>% filter(dice > 0) %>%
    ggplot() +
    aes(x=epoch, y=dice, group=file, colour=group) +
    scale_y_continuous(breaks = round(seq(min(data$dice), max(data$dice)+0.05, by = 0.05),2)) +
    theme_bw() +
    theme(legend.position = "top") +
    xlab("Epoch") + 
    ylab("Validation mean DICE score") +
    geom_line()
}
save <- function(filename) {
  ggsave(filename, width=20, height=10, units="cm", dpi=600)
}

# data %>% plot

data %>% filter(group %in% c("Cross Entropy", "Mean Squared Error", "Featurisation")) %>%
  mutate(group = recode_factor(group, "Cross Entropy"="Binary Cross Entropy")) %>%
  plot + labs(colour="Similarity Measure")
save("plots/similarity_measure.png")

data %>% filter(group %in% c("Cross Entropy", "Top-k", "Probabilistic")) %>%
  mutate(group = recode_factor(group, "Cross Entropy"="Entropy")) %>%
  plot + labs(colour="Sampling Procedure")
save("plots/sampling_procedure.png")

data %>% filter(group %in% c("Cross Entropy", "Gamma = 0", "2 Channels")) %>%
  mutate(group = recode_factor(group, "Cross Entropy"="Baseline")) %>%
  plot + labs(colour="Variation")
save("plots/variants.png")

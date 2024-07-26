library(dplyr)
library(ggpubr)
library(data.table)
library(gridExtra)
library(extrafont)

font_import(pattern = "Arial", prompt = FALSE)
loadfonts(device = "win")
cwd <- getwd()
base_path <- normalizePath(file.path(cwd))
mode <- 'fewshot'
path <- file.path(base_path, sprintf('/result/fig3/Systematic_benchmark_results_%s.csv', mode))
perform_all <- fread(path)
current_names <- names(perform_all)
current_names[current_names == "cosine_similarity"] <- "Cosine similarity"
current_names[current_names == "mse"] <- "Mean squared error"
current_names[current_names == "mae"] <- "Mean absolute error"
current_names[current_names == "pearson"] <- "Pearson correlation coefficient"

names(perform_all) <- current_names
perform_all <- perform_all %>% mutate(index = row_number())

perform_summary <- perform_all %>%
  group_by(methods, dataset) %>%
  summarise(across(c(mse, cosine_similarity), mean, na.rm = TRUE), .groups = 'drop')

plot_fig3 <- function(perform_all, mode, x_value, y_value) {
  model_labels <- c('scMM', 'CMAE', 'MultiVI', 'BABEL', 'Seurat', 'scMoGNN', 'cTP-net', 'sciPENN', 'scTranslator-scratch', 'scTranslator')
  num_model <- length(model_labels)
  colors <- scales::hue_pal()(num_model)
  names(colors) <- model_labels

  for (metric in c("Cosine similarity", "Pearson correlation coefficient")) {  # c("Mean squared error", "Mean absolute error")
    perform_all$methods <- factor(perform_all$methods, levels = model_labels)
    p <- ggboxplot(perform_all, x = "methods", y = metric, color = "methods", palette = colors) +
      geom_jitter(aes(color = methods), width = 0.2, height = 0, alpha = 0.6) +
      geom_boxplot(aes(color = methods)) +
      facet_wrap(~dataset, nrow = 1) +
      labs(y = metric, x = NULL) +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            panel.grid.major.y = element_line(size = 0.5, linetype = 'dashed', color = 'gray80'),
            panel.grid.major.x = element_blank(),
            panel.border = element_rect(color = "black", fill = NA, size = 0.5),
            strip.background = element_rect(color = "black", fill = "gray90", size = 0.5),
            strip.text = element_text(size = 10),
            text = element_text(size = 11),
            legend.position = "none",
            panel.spacing = unit(0.5, "lines"),
            axis.line.y = element_blank(),
            axis.line.x = element_blank()) +
      scale_y_continuous(limits = c(0, 0.98), breaks = seq(0, 1, by = 0.1))
    p <- p + theme(
      panel.border = element_rect(color = "black", fill = NA, size = 0.5),
      strip.background = element_rect(color = "black", fill = "gray90", size = 0.5)
    )
    print(p)

    }
}

x_value = 12
y_value = 3
windows(width = x_value, height = y_value)
plot_fig3(perform_all, mode, x_value, y_value)

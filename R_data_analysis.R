# ============================================================================
# COMPLETE ANALYSIS: SPEECH CATEGORIES & MOVEMENT FEATURES
# ============================================================================

# Load required packages
library(MASS)        # LDA
library(nnet)        # Multinomial logistic regression
library(lme4)        # Mixed effects models
library(car)         # ANOVA
library(ggplot2)     # Visualizations
library(gridExtra)   # Multiple plots
library(corrplot)    # Correlation plots
library(effsize)     # Effect sizes
library(caret)       # Cross-validation
library(reshape2)    # Data reshaping
library(scales)      # Scaling functions
library(pheatmap)    # Heatmaps

# Set output directory
output_dir <- "/Users/rachelpapirmeister/Downloads"
setwd(output_dir)

# Load data (renamed from episodes_for_lta.csv)
df <- read.csv("NLP_features_for_LTA.csv")

# Convert categorical variables
df$speech_category <- factor(df$speech_category,
                             levels = c("exploratory", "confirmatory", "exploitative"))
df$participant_id <- factor(df$participant_id)
df$game <- factor(df$game)

# Define movement features
movement_features <- c("movement_entropy", "direction_changes", "repeated_sequences",
                      "unique_positions", "num_moves", "num_revisits",
                      "prop_direction_changes")

# ============================================================================
# 1. LINEAR DISCRIMINANT ANALYSIS (LDA)
# ============================================================================

cat("\n", rep("=", 60), "\n")
cat("1. LINEAR DISCRIMINANT ANALYSIS\n")
cat(rep("=", 60), "\n\n")

# Prepare data for LDA
lda_data <- df[complete.cases(df[, c("speech_category", movement_features)]), ]

# Fit LDA model
lda_model <- lda(speech_category ~ movement_entropy + direction_changes +
                 repeated_sequences + unique_positions + num_moves +
                 num_revisits + prop_direction_changes,
                 data = lda_data)

# Predictions
lda_pred <- predict(lda_model)

# Confusion matrix
conf_matrix <- table(Predicted = lda_pred$class, Actual = lda_data$speech_category)

# Classification accuracy
accuracy_overall <- sum(diag(conf_matrix)) / sum(conf_matrix)
accuracy_per_class <- diag(conf_matrix) / rowSums(conf_matrix)

# Cross-validation
set.seed(123)
train_control <- trainControl(method = "cv", number = 10)
cv_model <- train(speech_category ~ movement_entropy + direction_changes +
                  repeated_sequences + unique_positions + num_moves +
                  num_revisits + prop_direction_changes,
                  data = lda_data, method = "lda", trControl = train_control)

# Save LDA results
sink("lda_results.txt")
cat("LINEAR DISCRIMINANT ANALYSIS RESULTS\n")
cat(rep("=", 60), "\n\n")

cat("Classification Accuracy:\n")
cat(sprintf("  Overall: %.2f%%\n", accuracy_overall * 100))
cat("\nPer-category accuracy:\n")
for(i in 1:length(accuracy_per_class)) {
  cat(sprintf("  %s: %.2f%%\n", names(accuracy_per_class)[i],
              accuracy_per_class[i] * 100))
}

cat("\n\nConfusion Matrix:\n")
print(conf_matrix)

cat("\n\nLDA Coefficients (Feature Loadings):\n")
print(lda_model$scaling)

cat("\n\nCross-Validation Accuracy (10-fold):\n")
cat(sprintf("  Mean: %.2f%%\n", cv_model$results$Accuracy * 100))
cat(sprintf("  SD: %.2f%%\n", cv_model$results$AccuracySD * 100))

cat("\n\nPrior Probabilities:\n")
print(lda_model$prior)

cat("\n\nGroup Means:\n")
print(lda_model$means)

sink()

cat("LDA results saved to lda_results.txt\n")

# LDA Biplot visualization
png("lda_biplot.png", width = 10, height = 8, units = "in", res = 300)
plot(lda_pred$x[,1], lda_pred$x[,2],
     col = c("red", "blue", "green")[lda_data$speech_category],
     pch = 19, xlab = "LD1", ylab = "LD2",
     main = "LDA: Speech Categories in Discriminant Space")
legend("topright", legend = levels(df$speech_category),
       col = c("red", "blue", "green"), pch = 19)
dev.off()

cat("LDA biplot saved\n")

# ============================================================================
# 2. MODEL COMPARISON (2 vs 3 categories)
# ============================================================================

cat("\n", rep("=", 60), "\n")
cat("2. MODEL COMPARISON (2 vs 3 categories)\n")
cat(rep("=", 60), "\n\n")

# Create 2-category version (remove confirmatory)
df_2cat <- df[df$speech_category != "confirmatory", ]
df_2cat$speech_category <- droplevels(df_2cat$speech_category)

# Fit 2-category model
lda_2cat <- lda(speech_category ~ movement_entropy + direction_changes +
                repeated_sequences + unique_positions + num_moves +
                num_revisits + prop_direction_changes,
                data = df_2cat)

# Fit 3-category model (full data)
lda_3cat <- lda(speech_category ~ movement_entropy + direction_changes +
                repeated_sequences + unique_positions + num_moves +
                num_revisits + prop_direction_changes,
                data = lda_data)

# 2-category multinomial
multi_2cat <- multinom(speech_category ~ movement_entropy + direction_changes +
                       repeated_sequences + unique_positions + num_moves +
                       num_revisits + prop_direction_changes,
                       data = df_2cat, trace = FALSE)

# 3-category multinomial
multi_3cat <- multinom(speech_category ~ movement_entropy + direction_changes +
                       repeated_sequences + unique_positions + num_moves +
                       num_revisits + prop_direction_changes,
                       data = lda_data, trace = FALSE)

# Calculate BIC and AIC
bic_2cat <- BIC(multi_2cat)
bic_3cat <- BIC(multi_3cat)
aic_2cat <- AIC(multi_2cat)
aic_3cat <- AIC(multi_3cat)

# Likelihood ratio test (chi-squared)
lr_test_stat <- -2 * (logLik(multi_2cat)[1] - logLik(multi_3cat)[1])
df_diff <- attr(logLik(multi_3cat), "df") - attr(logLik(multi_2cat), "df")
p_value <- pchisq(lr_test_stat, df_diff, lower.tail = FALSE)

# Save model comparison results
sink("model_comparison.txt")
cat("MODEL COMPARISON: 2 vs 3 CATEGORIES\n")
cat(rep("=", 60), "\n\n")

cat("2-Category Model (Exploratory + Exploitative only):\n")
cat(sprintf("  N observations: %d\n", nrow(df_2cat)))
cat(sprintf("  AIC: %.2f\n", aic_2cat))
cat(sprintf("  BIC: %.2f\n", bic_2cat))

cat("\n3-Category Model (Exploratory + Confirmatory + Exploitative):\n")
cat(sprintf("  N observations: %d\n", nrow(lda_data)))
cat(sprintf("  AIC: %.2f\n", aic_3cat))
cat(sprintf("  BIC: %.2f\n", bic_3cat))

cat("\n\nModel Selection:\n")
cat(sprintf("  dAIC (3-cat - 2-cat): %.2f\n", aic_3cat - aic_2cat))
cat(sprintf("  dBIC (3-cat - 2-cat): %.2f\n", bic_3cat - bic_2cat))

if(bic_3cat < bic_2cat) {
  cat("\n  *** 3-category model preferred (lower BIC) ***\n")
} else {
  cat("\n  *** 2-category model preferred (lower BIC) ***\n")
}

cat("\n\nLikelihood Ratio Test (Chi-squared):\n")
cat(sprintf("  Chi-sq = %.2f\n", lr_test_stat))
cat(sprintf("  df = %d\n", df_diff))
cat(sprintf("  p-value = %.4f\n", p_value))

if(p_value < 0.05) {
  cat("  *** 3-category model significantly better (p < 0.05) ***\n")
} else {
  cat("  *** No significant improvement with 3 categories ***\n")
}

cat("\n\nInterpretation:\n")
cat("  Lower BIC/AIC = better model fit\n")
cat("  BIC penalizes complexity more than AIC\n")
cat("  Negative dBIC means 3-category model is better\n")

sink()

cat("Model comparison saved to model_comparison.txt\n")

# ============================================================================
# 3. MULTINOMIAL LOGISTIC REGRESSION
# ============================================================================

cat("\n", rep("=", 60), "\n")
cat("3. MULTINOMIAL LOGISTIC REGRESSION\n")
cat(rep("=", 60), "\n\n")

# Fit multinomial logistic regression
multinom_model <- multinom(speech_category ~ movement_entropy + direction_changes +
                           repeated_sequences + unique_positions + num_moves +
                           num_revisits + prop_direction_changes,
                           data = lda_data, trace = FALSE)

# Get coefficients and standard errors
coefs <- summary(multinom_model)$coefficients
std_errors <- summary(multinom_model)$standard.errors

# Calculate z-scores and p-values
z_scores <- coefs / std_errors
p_values <- 2 * (1 - pnorm(abs(z_scores)))

# Calculate odds ratios
odds_ratios <- exp(coefs)

# Pseudo R-squared (McFadden)
null_model <- multinom(speech_category ~ 1, data = lda_data, trace = FALSE)
pseudo_r2 <- 1 - (logLik(multinom_model) / logLik(null_model))

# Save results
sink("logistic_regression.txt")
cat("MULTINOMIAL LOGISTIC REGRESSION RESULTS\n")
cat(rep("=", 60), "\n\n")

cat("Model Fit Statistics:\n")
cat(sprintf("  AIC: %.2f\n", AIC(multinom_model)))
cat(sprintf("  Pseudo R-sq (McFadden): %.4f\n", as.numeric(pseudo_r2)))
cat(sprintf("  Log-Likelihood: %.2f\n", logLik(multinom_model)))

cat("\n\nCoefficients (log-odds):\n")
print(round(coefs, 4))

cat("\n\nOdds Ratios:\n")
print(round(odds_ratios, 4))

cat("\n\nP-values:\n")
print(round(p_values, 4))

cat("\n\nSignificant predictors (p < 0.05):\n")
for(i in 1:nrow(p_values)) {
  cat(sprintf("\n%s vs Exploratory:\n", rownames(p_values)[i]))
  sig_vars <- which(p_values[i,] < 0.05)
  if(length(sig_vars) > 0) {
    for(j in sig_vars) {
      cat(sprintf("  %s: OR=%.3f, p=%.4f\n",
                  colnames(p_values)[j], odds_ratios[i,j], p_values[i,j]))
    }
  } else {
    cat("  None\n")
  }
}

sink()

cat("Logistic regression results saved\n")

# ============================================================================
# 4. STATISTICAL TESTS (ANOVA)
# ============================================================================

cat("\n", rep("=", 60), "\n")
cat("4. STATISTICAL TESTS (ANOVA & POST-HOC)\n")
cat(rep("=", 60), "\n\n")

sink("anova_results.txt")
cat("ONE-WAY ANOVA & POST-HOC TESTS\n")
cat(rep("=", 60), "\n\n")

for(feature in movement_features) {
  cat("\n", rep("-", 60), "\n")
  cat(toupper(gsub("_", " ", feature)), "\n")
  cat(rep("-", 60), "\n\n")

  # Get data
  exploratory <- df[df$speech_category == "exploratory", feature]
  confirmatory <- df[df$speech_category == "confirmatory", feature]
  exploitative <- df[df$speech_category == "exploitative", feature]

  # Remove NAs
  exploratory <- exploratory[!is.na(exploratory)]
  confirmatory <- confirmatory[!is.na(confirmatory)]
  exploitative <- exploitative[!is.na(exploitative)]

  # Descriptive statistics
  cat("Descriptive Statistics:\n")
  cat(sprintf("  Exploratory:   M=%.3f, SD=%.3f, n=%d\n",
              mean(exploratory), sd(exploratory), length(exploratory)))
  cat(sprintf("  Confirmatory:  M=%.3f, SD=%.3f, n=%d\n",
              mean(confirmatory), sd(confirmatory), length(confirmatory)))
  cat(sprintf("  Exploitative:  M=%.3f, SD=%.3f, n=%d\n",
              mean(exploitative), sd(exploitative), length(exploitative)))

  # One-way ANOVA
  formula_str <- paste(feature, "~ speech_category")
  anova_result <- aov(as.formula(formula_str), data = df)
  anova_summary <- summary(anova_result)

  # Extract F statistic and p-value (FIXED SYNTAX)
  f_stat <- anova_summary[[1]]$"F value"[1]
  p_val <- anova_summary[[1]]$"Pr(>F)"[1]
  df1 <- anova_summary[[1]]$Df[1]
  df2 <- anova_summary[[1]]$Df[2]

  cat("\nOne-way ANOVA:\n")
  cat(sprintf("  F(%d, %d) = %.4f\n", df1, df2, f_stat))
  cat(sprintf("  p-value = %.4f\n", p_val))

  if(p_val < 0.001) {
    cat("  Result: *** HIGHLY SIGNIFICANT (p < 0.001)\n")
  } else if(p_val < 0.01) {
    cat("  Result: ** SIGNIFICANT (p < 0.01)\n")
  } else if(p_val < 0.05) {
    cat("  Result: * SIGNIFICANT (p < 0.05)\n")
  } else {
    cat("  Result: NOT SIGNIFICANT (p >= 0.05)\n")
  }

  # Effect size (eta-squared) - FIXED SYNTAX
  ss_between <- anova_summary[[1]]$"Sum Sq"[1]
  ss_total <- sum(anova_summary[[1]]$"Sum Sq")
  eta_squared <- ss_between / ss_total

  cat(sprintf("\nEffect Size (eta-sq): %.4f", eta_squared))
  if(eta_squared < 0.01) {
    cat(" (small)\n")
  } else if(eta_squared < 0.06) {
    cat(" (medium)\n")
  } else {
    cat(" (large)\n")
  }

  # Post-hoc Tukey HSD if significant
  if(p_val < 0.05) {
    cat("\nPost-hoc Tukey HSD:\n")
    tukey_result <- TukeyHSD(anova_result)
    print(tukey_result)

    # Cohen's d for pairwise comparisons
    cat("\nPairwise Effect Sizes (Cohen's d):\n")

    d_conf_exp <- cohen.d(confirmatory, exploratory)$estimate
    d_expl_exp <- cohen.d(exploitative, exploratory)$estimate
    d_expl_conf <- cohen.d(exploitative, confirmatory)$estimate

    cat(sprintf("  Confirmatory vs Exploratory: d=%.3f\n", d_conf_exp))
    cat(sprintf("  Exploitative vs Exploratory: d=%.3f\n", d_expl_exp))
    cat(sprintf("  Exploitative vs Confirmatory: d=%.3f\n", d_expl_conf))
  }

  cat("\n")
}

sink()

cat("ANOVA results saved\n")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

cat("\n", rep("=", 60), "\n")
cat("5. CREATING VISUALIZATIONS\n")
cat(rep("=", 60), "\n\n")

# Confusion Matrix Heatmap
png("confusion_matrix_heatmap.png", width = 8, height = 6, units = "in", res = 300)
conf_matrix_prop <- prop.table(conf_matrix, margin = 2)
conf_melted <- melt(conf_matrix)
ggplot(conf_melted, aes(x = Actual, y = Predicted, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme_minimal() +
  labs(title = "Confusion Matrix: LDA Classification",
       x = "Actual Category", y = "Predicted Category") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
dev.off()

cat("Confusion matrix heatmap saved\n")

# Feature Importance (LDA coefficients)
png("feature_importance.png", width = 10, height = 6, units = "in", res = 300)
ld1_loadings <- abs(lda_model$scaling[,1])
ld2_loadings <- if(ncol(lda_model$scaling) > 1) abs(lda_model$scaling[,2]) else rep(0, length(ld1_loadings))

loadings_df <- data.frame(
  Feature = names(ld1_loadings),
  LD1 = ld1_loadings,
  LD2 = ld2_loadings
)

loadings_melted <- melt(loadings_df, id.vars = "Feature")

ggplot(loadings_melted, aes(x = reorder(Feature, value), y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(title = "Feature Importance (LDA Loadings)",
       x = "Features", y = "Absolute Loading") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
dev.off()

cat("Feature importance plot saved\n")

# Boxplots with significance
png("boxplots_by_category.png", width = 15, height = 10, units = "in", res = 300)
par(mfrow = c(2, 3))

for(feature in movement_features[1:6]) {
  boxplot(as.formula(paste(feature, "~ speech_category")), data = df,
          main = gsub("_", " ", toupper(feature)),
          xlab = "Speech Category", ylab = feature,
          col = c("red", "blue", "green"))
}
dev.off()

cat("Boxplots saved\n")

# Violin plots
png("violin_plots.png", width = 15, height = 5, units = "in", res = 300)
p1 <- ggplot(df, aes(x = speech_category, y = movement_entropy, fill = speech_category)) +
  geom_violin() + theme_minimal() +
  labs(title = "Movement Entropy", x = "", y = "Entropy")

p2 <- ggplot(df, aes(x = speech_category, y = prop_direction_changes, fill = speech_category)) +
  geom_violin() + theme_minimal() +
  labs(title = "Direction Changes", x = "", y = "Proportion")

p3 <- ggplot(df, aes(x = speech_category, y = repeated_sequences, fill = speech_category)) +
  geom_violin() + theme_minimal() +
  labs(title = "Repeated Sequences", x = "", y = "Count")

grid.arrange(p1, p2, p3, ncol = 3)
dev.off()

cat("Violin plots saved\n")

# Feature Profile Heatmap
png("feature_profile_heatmap.png", width = 10, height = 4, units = "in", res = 300)
category_means <- aggregate(. ~ speech_category,
                            data = df[, c("speech_category", movement_features)],
                            FUN = mean, na.rm = TRUE)

# Normalize
category_means_scaled <- category_means
for(i in 2:ncol(category_means_scaled)) {
  category_means_scaled[,i] <- rescale(category_means[,i], to = c(0, 1))
}

rownames(category_means_scaled) <- category_means_scaled$speech_category
category_means_scaled$speech_category <- NULL

pheatmap(as.matrix(category_means_scaled),
         cluster_rows = FALSE, cluster_cols = FALSE,
         display_numbers = TRUE, number_format = "%.2f",
         main = "Movement Feature Profiles by Speech Category",
         color = colorRampPalette(c("red", "yellow", "green"))(50))
dev.off()

cat("Feature profile heatmap saved\n")

# Summary Statistics File
sink("summary_statistics.txt")
cat("SUMMARY STATISTICS BY CATEGORY\n")
cat(rep("=", 60), "\n\n")

cat(sprintf("Total episodes: %d\n", nrow(df)))
cat(sprintf("Participants: %d\n", length(unique(df$participant_id))))
cat("\nSpeech Category Distribution:\n")
print(table(df$speech_category))

cat("\n\nMovement Features by Category:\n")
cat(rep("=", 60), "\n")

for(category in c("exploratory", "confirmatory", "exploitative")) {
  cat(sprintf("\n%s:\n", toupper(category)))
  subset_data <- df[df$speech_category == category, ]

  for(feature in movement_features) {
    vals <- subset_data[[feature]][!is.na(subset_data[[feature]])]
    cat(sprintf("  %s: M=%.3f, SD=%.3f, n=%d\n",
                feature, mean(vals), sd(vals), length(vals)))
  }
}

sink()

cat("Summary statistics saved\n")

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

cat("\n", rep("=", 60), "\n")
cat("ALL ANALYSES COMPLETE!\n")
cat(rep("=", 60), "\n\n")
cat("Files saved to:", output_dir, "\n\n")
cat("Results files:\n")
cat("  - lda_results.txt\n")
cat("  - model_comparison.txt\n")
cat("  - logistic_regression.txt\n")
cat("  - anova_results.txt\n")
cat("  - summary_statistics.txt\n\n")
cat("Visualization files:\n")
cat("  - lda_biplot.png\n")
cat("  - confusion_matrix_heatmap.png\n")
cat("  - feature_importance.png\n")
cat("  - boxplots_by_category.png\n")
cat("  - violin_plots.png\n")
cat("  - feature_profile_heatmap.png\n\n")
cat("Analysis pipeline complete!\n")

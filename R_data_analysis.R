# ============================================================================
# COMPLETE ANALYSIS: SPEECH CATEGORIES & MOVEMENT FEATURES
# ============================================================================

# Load required packages
library(MASS)        # LDA
library(nnet)        # Multinomial logistic regression
library(lme4)        # Mixed effects models
library(car)         # ANOVA / Type III tests
library(ggplot2)     # Visualizations
library(gridExtra)   # Multiple plots
library(corrplot)    # Correlation plots
library(effsize)     # Effect sizes
library(caret)       # Cross-validation
library(reshape2)    # Data reshaping
library(scales)      # Scaling functions
library(pheatmap)    # Heatmaps
library(emmeans)     # Post-hoc pairwise comparisons

# Set output directory
output_dir <- "/Users/rachelpapirmeister/Downloads"
setwd(output_dir)

# Prompt user to select the data file
cat("Please select the NLP_features_for_LTA.csv file...\n")
data_file <- file.choose()
cat(sprintf("Selected: %s\n\n", data_file))

# Load data
df <- read.csv(data_file)

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

# Create 2-category version by MERGING confirmatory into exploitative
# This keeps the same sample size for valid comparison
df_2cat <- lda_data
df_2cat$speech_category_2 <- as.character(df_2cat$speech_category)
df_2cat$speech_category_2[df_2cat$speech_category_2 == "confirmatory"] <- "exploitative"
df_2cat$speech_category_2 <- factor(df_2cat$speech_category_2,
                                     levels = c("exploratory", "exploitative"))

cat(sprintf("Sample size: %d (same for both models)\n", nrow(lda_data)))
cat("2-category approach: Confirmatory merged into Exploitative\n\n")

# Fit 2-category multinomial (on merged data)
multi_2cat <- multinom(speech_category_2 ~ movement_entropy + direction_changes +
                       repeated_sequences + unique_positions + num_moves +
                       num_revisits + prop_direction_changes,
                       data = df_2cat, trace = FALSE)

# Fit 3-category multinomial (original categories)
multi_3cat <- multinom(speech_category ~ movement_entropy + direction_changes +
                       repeated_sequences + unique_positions + num_moves +
                       num_revisits + prop_direction_changes,
                       data = lda_data, trace = FALSE)

# Calculate BIC and AIC
bic_2cat <- BIC(multi_2cat)
bic_3cat <- BIC(multi_3cat)
aic_2cat <- AIC(multi_2cat)
aic_3cat <- AIC(multi_3cat)

# Cross-validation comparison (more robust)
set.seed(123)
cv_control <- trainControl(method = "cv", number = 10)

cv_2cat <- train(speech_category_2 ~ movement_entropy + direction_changes +
                 repeated_sequences + unique_positions + num_moves +
                 num_revisits + prop_direction_changes,
                 data = df_2cat, method = "multinom", trControl = cv_control,
                 trace = FALSE)

cv_3cat <- train(speech_category ~ movement_entropy + direction_changes +
                 repeated_sequences + unique_positions + num_moves +
                 num_revisits + prop_direction_changes,
                 data = lda_data, method = "multinom", trControl = cv_control,
                 trace = FALSE)

# Likelihood ratio test (valid because same N)
# 3-cat is nested in sense that it splits one category
lr_test_stat <- -2 * (logLik(multi_2cat)[1] - logLik(multi_3cat)[1])
df_diff <- attr(logLik(multi_3cat), "df") - attr(logLik(multi_2cat), "df")
p_value <- pchisq(lr_test_stat, df_diff, lower.tail = FALSE)

# Save model comparison results
sink("model_comparison.txt")
cat("MODEL COMPARISON: 2 vs 3 CATEGORIES\n")
cat(rep("=", 60), "\n\n")

cat("Approach: Compare models on SAME data (N = ", nrow(lda_data), ")\n", sep = "")
cat("2-category: Confirmatory merged into Exploitative\n")
cat("3-category: Exploratory, Confirmatory, Exploitative (separate)\n\n")

cat("Category Distribution:\n")
cat("  3-category:\n")
print(table(lda_data$speech_category))
cat("\n  2-category (merged):\n")
print(table(df_2cat$speech_category_2))

cat("\n\n", rep("-", 50), "\n", sep = "")
cat("MODEL FIT STATISTICS\n")
cat(rep("-", 50), "\n\n")

cat("2-Category Model:\n")
cat(sprintf("  AIC: %.2f\n", aic_2cat))
cat(sprintf("  BIC: %.2f\n", bic_2cat))
cat(sprintf("  CV Accuracy: %.2f%% (SD: %.2f%%)\n",
            cv_2cat$results$Accuracy[1] * 100,
            cv_2cat$results$AccuracySD[1] * 100))

cat("\n3-Category Model:\n")
cat(sprintf("  AIC: %.2f\n", aic_3cat))
cat(sprintf("  BIC: %.2f\n", bic_3cat))
cat(sprintf("  CV Accuracy: %.2f%% (SD: %.2f%%)\n",
            cv_3cat$results$Accuracy[1] * 100,
            cv_3cat$results$AccuracySD[1] * 100))

cat("\n\n", rep("-", 50), "\n", sep = "")
cat("MODEL COMPARISON\n")
cat(rep("-", 50), "\n\n")

cat(sprintf("Delta AIC (3-cat - 2-cat): %.2f\n", aic_3cat - aic_2cat))
cat(sprintf("Delta BIC (3-cat - 2-cat): %.2f\n", bic_3cat - bic_2cat))
cat(sprintf("Delta CV Accuracy: %.2f%%\n",
            (cv_3cat$results$Accuracy[1] - cv_2cat$results$Accuracy[1]) * 100))

cat("\nLikelihood Ratio Test:\n")
cat(sprintf("  Chi-sq = %.2f, df = %d, p = %.4f\n", lr_test_stat, df_diff, p_value))

cat("\n\n", rep("-", 50), "\n", sep = "")
cat("INTERPRETATION\n")
cat(rep("-", 50), "\n\n")

cat("Information Criteria (lower = better):\n")
if(bic_3cat < bic_2cat) {
  cat("  BIC favors 3-category model (confirmatory is distinct)\n")
} else {
  cat("  BIC favors 2-category model (confirmatory not distinct)\n")
}

if(aic_3cat < aic_2cat) {
  cat("  AIC favors 3-category model\n")
} else {
  cat("  AIC favors 2-category model\n")
}

cat("\nCross-Validation Accuracy:\n")
if(cv_3cat$results$Accuracy[1] > cv_2cat$results$Accuracy[1]) {
  cat("  3-category model has better predictive accuracy\n")
} else {
  cat("  2-category model has equal or better predictive accuracy\n")
}

cat("\nLikelihood Ratio Test:\n")
if(p_value < 0.05) {
  cat("  Significant (p < .05): 3-category model fits significantly better\n")
  cat("  The 'confirmatory' category captures distinct movement patterns\n")
} else {
  cat("  Not significant (p >= .05): No significant improvement with 3 categories\n")
  cat("  The 'confirmatory' category may not be distinct from exploitative\n")
}

cat("\n\nRecommendation:\n")
# Count votes
votes_3cat <- sum(c(bic_3cat < bic_2cat,
                    aic_3cat < aic_2cat,
                    cv_3cat$results$Accuracy[1] > cv_2cat$results$Accuracy[1],
                    p_value < 0.05))

if(votes_3cat >= 3) {
  cat("  STRONG evidence for 3-category model\n")
} else if(votes_3cat == 2) {
  cat("  MIXED evidence - consider theoretical justification\n")
} else {
  cat("  Evidence favors 2-category model (parsimony)\n")
}

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
# 4. STATISTICAL TESTS (Mixed-Effects Models)
# ============================================================================

cat("\n", rep("=", 60), "\n")
cat("4. STATISTICAL TESTS (MIXED-EFFECTS MODELS)\n")
cat(rep("=", 60), "\n\n")

# Number of tests for Bonferroni correction
n_tests <- length(movement_features)
bonferroni_alpha <- 0.05 / n_tests

cat(sprintf("Bonferroni-corrected alpha: %.4f (for %d tests)\n\n", bonferroni_alpha, n_tests))

sink("mixed_effects_results.txt")
cat("LINEAR MIXED-EFFECTS MODELS\n")
cat(rep("=", 60), "\n\n")

cat("Model: feature ~ speech_category + (1|participant_id)\n")
cat("This accounts for multiple observations per participant.\n\n")

cat(sprintf("Bonferroni-corrected alpha: %.4f (for %d tests)\n", bonferroni_alpha, n_tests))
cat("Note: p-values marked with * are significant at corrected alpha\n\n")

# Store results for summary
results_summary <- data.frame(
  Feature = character(),
  F_value = numeric(),
  p_value = numeric(),
  p_corrected_sig = character(),
  eta_squared = numeric(),
  effect_size = character(),
  stringsAsFactors = FALSE
)

for(feature in movement_features) {
  cat("\n", rep("-", 60), "\n")
  cat(toupper(gsub("_", " ", feature)), "\n")
  cat(rep("-", 60), "\n\n")

  # Get data for descriptives
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

  # Fit mixed-effects model
  formula_str <- paste(feature, "~ speech_category + (1|participant_id)")

  # Try to fit mixed model; fall back to standard ANOVA if it fails
  tryCatch({
    mixed_model <- lmer(as.formula(formula_str), data = df, REML = TRUE)

    # Get ANOVA-style results using car::Anova (Type III)
    anova_result <- Anova(mixed_model, type = 3)

    # Extract statistics
    chisq_stat <- anova_result$Chisq[1]
    p_val <- anova_result$`Pr(>Chisq)`[1]
    df_effect <- anova_result$Df[1]

    cat("\nMixed-Effects Model (Type III Wald Chi-square):\n")
    cat(sprintf("  Chi-sq(%d) = %.4f\n", df_effect, chisq_stat))
    cat(sprintf("  p-value = %.4f\n", p_val))

    # Calculate approximate eta-squared (marginal R-squared approach)
    # Using variance components
    var_fixed <- var(predict(mixed_model, re.form = NA))
    var_random <- as.numeric(VarCorr(mixed_model)$participant_id)
    var_resid <- sigma(mixed_model)^2
    var_total <- var_fixed + var_random + var_resid
    eta_squared <- var_fixed / var_total

  }, error = function(e) {
    # Fall back to standard ANOVA if mixed model fails
    cat("\nNote: Mixed model failed, using standard ANOVA\n")
    formula_str <- paste(feature, "~ speech_category")
    anova_result <- aov(as.formula(formula_str), data = df)
    anova_summary <- summary(anova_result)

    chisq_stat <<- anova_summary[[1]]$"F value"[1]
    p_val <<- anova_summary[[1]]$"Pr(>F)"[1]
    df_effect <<- anova_summary[[1]]$Df[1]

    ss_between <- anova_summary[[1]]$"Sum Sq"[1]
    ss_total <- sum(anova_summary[[1]]$"Sum Sq")
    eta_squared <<- ss_between / ss_total

    cat(sprintf("  F(%d, %d) = %.4f\n", df_effect, anova_summary[[1]]$Df[2], chisq_stat))
    cat(sprintf("  p-value = %.4f\n", p_val))
  })

  # Significance with Bonferroni correction
  if(p_val < 0.001) {
    sig_label <- "***"
  } else if(p_val < 0.01) {
    sig_label <- "**"
  } else if(p_val < 0.05) {
    sig_label <- "*"
  } else {
    sig_label <- "ns"
  }

  bonf_sig <- ifelse(p_val < bonferroni_alpha, "YES", "no")
  cat(sprintf("  Uncorrected: %s | Bonferroni-corrected: %s\n", sig_label, bonf_sig))

  # Effect size (eta-squared) with CORRECTED thresholds
  cat(sprintf("\nEffect Size (eta-sq): %.4f", eta_squared))
  if(eta_squared < 0.01) {
    effect_label <- "negligible"
    cat(" (negligible, < .01)\n")
  } else if(eta_squared < 0.06) {
    effect_label <- "small"
    cat(" (small, .01-.06)\n")
  } else if(eta_squared < 0.14) {
    effect_label <- "medium"
    cat(" (medium, .06-.14)\n")
  } else {
    effect_label <- "large"
    cat(" (large, >= .14)\n")
  }

  # Store for summary
  results_summary <- rbind(results_summary, data.frame(
    Feature = feature,
    F_value = round(chisq_stat, 3),
    p_value = round(p_val, 4),
    p_corrected_sig = bonf_sig,
    eta_squared = round(eta_squared, 4),
    effect_size = effect_label,
    stringsAsFactors = FALSE
  ))

  # Post-hoc tests if significant at Bonferroni level
  if(p_val < bonferroni_alpha) {
    cat("\nPost-hoc pairwise comparisons (emmeans):\n")

    tryCatch({
      library(emmeans)
      emm <- emmeans(mixed_model, ~ speech_category)
      pairs_result <- pairs(emm, adjust = "tukey")
      print(pairs_result)

      # Effect sizes
      cat("\nPairwise Effect Sizes (Cohen's d):\n")
      d_conf_exp <- cohen.d(confirmatory, exploratory)$estimate
      d_expl_exp <- cohen.d(exploitative, exploratory)$estimate
      d_expl_conf <- cohen.d(exploitative, confirmatory)$estimate

      cat(sprintf("  Confirmatory vs Exploratory: d=%.3f\n", d_conf_exp))
      cat(sprintf("  Exploitative vs Exploratory: d=%.3f\n", d_expl_exp))
      cat(sprintf("  Exploitative vs Confirmatory: d=%.3f\n", d_expl_conf))
    }, error = function(e) {
      cat("  (emmeans package required for post-hoc tests)\n")
    })
  } else if(p_val < 0.05) {
    cat("\nNote: Significant at uncorrected alpha but NOT at Bonferroni-corrected level\n")
  }

  cat("\n")
}

# Print summary table
cat("\n", rep("=", 60), "\n")
cat("SUMMARY TABLE\n")
cat(rep("=", 60), "\n\n")
print(results_summary)

cat("\n\nEffect Size Guidelines (eta-squared):\n")
cat("  < .01  = negligible\n")
cat("  .01-.06 = small\n")
cat("  .06-.14 = medium\n")
cat("  >= .14  = large\n")

sink()

cat("Mixed-effects results saved to mixed_effects_results.txt\n")

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
# 6. ORDER EFFECTS ANALYSIS (Two-Way Mixed ANOVA)
# ============================================================================

cat("\n", rep("=", 60), "\n")
cat("6. ORDER EFFECTS ANALYSIS\n")
cat(rep("=", 60), "\n\n")

# Check if order effects data file exists
order_data_file <- file.path(output_dir, "order_effects_data.csv")

if (file.exists(order_data_file)) {
  cat("Loading order effects data...\n")
  order_df <- read.csv(order_data_file)

  # Convert to factors
  order_df$participant_id <- factor(order_df$participant_id)
  order_df$game_type <- factor(order_df$game_type)
  order_df$order_position <- factor(order_df$order_position)

  cat(sprintf("  Participants: %d\n", length(unique(order_df$participant_id))))
  cat(sprintf("  Total observations: %d\n", nrow(order_df)))

  # Descriptive statistics
  cat("\nDescriptive Statistics:\n")
  cat("\nBy Game Type:\n")
  print(aggregate(num_moves ~ game_type, data = order_df,
                  FUN = function(x) c(mean = mean(x), sd = sd(x), n = length(x))))

  cat("\nBy Order Position:\n")
  print(aggregate(num_moves ~ order_position, data = order_df,
                  FUN = function(x) c(mean = mean(x), sd = sd(x), n = length(x))))

  cat("\nBy Game Type x Order Position:\n")
  print(aggregate(num_moves ~ game_type + order_position, data = order_df,
                  FUN = function(x) c(mean = mean(x), sd = sd(x), n = length(x))))

  # Two-Way Mixed ANOVA
  # Game Type = within-subjects, Order Position = between-subjects
  cat("\n\nTwo-Way Mixed ANOVA Results:\n")
  cat(rep("-", 50), "\n")

  # Using ezANOVA if available, otherwise use standard aov
  if (requireNamespace("ez", quietly = TRUE)) {
    library(ez)
    anova_result <- ezANOVA(
      data = order_df,
      dv = num_moves,
      wid = participant_id,
      within = game_type,
      between = order_position,
      type = 3
    )
    print(anova_result)
  } else {
    # Standard ANOVA approach
    anova_model <- aov(num_moves ~ game_type * order_position +
                       Error(participant_id/game_type), data = order_df)
    print(summary(anova_model))
  }

  # Save order effects results
  sink("order_effects_results.txt")
  cat("ORDER EFFECTS ANALYSIS RESULTS\n")
  cat(rep("=", 60), "\n\n")

  cat("Research Question: Does the order in which games were played\n")
  cat("affect performance (num_moves), controlling for game type?\n\n")

  cat("Descriptive Statistics:\n")
  cat(rep("-", 50), "\n")

  cat("\nBy Game Type:\n")
  game_stats <- aggregate(num_moves ~ game_type, data = order_df,
                          FUN = function(x) c(mean = round(mean(x), 2),
                                             sd = round(sd(x), 2),
                                             n = length(x)))
  print(game_stats)

  cat("\nBy Order Position:\n")
  order_stats <- aggregate(num_moves ~ order_position, data = order_df,
                           FUN = function(x) c(mean = round(mean(x), 2),
                                              sd = round(sd(x), 2),
                                              n = length(x)))
  print(order_stats)

  cat("\nBy Game Type x Order Position:\n")
  cross_stats <- aggregate(num_moves ~ game_type + order_position, data = order_df,
                           FUN = function(x) c(mean = round(mean(x), 2),
                                              sd = round(sd(x), 2),
                                              n = length(x)))
  print(cross_stats)

  cat("\n\nTwo-Way Mixed ANOVA:\n")
  cat(rep("-", 50), "\n")
  cat("DV: num_moves\n")
  cat("Within-subjects factor: game_type (Game A vs Game B)\n")
  cat("Between-subjects factor: order_position (1st, 2nd, 3rd)\n\n")

  anova_model <- aov(num_moves ~ game_type * order_position +
                     Error(participant_id/game_type), data = order_df)
  print(summary(anova_model))

  # Interpretation
  cat("\n\nInterpretation Guide:\n")
  cat(rep("-", 50), "\n")
  cat("1. Main effect of ORDER POSITION:\n")
  cat("   - If p < .05: Order significantly affected performance\n")
  cat("   - If p >= .05: Order did NOT significantly affect performance\n")
  cat("     (counterbalancing was effective)\n\n")

  cat("2. Main effect of GAME TYPE:\n")
  cat("   - If p < .05: Performance differed between Game A and B\n")
  cat("   - This is expected and not related to order effects\n\n")

  cat("3. INTERACTION (Game Type x Order Position):\n")
  cat("   - If p < .05: The effect of order depends on which game\n")
  cat("   - If p >= .05: Order affected both games similarly (or not at all)\n")

  sink()

  cat("Order effects results saved to order_effects_results.txt\n")

  # Visualization
  png("order_effects_plot.png", width = 10, height = 6, units = "in", res = 300)
  par(mfrow = c(1, 2))

  # Plot 1: Boxplot by Order Position
  boxplot(num_moves ~ order_position, data = order_df,
          main = "Num Moves by Order Position",
          xlab = "Order Position (1st, 2nd, 3rd)",
          ylab = "Number of Moves",
          col = c("lightblue", "lightgreen", "lightyellow"))

  # Plot 2: Interaction plot
  interaction.plot(order_df$order_position, order_df$game_type, order_df$num_moves,
                   main = "Game Type x Order Interaction",
                   xlab = "Order Position",
                   ylab = "Mean Number of Moves",
                   col = c("red", "blue"),
                   lwd = 2,
                   legend = TRUE)

  dev.off()

  cat("Order effects plot saved\n")

} else {
  cat("Order effects data file not found.\n")
  cat("Run prepare_order_effects_data.py first to generate order_effects_data.csv\n")
}

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
cat("  - mixed_effects_results.txt\n")
cat("  - summary_statistics.txt\n")
cat("  - order_effects_results.txt\n\n")
cat("Visualization files:\n")
cat("  - lda_biplot.png\n")
cat("  - confusion_matrix_heatmap.png\n")
cat("  - feature_importance.png\n")
cat("  - boxplots_by_category.png\n")
cat("  - violin_plots.png\n")
cat("  - feature_profile_heatmap.png\n")
cat("  - order_effects_plot.png\n\n")
cat("Analysis pipeline complete!\n")

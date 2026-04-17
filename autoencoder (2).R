############################################################
## TWO-STAGE NESTED CV (Outer 5-fold, Inner 3-fold)
## + ROBUST PCA / SPCA / IPCA
## + STRONGER AUTOENCODER / HYBRIDS
## + HPC SAFE (clear_session + tryCatch)
## + IPCA diagnostics
## + Confusion matrices (Sub.population and D0)
## + IMPORTANT FIX: standardize embeddings by TRAIN
##
## Examples:
##   Rscript --vanilla nested_job_fixed.R --analysis MAIN_no_ADMIX --method AE
##   Rscript --vanilla nested_job_fixed.R --analysis SUPP_with_ADMIX --method SPCA
##   Rscript --vanilla nested_job_fixed.R --analysis MAIN_no_ADMIX --method PCA-AE
##
## Confusion once (no CV):
##   Rscript --vanilla nested_job_fixed.R --analysis MAIN_no_ADMIX --method AE --confusion_once TRUE
############################################################

rm(list = ls(all = TRUE))
set.seed(123)
options(stringsAsFactors = FALSE)

# =========================
# 0) SETUP
# =========================
workdir <- "/blue/munoz/cazevedo/2026/BBP_Simulation/Test"
setwd(workdir)

required_packages <- c(
  "mixOmics", "keras", "tensorflow", "cluster",
  "caret", "mclust", "aricode", "dplyr", "readr"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}
invisible(lapply(required_packages, install_if_missing))

suppressPackageStartupMessages({
  library(mixOmics)
  library(keras)
  library(tensorflow)
  library(cluster)
  library(caret)
  library(mclust)
  library(aricode)
  library(dplyr)
  library(readr)
})

tf <- tensorflow::tf

try({
  tf$config$threading$set_intra_op_parallelism_threads(1L)
  tf$config$threading$set_inter_op_parallelism_threads(1L)
}, silent = TRUE)

# =========================
# 0.1) ARGS
# =========================
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit == length(args)) return(default)
  args[hit + 1]
}

ANALYSIS_TO_RUN <- get_arg("--analysis", "MAIN_no_ADMIX")
METHOD_TO_RUN   <- get_arg("--method", "PCA")
CONFUSION_ONCE  <- tolower(get_arg("--confusion_once", "FALSE")) %in% c("true", "t", "1", "yes", "y")

cat("\n===============================\n")
cat("ANALYSIS: ", ANALYSIS_TO_RUN, "\n", sep = "")
cat("METHOD  : ", METHOD_TO_RUN, "\n", sep = "")
cat("remove_admix = ", ifelse(ANALYSIS_TO_RUN == "MAIN_no_ADMIX", "TRUE", "FALSE"), "\n", sep = "")
cat("confusion_once = ", CONFUSION_ONCE, "\n", sep = "")
cat("===============================\n\n")

# =========================
# 1) INPUTS
# =========================
data_file   <- "data_snp.txt"
id_col      <- "IID"
label_col   <- "Sub.population"
admix_label <- "ADMIX"

dados <- read.table(data_file, header = TRUE, check.names = FALSE)
stopifnot(id_col %in% names(dados), label_col %in% names(dados))

snp_cols <- setdiff(names(dados), c(id_col, label_col))
stopifnot(length(snp_cols) > 10)

# =========================
# 2) UTILS
# =========================
choose_k_asw <- function(Z, k_min = 2, k_max = 10) {
  d  <- dist(Z)
  hc <- hclust(d, method = "average")
  ks <- k_min:k_max

  asw <- sapply(ks, function(k) {
    cl <- cutree(hc, k)
    sil <- silhouette(cl, d)
    mean(sil[, 3])
  })

  list(k_best = ks[which.max(asw)], ks = ks, asw = asw)
}

upgma_cluster <- function(Z, k) {
  d  <- dist(Z)
  hc <- hclust(d, method = "average")
  as.integer(cutree(hc, k))
}

purity_score <- function(cl, y) {
  tab <- table(cl, y)
  sum(apply(tab, 1, max)) / length(y)
}

cluster_metrics <- function(cl, y) {
  list(
    ARI = mclust::adjustedRandIndex(cl, y),
    NMI = aricode::NMI(cl, y),
    Purity = purity_score(cl, y)
  )
}

relabel_by_majority <- function(cl, y_true) {
  tab <- table(cl, y_true)
  mapping <- apply(tab, 1, function(r) names(which.max(r)))
  y_pred <- mapping[as.character(cl)]
  factor(y_pred, levels = levels(y_true))
}

confusion_tables <- function(cl, y_true, D0 = NULL) {
  yhat <- relabel_by_majority(cl, y_true)

  out <- list(
    confusion_true = table(Pred = yhat, True = y_true),
    metrics_true = cluster_metrics(cl, y_true)
  )

  if (!is.null(D0)) {
    D0f <- as.factor(D0)
    D0hat <- relabel_by_majority(cl, D0f)
    out$confusion_D0 <- table(Pred = D0hat, True = D0f)
    out$metrics_D0 <- list(ARI = mclust::adjustedRandIndex(cl, D0f))
  }

  out
}

scale_by_train <- function(Z_train, Z_val) {
  mu  <- colMeans(Z_train)
  sdv <- apply(Z_train, 2, sd)

  sdv[!is.finite(sdv)] <- 1
  sdv[sdv < 1e-12] <- 1

  Zt <- sweep(Z_train, 2, mu, "-")
  Zv <- sweep(Z_val,   2, mu, "-")

  Zt <- sweep(Zt, 2, sdv, "/")
  Zv <- sweep(Zv, 2, sdv, "/")

  list(Z_train = Zt, Z_val = Zv)
}

eval_split <- function(Z_train, Z_val, y_val, D0_val = NULL, k_min = 2, k_max = 10) {
  sc <- scale_by_train(Z_train, Z_val)
  Z_train <- sc$Z_train
  Z_val   <- sc$Z_val

  k_info <- choose_k_asw(Z_train, k_min = k_min, k_max = k_max)
  k <- k_info$k_best
  cl_val <- upgma_cluster(Z_val, k)

  met <- cluster_metrics(cl_val, y_val)
  ari_d0 <- if (!is.null(D0_val)) mclust::adjustedRandIndex(cl_val, D0_val) else NA_real_

  list(
    k = k,
    ARI_true = met$ARI,
    NMI_true = met$NMI,
    Purity_true = met$Purity,
    ARI_D0 = ari_d0
  )
}

compute_D0 <- function(X_scaled, k_min = 2, k_max = 10) {
  k_info <- choose_k_asw(X_scaled, k_min = k_min, k_max = k_max)
  kD0 <- k_info$k_best
  D0 <- upgma_cluster(X_scaled, kD0)

  list(
    D0 = D0,
    kD0 = kD0,
    asw_curve = data.frame(K = k_info$ks, ASW = k_info$asw)
  )
}

sanitize_train_val <- function(X_train, X_val, var_tol = 1e-10) {
  X_train <- as.matrix(X_train)
  X_val   <- as.matrix(X_val)

  X_train[!is.finite(X_train)] <- NA_real_
  X_val[!is.finite(X_val)]     <- NA_real_

  train_means <- colMeans(X_train, na.rm = TRUE)
  train_means[!is.finite(train_means)] <- 0

  for (j in seq_len(ncol(X_train))) {
    if (anyNA(X_train[, j])) X_train[is.na(X_train[, j]), j] <- train_means[j]
    if (anyNA(X_val[, j]))   X_val[is.na(X_val[, j]), j]     <- train_means[j]
  }

  sds <- apply(X_train, 2, sd, na.rm = TRUE)
  keep <- is.finite(sds) & (sds > var_tol)

  if (sum(keep) < 2) return(NULL)

  X_train <- X_train[, keep, drop = FALSE]
  X_val   <- X_val[, keep, drop = FALSE]

  list(X_train = X_train, X_val = X_val, keep = keep)
}

# =========================
# 2.1) IPCA DIAGNOSTICS
# =========================
perm_test_ari <- function(cl_pred, y_true, n_perm = 20) {
  set.seed(123)
  out <- numeric(n_perm)
  for (b in 1:n_perm) {
    out[b] <- mclust::adjustedRandIndex(cl_pred, sample(y_true))
  }
  out
}

basic_matrix_checks <- function(Z) {
  n_nf <- sum(!is.finite(Z))
  v <- apply(Z, 2, sd)
  n_zero <- sum(v < 1e-12, na.rm = TRUE)
  list(n_not_finite = n_nf, n_zero_sd = n_zero)
}

# =========================
# 3) FEATURES (ROBUST)
# =========================
feat_pca <- function(X_train, X_val, ncomp) {
  cleaned <- sanitize_train_val(X_train, X_val, var_tol = 1e-10)
  if (is.null(cleaned)) return(NULL)

  X_train <- cleaned$X_train
  X_val   <- cleaned$X_val

  max_ok <- min(nrow(X_train) - 1, ncol(X_train))
  if (is.na(max_ok) || max_ok < 2) return(NULL)

  ncomp_eff <- min(ncomp, max_ok)

  fit <- tryCatch(
    mixOmics::pca(X_train, ncomp = ncomp_eff, scale = FALSE),
    error = function(e) {
      message("PCA failed: ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(fit)) return(NULL)

  L <- fit$loadings$X[, seq_len(ncomp_eff), drop = FALSE]
  Z_val <- tryCatch(X_val %*% L, error = function(e) NULL)
  if (is.null(Z_val)) return(NULL)

  Z_train <- fit$X[, seq_len(ncomp_eff), drop = FALSE]
  if (any(!is.finite(Z_train)) || any(!is.finite(Z_val))) return(NULL)

  list(Z_train = Z_train, Z_val = Z_val, ncomp_eff = ncomp_eff)
}

feat_spca_sparse <- function(X_train, X_val, ncomp) {
  cleaned <- sanitize_train_val(X_train, X_val, var_tol = 1e-10)
  if (is.null(cleaned)) return(NULL)

  X_train <- cleaned$X_train
  X_val   <- cleaned$X_val

  max_ok <- min(nrow(X_train) - 1, ncol(X_train))
  if (is.na(max_ok) || max_ok < 2) return(NULL)

  ncomp_eff <- min(ncomp, max_ok)

  fit <- tryCatch(
    mixOmics::spca(X_train, ncomp = ncomp_eff, scale = FALSE),
    error = function(e) {
      message("SPCA failed: ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(fit)) return(NULL)
  if (is.null(fit$loadings$X) || is.null(fit$X)) return(NULL)

  r_eff <- min(ncomp_eff, ncol(fit$loadings$X), ncol(fit$X))
  if (is.na(r_eff) || r_eff < 2) return(NULL)

  L <- fit$loadings$X[, seq_len(r_eff), drop = FALSE]
  Z_val <- tryCatch(
    X_val %*% L,
    error = function(e) {
      message("SPCA projection failed: ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(Z_val)) return(NULL)

  Z_train <- fit$X[, seq_len(r_eff), drop = FALSE]
  if (any(!is.finite(Z_train)) || any(!is.finite(Z_val))) return(NULL)

  list(Z_train = Z_train, Z_val = Z_val, ncomp_eff = r_eff)
}

feat_ipca <- function(X_train, X_val, ncomp) {
  cleaned <- sanitize_train_val(X_train, X_val, var_tol = 1e-10)
  if (is.null(cleaned)) return(NULL)

  X_train <- cleaned$X_train
  X_val   <- cleaned$X_val

  max_ok <- min(nrow(X_train) - 1, ncol(X_train))
  if (is.na(max_ok) || max_ok < 2) return(NULL)

  ncomp_eff <- min(ncomp, max_ok)

  fit <- tryCatch(
    mixOmics::ipca(X_train, ncomp = ncomp_eff, scale = FALSE),
    error = function(e) {
      message("IPCA failed: ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(fit)) return(NULL)
  if (is.null(fit$loadings$X) || is.null(fit$X)) return(NULL)

  r_eff <- min(ncomp_eff, ncol(fit$loadings$X), ncol(fit$X))
  if (is.na(r_eff) || r_eff < 2) return(NULL)

  L <- fit$loadings$X[, seq_len(r_eff), drop = FALSE]
  Z_val <- tryCatch(
    X_val %*% L,
    error = function(e) {
      message("IPCA projection failed: ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(Z_val)) return(NULL)

  Z_train <- fit$X[, seq_len(r_eff), drop = FALSE]
  if (any(!is.finite(Z_train)) || any(!is.finite(Z_val))) return(NULL)

  list(Z_train = Z_train, Z_val = Z_val, ncomp_eff = r_eff)
}

# =========================
# 4) STRONGER AE
# =========================
build_dae <- function(input_dim,
                      hidden = c(512, 256),
                      bottleneck_dim = 64,
                      lr = 5e-4,
                      l2 = 1e-7,
                      dropout_in = 0.00,
                      dropout_hidden = 0.05) {

  inp <- layer_input(shape = input_dim, name = "input")

  x <- inp
  if (dropout_in > 0) {
    x <- x %>% layer_dropout(rate = dropout_in, name = "noise_dropout")
  }

  for (i in seq_along(hidden)) {
    x <- x %>%
      layer_dense(
        units = hidden[i],
        activation = "relu",
        kernel_regularizer = regularizer_l2(l2),
        name = paste0("enc_dense_", i)
      )

    if (dropout_hidden > 0) {
      x <- x %>% layer_dropout(rate = dropout_hidden, name = paste0("enc_do_", i))
    }
  }

  z <- x %>%
    layer_dense(
      units = bottleneck_dim,
      activation = "linear",
      kernel_regularizer = regularizer_l2(l2),
      name = "bottleneck"
    )

  x2 <- z
  for (i in rev(seq_along(hidden))) {
    x2 <- x2 %>%
      layer_dense(
        units = hidden[i],
        activation = "relu",
        kernel_regularizer = regularizer_l2(l2),
        name = paste0("dec_dense_", i)
      )
  }

  out <- x2 %>%
    layer_dense(units = input_dim, activation = "linear", name = "recon")

  autoencoder <- keras_model(inputs = inp, outputs = out)
  encoder     <- keras_model(inputs = inp, outputs = z)

  autoencoder %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = loss_huber()
  )

  list(autoencoder = autoencoder, encoder = encoder)
}

dae_fit_transform_safe <- function(X_train, X_val,
                                   hidden = c(512, 256),
                                   bottleneck_dim = 64,
                                   lr = 5e-4,
                                   l2 = 1e-7,
                                   dropout_in = 0.00,
                                   dropout_hidden = 0.05,
                                   epochs = 200,
                                   batch_size = 32,
                                   patience = 25,
                                   verbose = 0,
                                   seed = 123) {

  out <- tryCatch({

    tensorflow::tf$random$set_seed(seed)
    keras::k_clear_session()
    gc()

    m <- build_dae(
      input_dim = ncol(X_train),
      hidden = hidden,
      bottleneck_dim = bottleneck_dim,
      lr = lr,
      l2 = l2,
      dropout_in = dropout_in,
      dropout_hidden = dropout_hidden
    )

    cb_early <- callback_early_stopping(
      monitor = "val_loss",
      patience = patience,
      restore_best_weights = TRUE
    )

    cb_lr <- callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      factor = 0.5,
      patience = 8,
      min_lr = 1e-5
    )

    hist <- m$autoencoder %>% fit(
      x = X_train,
      y = X_train,
      validation_split = 0.2,
      epochs = epochs,
      batch_size = batch_size,
      callbacks = list(cb_early, cb_lr),
      verbose = verbose
    )

    Ztr <- as.matrix(predict(m$encoder, X_train))
    Zva <- as.matrix(predict(m$encoder, X_val))

    list(
      Z_train = Ztr,
      Z_val = Zva,
      final_val_loss = min(unlist(hist$metrics$val_loss))
    )

  }, error = function(e) {
    message("AE fit failed: ", conditionMessage(e))
    NULL
  })

  out
}

feat_hybrid_dae <- function(extractor_fun, X_train, X_val, ncomp,
                            bottleneck_dim = 64,
                            hidden = c(256, 128),
                            lr = 5e-4,
                            l2 = 1e-7,
                            dropout_in = 0.00,
                            dropout_hidden = 0.05,
                            epochs = 200,
                            batch_size = 32,
                            seed = 123) {
  Z <- extractor_fun(X_train, X_val, ncomp)
  if (is.null(Z)) return(NULL)

  dae_fit_transform_safe(
    Z$Z_train, Z$Z_val,
    hidden = hidden,
    bottleneck_dim = bottleneck_dim,
    lr = lr,
    l2 = l2,
    dropout_in = dropout_in,
    dropout_hidden = dropout_hidden,
    epochs = epochs,
    batch_size = batch_size,
    patience = 25,
    verbose = 0,
    seed = seed
  )
}

# =========================
# 5) STAGE 1: PCA screening
# =========================
select_top_ncomp_stage1 <- function(X_train_outer, y_train_outer,
                                   inner_k = 3, k_min = 2, k_max = 10,
                                   ncomp_grid = seq(2, 20, 2), topN = 6) {
  inner_folds <- caret::createFolds(y_train_outer, k = inner_k, list = TRUE, returnTrain = FALSE)
  res <- data.frame()

  for (ncomp in ncomp_grid) {
    ari_vals <- c()

    for (fi in seq_len(inner_k)) {
      idx_val <- inner_folds[[fi]]
      idx_tr  <- setdiff(seq_len(nrow(X_train_outer)), idx_val)

      Z <- feat_pca(
        X_train_outer[idx_tr, , drop = FALSE],
        X_train_outer[idx_val, , drop = FALSE],
        ncomp = ncomp
      )
      if (is.null(Z)) next

      split_res <- eval_split(
        Z$Z_train, Z$Z_val, y_train_outer[idx_val],
        D0_val = NULL, k_min = k_min, k_max = k_max
      )
      ari_vals <- c(ari_vals, split_res$ARI_true)
    }

    if (length(ari_vals) == 0) next
    res <- rbind(res, data.frame(ncomp = ncomp, meanARI = mean(ari_vals, na.rm = TRUE)))
  }

  if (nrow(res) == 0) stop("Stage1 failed: no viable ncomp in inner folds.")
  res <- res[order(-res$meanARI), ]
  unique(head(res$ncomp, topN))
}

# =========================
# 6) MAIN RUN: nested CV
# =========================
run_two_stage_nested_one_method <- function(dados_in, analysis_name, method_name,
                                            remove_admix = TRUE,
                                            k_min = 2, k_max = 10,
                                            outer_k = 5, inner_k = 3,
                                            max_ncomp_cap = 20,
                                            topN_hybrid = 6,
                                            grid_bn = c(32, 64, 128),
                                            grid_hidden = list(c(512, 256), c(256, 128)),
                                            grid_dropout_in = c(0.00, 0.05),
                                            grid_dropout_hidden = c(0.00, 0.05),
                                            grid_l2 = c(1e-7, 1e-6),
                                            epochs = 200,
                                            batch_size = 32,
                                            lr = 5e-4,
                                            ipca_perm_B = 20) {

  dd <- if (remove_admix) dados_in[dados_in[[label_col]] != admix_label, , drop = FALSE] else dados_in
  ids <- as.character(dd[[id_col]])
  y   <- as.factor(dd[[label_col]])

  X_raw <- as.matrix(dd[, snp_cols])
  mode(X_raw) <- "numeric"
  X <- scale(X_raw)
  rownames(X) <- ids

  out_dir <- file.path(getwd(), paste0("results_two_stage_nested_", analysis_name, "_", method_name))
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(out_dir, "inner_grids"), showWarnings = FALSE, recursive = TRUE)

  d0 <- compute_D0(X, k_min = k_min, k_max = k_max)
  write.csv(d0$asw_curve, file.path(out_dir, "D0_ASW_curve.csv"), row.names = FALSE)

  outer_folds <- caret::createFolds(y, k = outer_k, list = TRUE, returnTrain = FALSE)

  ALL_INNER <- list()
  BEST_OUTER <- list()
  mth <- method_name

  for (fo in seq_len(outer_k)) {

    cat("\n--- OUTER FOLD ", fo, "/", outer_k, " ---\n", sep = "")

    idx_test  <- outer_folds[[fo]]
    idx_train <- setdiff(seq_len(nrow(X)), idx_test)

    X_train_outer <- X[idx_train, , drop = FALSE]
    X_test_outer  <- X[idx_test,  , drop = FALSE]
    y_train_outer <- y[idx_train]
    y_test_outer  <- y[idx_test]

    min_inner_train <- floor((inner_k - 1) / inner_k * nrow(X_train_outer))
    max_possible <- min(min_inner_train - 1, ncol(X_train_outer), max_ncomp_cap)
    if (is.na(max_possible) || max_possible < 2) stop("Not enough samples for PCA grid in outer fold.")
    ncomp_grid_big <- seq(2, max_possible, by = 2)

    ncomp_top <- select_top_ncomp_stage1(
      X_train_outer, y_train_outer,
      inner_k = inner_k, k_min = k_min, k_max = k_max,
      ncomp_grid = ncomp_grid_big, topN = topN_hybrid
    )

    inner_folds <- caret::createFolds(y_train_outer, k = inner_k, list = TRUE, returnTrain = FALSE)

    if (mth %in% c("PCA", "SPCA", "IPCA")) {
      candidates <- expand.grid(ncomp = ncomp_grid_big, bn = NA, hd = NA, doin = NA, dohd = NA, l2 = NA)
    } else if (mth == "AE") {
      candidates <- expand.grid(
        ncomp = NA, bn = grid_bn, hd = seq_along(grid_hidden),
        doin = grid_dropout_in, dohd = grid_dropout_hidden, l2 = grid_l2
      )
    } else if (mth %in% c("PCA-AE", "SPCA-AE", "IPCA-AE")) {
      candidates <- expand.grid(
        ncomp = ncomp_top, bn = grid_bn, hd = seq_along(grid_hidden),
        doin = grid_dropout_in, dohd = grid_dropout_hidden, l2 = grid_l2
      )
    } else if (mth == "D0") {
      candidates <- data.frame(ncomp = NA, bn = NA, hd = NA, doin = NA, dohd = NA, l2 = NA)
    } else {
      stop("Unknown method: ", mth)
    }

    inner_grid_rows <- data.frame()

    for (ci in seq_len(nrow(candidates))) {

      ncomp <- candidates$ncomp[ci]
      bn    <- candidates$bn[ci]
      hd_i  <- candidates$hd[ci]
      doin  <- candidates$doin[ci]
      dohd  <- candidates$dohd[ci]
      l2v   <- candidates$l2[ci]

      ari_vals <- c()
      nmi_vals <- c()
      pur_vals <- c()
      ari_d0_vals <- c()
      k_vals <- c()
      time_vals <- c()

      for (fi in seq_len(inner_k)) {

        idx_val <- inner_folds[[fi]]
        idx_tr  <- setdiff(seq_len(nrow(X_train_outer)), idx_val)

        X_tr <- X_train_outer[idx_tr, , drop = FALSE]
        X_va <- X_train_outer[idx_val, , drop = FALSE]
        y_va <- y_train_outer[idx_val]
        D0_va <- d0$D0[idx_train][idx_val]

        t0 <- proc.time()[3]
        split_res <- NULL

        if (mth == "D0") {

          split_res <- eval_split(X_tr, X_va, y_va, D0_val = D0_va, k_min = k_min, k_max = k_max)

        } else if (mth == "PCA") {

          Z <- feat_pca(X_tr, X_va, ncomp = ncomp)
          if (is.null(Z)) next
          split_res <- eval_split(Z$Z_train, Z$Z_val, y_va, D0_val = D0_va, k_min = k_min, k_max = k_max)

        } else if (mth == "SPCA") {

          Z <- feat_spca_sparse(X_tr, X_va, ncomp = ncomp)
          if (is.null(Z)) next
          split_res <- eval_split(Z$Z_train, Z$Z_val, y_va, D0_val = D0_va, k_min = k_min, k_max = k_max)

        } else if (mth == "IPCA") {

          Z <- feat_ipca(X_tr, X_va, ncomp = ncomp)
          if (is.null(Z)) next
          split_res <- eval_split(Z$Z_train, Z$Z_val, y_va, D0_val = D0_va, k_min = k_min, k_max = k_max)

        } else if (mth == "AE") {

          hidden_vec <- grid_hidden[[hd_i]]
          AE <- dae_fit_transform_safe(
            X_tr, X_va,
            hidden = hidden_vec,
            bottleneck_dim = bn,
            lr = lr, l2 = l2v,
            dropout_in = doin, dropout_hidden = dohd,
            epochs = epochs, batch_size = batch_size,
            patience = 25, verbose = 0, seed = 123
          )
          if (is.null(AE)) next
          split_res <- eval_split(AE$Z_train, AE$Z_val, y_va, D0_val = D0_va, k_min = k_min, k_max = k_max)

        } else if (mth == "PCA-AE") {

          hidden_vec <- grid_hidden[[hd_i]]
          HY <- feat_hybrid_dae(
            feat_pca, X_tr, X_va, ncomp = ncomp,
            bottleneck_dim = bn,
            hidden = hidden_vec,
            lr = lr, l2 = l2v,
            dropout_in = doin, dropout_hidden = dohd,
            epochs = epochs, batch_size = batch_size, seed = 123
          )
          if (is.null(HY)) next
          split_res <- eval_split(HY$Z_train, HY$Z_val, y_va, D0_val = D0_va, k_min = k_min, k_max = k_max)

        } else if (mth == "SPCA-AE") {

          hidden_vec <- grid_hidden[[hd_i]]
          HY <- feat_hybrid_dae(
            feat_spca_sparse, X_tr, X_va, ncomp = ncomp,
            bottleneck_dim = bn,
            hidden = hidden_vec,
            lr = lr, l2 = l2v,
            dropout_in = doin, dropout_hidden = dohd,
            epochs = epochs, batch_size = batch_size, seed = 123
          )
          if (is.null(HY)) next
          split_res <- eval_split(HY$Z_train, HY$Z_val, y_va, D0_val = D0_va, k_min = k_min, k_max = k_max)

        } else if (mth == "IPCA-AE") {

          hidden_vec <- grid_hidden[[hd_i]]
          HY <- feat_hybrid_dae(
            feat_ipca, X_tr, X_va, ncomp = ncomp,
            bottleneck_dim = bn,
            hidden = hidden_vec,
            lr = lr, l2 = l2v,
            dropout_in = doin, dropout_hidden = dohd,
            epochs = epochs, batch_size = batch_size, seed = 123
          )
          if (is.null(HY)) next
          split_res <- eval_split(HY$Z_train, HY$Z_val, y_va, D0_val = D0_va, k_min = k_min, k_max = k_max)
        }

        t1 <- proc.time()[3]
        time_vals <- c(time_vals, (t1 - t0))

        ari_vals <- c(ari_vals, split_res$ARI_true)
        nmi_vals <- c(nmi_vals, split_res$NMI_true)
        pur_vals <- c(pur_vals, split_res$Purity_true)
        ari_d0_vals <- c(ari_d0_vals, split_res$ARI_D0)
        k_vals <- c(k_vals, split_res$k)

        gc()
      }

      if (length(ari_vals) == 0) next

      inner_grid_rows <- bind_rows(inner_grid_rows, data.frame(
        analysis = analysis_name,
        method = mth,
        outer_fold = fo,
        ncomp = ifelse(is.na(ncomp), NA, ncomp),
        bottleneck = ifelse(is.na(bn), NA, bn),
        hidden = ifelse(is.na(hd_i), NA, paste(grid_hidden[[hd_i]], collapse = ";")),
        dropout_in = ifelse(is.na(doin), NA, doin),
        dropout_hidden = ifelse(is.na(dohd), NA, dohd),
        l2 = ifelse(is.na(l2v), NA, l2v),
        inner_ARI_true_mean = mean(ari_vals, na.rm = TRUE),
        inner_ARI_true_sd = sd(ari_vals, na.rm = TRUE),
        inner_NMI_true_mean = mean(nmi_vals, na.rm = TRUE),
        inner_Purity_true_mean = mean(pur_vals, na.rm = TRUE),
        inner_ARI_D0_mean = mean(ari_d0_vals, na.rm = TRUE),
        inner_k_mean = mean(k_vals, na.rm = TRUE),
        inner_time_mean_sec = mean(time_vals, na.rm = TRUE),
        stage1_topN_ncomp = paste(ncomp_top, collapse = ";"),
        stage1_biggrid_max_ncomp = max_possible
      ))
    }

    write.csv(
      inner_grid_rows,
      file.path(out_dir, "inner_grids", paste0("INNER_GRID_", mth, "_outer", fo, ".csv")),
      row.names = FALSE
    )

    ALL_INNER[[paste(analysis_name, mth, fo, sep = "_")]] <- inner_grid_rows

    if (nrow(inner_grid_rows) == 0) stop("No inner results for method ", mth, " outer fold ", fo)

    best_inner <- inner_grid_rows[which.max(inner_grid_rows$inner_ARI_true_mean), , drop = FALSE]

    # -------- OUTER EVAL --------
    t0o <- proc.time()[3]
    cl_test <- NULL
    k_outer <- NA_integer_

    if (mth == "D0") {

      k_outer <- choose_k_asw(X_train_outer, k_min = k_min, k_max = k_max)$k_best
      cl_test <- upgma_cluster(X_test_outer, k_outer)

    } else if (mth == "PCA") {

      Z <- feat_pca(X_train_outer, X_test_outer, ncomp = best_inner$ncomp)
      scT <- scale_by_train(Z$Z_train, Z$Z_train)
      k_outer <- choose_k_asw(scT$Z_train, k_min = k_min, k_max = k_max)$k_best
      sc <- scale_by_train(Z$Z_train, Z$Z_val)
      cl_test <- upgma_cluster(sc$Z_val, k_outer)

    } else if (mth == "SPCA") {

      Z <- feat_spca_sparse(X_train_outer, X_test_outer, ncomp = best_inner$ncomp)
      scT <- scale_by_train(Z$Z_train, Z$Z_train)
      k_outer <- choose_k_asw(scT$Z_train, k_min = k_min, k_max = k_max)$k_best
      sc <- scale_by_train(Z$Z_train, Z$Z_val)
      cl_test <- upgma_cluster(sc$Z_val, k_outer)

    } else if (mth == "IPCA") {

      Z <- feat_ipca(X_train_outer, X_test_outer, ncomp = best_inner$ncomp)
      scT <- scale_by_train(Z$Z_train, Z$Z_train)
      k_outer <- choose_k_asw(scT$Z_train, k_min = k_min, k_max = k_max)$k_best
      sc <- scale_by_train(Z$Z_train, Z$Z_val)
      cl_test <- upgma_cluster(sc$Z_val, k_outer)

      chk <- basic_matrix_checks(sc$Z_val)
      write.csv(
        data.frame(
          outer_fold = fo,
          ncomp = best_inner$ncomp,
          k_outer = k_outer,
          Z_not_finite = chk$n_not_finite,
          Z_zero_sd_cols = chk$n_zero_sd
        ),
        file.path(out_dir, paste0("IPCA_DIAGNOSTICS_outer", fo, ".csv")),
        row.names = FALSE
      )

      ari_perm <- perm_test_ari(cl_test, y_test_outer, n_perm = ipca_perm_B)
      write.csv(
        data.frame(outer_fold = fo, ARI_perm = ari_perm),
        file.path(out_dir, paste0("PERMUTATION_CHECK_outer", fo, ".csv")),
        row.names = FALSE
      )

    } else if (mth == "AE") {

      hidden_vec <- as.integer(strsplit(best_inner$hidden, ";")[[1]])
      AE <- dae_fit_transform_safe(
        X_train_outer, X_test_outer,
        hidden = hidden_vec,
        bottleneck_dim = best_inner$bottleneck,
        lr = lr, l2 = best_inner$l2,
        dropout_in = best_inner$dropout_in,
        dropout_hidden = best_inner$dropout_hidden,
        epochs = epochs, batch_size = batch_size,
        patience = 25, verbose = 0, seed = 123
      )
      if (is.null(AE)) stop("AE failed on outer fold ", fo)

      scT <- scale_by_train(AE$Z_train, AE$Z_train)
      k_outer <- choose_k_asw(scT$Z_train, k_min = k_min, k_max = k_max)$k_best
      sc <- scale_by_train(AE$Z_train, AE$Z_val)
      cl_test <- upgma_cluster(sc$Z_val, k_outer)

    } else if (mth == "PCA-AE") {

      hidden_vec <- as.integer(strsplit(best_inner$hidden, ";")[[1]])
      HY <- feat_hybrid_dae(
        feat_pca, X_train_outer, X_test_outer, ncomp = best_inner$ncomp,
        bottleneck_dim = best_inner$bottleneck, hidden = hidden_vec,
        lr = lr, l2 = best_inner$l2,
        dropout_in = best_inner$dropout_in,
        dropout_hidden = best_inner$dropout_hidden,
        epochs = epochs, batch_size = batch_size, seed = 123
      )
      if (is.null(HY)) stop("PCA-AE failed on outer fold ", fo)

      scT <- scale_by_train(HY$Z_train, HY$Z_train)
      k_outer <- choose_k_asw(scT$Z_train, k_min = k_min, k_max = k_max)$k_best
      sc <- scale_by_train(HY$Z_train, HY$Z_val)
      cl_test <- upgma_cluster(sc$Z_val, k_outer)

    } else if (mth == "SPCA-AE") {

      hidden_vec <- as.integer(strsplit(best_inner$hidden, ";")[[1]])
      HY <- feat_hybrid_dae(
        feat_spca_sparse, X_train_outer, X_test_outer, ncomp = best_inner$ncomp,
        bottleneck_dim = best_inner$bottleneck, hidden = hidden_vec,
        lr = lr, l2 = best_inner$l2,
        dropout_in = best_inner$dropout_in,
        dropout_hidden = best_inner$dropout_hidden,
        epochs = epochs, batch_size = batch_size, seed = 123
      )
      if (is.null(HY)) stop("SPCA-AE failed on outer fold ", fo)

      scT <- scale_by_train(HY$Z_train, HY$Z_train)
      k_outer <- choose_k_asw(scT$Z_train, k_min = k_min, k_max = k_max)$k_best
      sc <- scale_by_train(HY$Z_train, HY$Z_val)
      cl_test <- upgma_cluster(sc$Z_val, k_outer)

    } else if (mth == "IPCA-AE") {

      hidden_vec <- as.integer(strsplit(best_inner$hidden, ";")[[1]])
      HY <- feat_hybrid_dae(
        feat_ipca, X_train_outer, X_test_outer, ncomp = best_inner$ncomp,
        bottleneck_dim = best_inner$bottleneck, hidden = hidden_vec,
        lr = lr, l2 = best_inner$l2,
        dropout_in = best_inner$dropout_in,
        dropout_hidden = best_inner$dropout_hidden,
        epochs = epochs, batch_size = batch_size, seed = 123
      )
      if (is.null(HY)) stop("IPCA-AE failed on outer fold ", fo)

      scT <- scale_by_train(HY$Z_train, HY$Z_train)
      k_outer <- choose_k_asw(scT$Z_train, k_min = k_min, k_max = k_max)$k_best
      sc <- scale_by_train(HY$Z_train, HY$Z_val)
      cl_test <- upgma_cluster(sc$Z_val, k_outer)
    }

    t1o <- proc.time()[3]
    time_outer <- t1o - t0o

    met <- cluster_metrics(cl_test, y_test_outer)
    ari_d0 <- mclust::adjustedRandIndex(cl_test, d0$D0[idx_test])

    BEST_OUTER[[paste(analysis_name, mth, fo, sep = "_")]] <- data.frame(
      analysis = analysis_name,
      method = mth,
      outer_fold = fo,
      ncomp = best_inner$ncomp,
      bottleneck = best_inner$bottleneck,
      hidden = best_inner$hidden,
      k_outer = k_outer,
      outer_ARI_true = met$ARI,
      outer_NMI_true = met$NMI,
      outer_Purity_true = met$Purity,
      outer_ARI_D0 = ari_d0,
      time_outer_sec = time_outer,
      inner_selected_ARI_true = best_inner$inner_ARI_true_mean,
      stage1_topN_ncomp = best_inner$stage1_topN_ncomp,
      stage1_biggrid_max_ncomp = max_possible
    )

    cm <- confusion_tables(cl_test, y_test_outer, D0 = d0$D0[idx_test])
    write.csv(as.data.frame.matrix(cm$confusion_true),
              file.path(out_dir, paste0("CONFUSION_TRUE_outer", fo, ".csv")))
    write.csv(as.data.frame.matrix(cm$confusion_D0),
              file.path(out_dir, paste0("CONFUSION_D0_outer", fo, ".csv")))

    gc()
  }

  inner_all <- bind_rows(ALL_INNER)
  best_all  <- bind_rows(BEST_OUTER)

  write.csv(inner_all, file.path(out_dir, "ALL_INNER_RESULTS.csv"), row.names = FALSE)
  write.csv(best_all,  file.path(out_dir, "BEST_OUTER_FOLDS.csv"), row.names = FALSE)

  summary_outer <- best_all %>%
    group_by(method) %>%
    summarise(
      outer_ARI_true_mean = mean(outer_ARI_true, na.rm = TRUE),
      outer_ARI_true_sd = sd(outer_ARI_true, na.rm = TRUE),
      outer_NMI_true_mean = mean(outer_NMI_true, na.rm = TRUE),
      outer_Purity_true_mean = mean(outer_Purity_true, na.rm = TRUE),
      outer_ARI_D0_mean = mean(outer_ARI_D0, na.rm = TRUE),
      time_outer_mean_sec = mean(time_outer_sec, na.rm = TRUE),
      time_outer_sd_sec = sd(time_outer_sec, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(outer_ARI_true_mean))

  write.csv(summary_outer, file.path(out_dir, "SUMMARY_OUTER.csv"), row.names = FALSE)

  cat("\nSaved outputs to: ", out_dir, "\n", sep = "")
  cat("Dataset D0 K = ", d0$kD0, "\n", sep = "")
  print(summary_outer)

  invisible(list(out_dir = out_dir, d0 = d0, summary_outer = summary_outer))
}

# =========================
# 6.1) CONFUSION ONCE (no CV)
# =========================
make_confusion_once_from_best <- function(dados_in, analysis_name, method_name, remove_admix = TRUE,
                                         k_min = 2, k_max = 10,
                                         best_csv_path) {

  dd <- if (remove_admix) dados_in[dados_in[[label_col]] != admix_label, , drop = FALSE] else dados_in
  ids <- as.character(dd[[id_col]])
  y   <- as.factor(dd[[label_col]])

  X_raw <- as.matrix(dd[, snp_cols])
  mode(X_raw) <- "numeric"
  X <- scale(X_raw)
  rownames(X) <- ids

  out_dir <- file.path(getwd(), paste0("results_two_stage_nested_", analysis_name, "_", method_name))
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  d0 <- compute_D0(X, k_min = k_min, k_max = k_max)

  best <- read.csv(best_csv_path, stringsAsFactors = FALSE)
  best <- best[order(-best$outer_ARI_true), , drop = FALSE]
  b <- best[1, , drop = FALSE]

  mth <- method_name
  cl_full <- NULL
  k_full <- NULL

  if (mth == "D0") {

    k_full <- choose_k_asw(X, k_min = k_min, k_max = k_max)$k_best
    cl_full <- upgma_cluster(X, k_full)

  } else if (mth == "PCA") {

    Z <- feat_pca(X, X, ncomp = b$ncomp)
    sc <- scale_by_train(Z$Z_train, Z$Z_train)
    k_full <- choose_k_asw(sc$Z_train, k_min = k_min, k_max = k_max)$k_best
    cl_full <- upgma_cluster(sc$Z_train, k_full)

  } else if (mth == "SPCA") {

    Z <- feat_spca_sparse(X, X, ncomp = b$ncomp)
    sc <- scale_by_train(Z$Z_train, Z$Z_train)
    k_full <- choose_k_asw(sc$Z_train, k_min = k_min, k_max = k_max)$k_best
    cl_full <- upgma_cluster(sc$Z_train, k_full)

  } else if (mth == "IPCA") {

    Z <- feat_ipca(X, X, ncomp = b$ncomp)
    sc <- scale_by_train(Z$Z_train, Z$Z_train)
    k_full <- choose_k_asw(sc$Z_train, k_min = k_min, k_max = k_max)$k_best
    cl_full <- upgma_cluster(sc$Z_train, k_full)

  } else if (mth == "AE") {

    hidden_vec <- as.integer(strsplit(b$hidden, ";")[[1]])
    AE <- dae_fit_transform_safe(
      X, X,
      hidden = hidden_vec,
      bottleneck_dim = b$bottleneck,
      lr = 5e-4, l2 = b$l2,
      dropout_in = b$dropout_in,
      dropout_hidden = b$dropout_hidden,
      epochs = 200, batch_size = 32,
      patience = 25, verbose = 0, seed = 123
    )
    if (is.null(AE)) stop("AE failed in confusion_once")

    sc <- scale_by_train(AE$Z_train, AE$Z_train)
    k_full <- choose_k_asw(sc$Z_train, k_min = k_min, k_max = k_max)$k_best
    cl_full <- upgma_cluster(sc$Z_train, k_full)

  } else if (mth == "PCA-AE") {

    hidden_vec <- as.integer(strsplit(b$hidden, ";")[[1]])
    HY <- feat_hybrid_dae(
      feat_pca, X, X, ncomp = b$ncomp,
      bottleneck_dim = b$bottleneck,
      hidden = hidden_vec,
      lr = 5e-4, l2 = b$l2,
      dropout_in = b$dropout_in,
      dropout_hidden = b$dropout_hidden,
      epochs = 200, batch_size = 32, seed = 123
    )
    if (is.null(HY)) stop("PCA-AE failed in confusion_once")

    sc <- scale_by_train(HY$Z_train, HY$Z_train)
    k_full <- choose_k_asw(sc$Z_train, k_min = k_min, k_max = k_max)$k_best
    cl_full <- upgma_cluster(sc$Z_train, k_full)

  } else if (mth == "SPCA-AE") {

    hidden_vec <- as.integer(strsplit(b$hidden, ";")[[1]])
    HY <- feat_hybrid_dae(
      feat_spca_sparse, X, X, ncomp = b$ncomp,
      bottleneck_dim = b$bottleneck,
      hidden = hidden_vec,
      lr = 5e-4, l2 = b$l2,
      dropout_in = b$dropout_in,
      dropout_hidden = b$dropout_hidden,
      epochs = 200, batch_size = 32, seed = 123
    )
    if (is.null(HY)) stop("SPCA-AE failed in confusion_once")

    sc <- scale_by_train(HY$Z_train, HY$Z_train)
    k_full <- choose_k_asw(sc$Z_train, k_min = k_min, k_max = k_max)$k_best
    cl_full <- upgma_cluster(sc$Z_train, k_full)

  } else if (mth == "IPCA-AE") {

    hidden_vec <- as.integer(strsplit(b$hidden, ";")[[1]])
    HY <- feat_hybrid_dae(
      feat_ipca, X, X, ncomp = b$ncomp,
      bottleneck_dim = b$bottleneck,
      hidden = hidden_vec,
      lr = 5e-4, l2 = b$l2,
      dropout_in = b$dropout_in,
      dropout_hidden = b$dropout_hidden,
      epochs = 200, batch_size = 32, seed = 123
    )
    if (is.null(HY)) stop("IPCA-AE failed in confusion_once")

    sc <- scale_by_train(HY$Z_train, HY$Z_train)
    k_full <- choose_k_asw(sc$Z_train, k_min = k_min, k_max = k_max)$k_best
    cl_full <- upgma_cluster(sc$Z_train, k_full)

  } else {
    stop("Unknown method in confusion_once: ", mth)
  }

  cm <- confusion_tables(cl_full, y, D0 = d0$D0)

  write.csv(as.data.frame.matrix(cm$confusion_true),
            file.path(out_dir, "CONFUSION_ONCE_TRUE_full.csv"))
  write.csv(as.data.frame.matrix(cm$confusion_D0),
            file.path(out_dir, "CONFUSION_ONCE_D0_full.csv"))

  df_assign <- data.frame(
    IID = ids,
    Sub.population = as.character(y),
    D0 = d0$D0,
    cluster = cl_full
  )
  write.csv(df_assign, file.path(out_dir, "ASSIGNMENTS_full.csv"), row.names = FALSE)

  cat("\nConfusion matrices saved in:\n  ", out_dir, "\n", sep = "")
}

# =========================
# 7) RUN
# =========================
remove_admix_flag <- (ANALYSIS_TO_RUN == "MAIN_no_ADMIX")
out_dir <- file.path(getwd(), paste0("results_two_stage_nested_", ANALYSIS_TO_RUN, "_", METHOD_TO_RUN))

if (CONFUSION_ONCE) {
  best_path <- file.path(out_dir, "BEST_OUTER_FOLDS.csv")
  if (!file.exists(best_path)) {
    stop("BEST_OUTER_FOLDS.csv not found. Run CV at least once before confusion_once.\nMissing: ", best_path)
  }

  make_confusion_once_from_best(
    dados_in = dados,
    analysis_name = ANALYSIS_TO_RUN,
    method_name = METHOD_TO_RUN,
    remove_admix = remove_admix_flag,
    k_min = 2, k_max = 10,
    best_csv_path = best_path
  )

} else {

  run_two_stage_nested_one_method(
    dados_in = dados,
    analysis_name = ANALYSIS_TO_RUN,
    method_name = METHOD_TO_RUN,
    remove_admix = remove_admix_flag,
    k_min = 2, k_max = 10,
    outer_k = 5, inner_k = 3,
    max_ncomp_cap = 20,
    topN_hybrid = 6,
    grid_bn = c(32, 64, 128),
    grid_hidden = list(c(512, 256), c(256, 128)),
    grid_dropout_in = c(0.00, 0.05),
    grid_dropout_hidden = c(0.00, 0.05),
    grid_l2 = c(1e-7, 1e-6),
    epochs = 200,
    batch_size = 32,
    lr = 5e-4,
    ipca_perm_B = 20
  )
}

cat("\nNOTE:\n")
cat("mixOmics::spca() is Sparse PCA (sPCA), NOT supervised PCA.\n")
cat("SPCA/IPCA robustness was improved with sanitize_train_val() + tryCatch().\n")
cat("AE was strengthened: no BatchNorm, lower regularization, larger bottleneck, bigger hidden layers, Huber loss.\n")
cat("The key clustering fix is scale_by_train() before distance/clustering.\n")
cat("CUDA warnings are OK on CPU nodes.\n\n")
#!/usr/bin/env Rscript
# Thin bridge to the GPL-3 sgsR package. Nemora does not copy or relicense
# sgsR code; this script calls the installed package and writes portable
# artifacts for the Python wrapper.

args <- commandArgs(trailingOnly = TRUE)

has_arg <- function(flag) {
  !is.na(match(flag, args))
}

get_arg <- function(flag) {
  index <- match(flag, args)
  if (is.na(index)) {
    stop(sprintf("missing required argument: %s", flag), call. = FALSE)
  }
  if (index == length(args)) {
    stop(sprintf("argument %s has no value", flag), call. = FALSE)
  }
  args[[index + 1L]]
}

metric_path <- get_arg("--metric")
out_dir <- get_arg("--out")
n_samples <- as.integer(get_arg("--n-samples"))
n_strata <- as.integer(get_arg("--n-strata"))
allocation <- get_arg("--allocation")
method <- get_arg("--method")
seed <- as.integer(get_arg("--seed"))
mindist <- if (has_arg("--mindist")) as.numeric(get_arg("--mindist")) else NULL

if (!file.exists(metric_path)) {
  stop(sprintf("metric raster does not exist: %s", metric_path), call. = FALSE)
}
if (is.na(n_samples) || n_samples < 1L) {
  stop("--n-samples must be a positive integer", call. = FALSE)
}
if (is.na(n_strata) || n_strata < 2L) {
  stop("--n-strata must be an integer >= 2", call. = FALSE)
}
if (!(allocation %in% c("prop", "equal", "optim"))) {
  stop("--allocation must be one of: prop, equal, optim", call. = FALSE)
}
if (!(method %in% c("Queinnec", "random"))) {
  stop("--method must be one of: Queinnec, random", call. = FALSE)
}
if (is.na(seed)) {
  stop("--seed must be an integer", call. = FALSE)
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(sgsR)
  library(terra)
  library(sf)
})

metric <- terra::rast(metric_path)
if (terra::nlyr(metric) != 1L) {
  stop("metric raster must have exactly one layer", call. = FALSE)
}
valid_cells <- terra::global(!is.na(metric), "sum", na.rm = TRUE)[1, 1]
if (is.na(valid_cells) || valid_cells == 0) {
  stop("structural metric has no valid cells", call. = FALSE)
}

set.seed(seed)
strata_raster <- sgsR::strat_quantiles(mraster = metric, nStrata = n_strata)
strata_values <- as.numeric(terra::values(strata_raster, mat = FALSE))
strata_values <- strata_values[!is.na(strata_values)]
if (length(unique(strata_values)) < 2L) {
  stop(
    paste(
      "structural metric produced fewer than two strata;",
      "provide a non-constant metric layer"
    ),
    call. = FALSE
  )
}

allocation_table <- sgsR::calculate_allocation(
  sraster = strata_raster,
  nSamp = n_samples,
  allocation = allocation,
  mraster = metric,
  force = TRUE
)

set.seed(seed)
samples <- sgsR::sample_strat(
  sraster = strata_raster,
  nSamp = n_samples,
  allocation = allocation,
  method = method,
  mraster = metric,
  mindist = mindist,
  force = TRUE,
  plot = FALSE
)
if (nrow(samples) == 0L) {
  stop("sgsR returned no samples", call. = FALSE)
}

terra::writeRaster(
  strata_raster,
  filename = file.path(out_dir, "strata.tif"),
  overwrite = TRUE
)

frequency <- as.data.frame(terra::freq(strata_raster))
frequency <- data.frame(strata = frequency$value, cells = frequency$count)

minimum <- as.data.frame(terra::zonal(metric, strata_raster, fun = "min", na.rm = TRUE))
names(minimum)[2] <- "min"
maximum <- as.data.frame(terra::zonal(metric, strata_raster, fun = "max", na.rm = TRUE))
names(maximum)[2] <- "max"
average <- as.data.frame(terra::zonal(metric, strata_raster, fun = "mean", na.rm = TRUE))
names(average)[2] <- "mean"
lookup <- Reduce(
  function(left, right) merge(left, right, by = "strata"),
  list(frequency, minimum, maximum, average)
)
lookup <- lookup[order(lookup$strata), , drop = FALSE]

write.csv(
  allocation_table,
  file = file.path(out_dir, "allocation.csv"),
  row.names = FALSE,
  quote = FALSE
)
write.csv(
  lookup,
  file = file.path(out_dir, "strata.csv"),
  row.names = FALSE,
  quote = FALSE
)

coordinates <- sf::st_coordinates(samples)
sample_table <- as.data.frame(samples)
sample_table$geometry <- NULL
sample_table$x <- coordinates[, 1]
sample_table$y <- coordinates[, 2]
sample_table <- sample_table[, c("x", "y", "strata", "type", "rule")]
write.csv(
  sample_table,
  file = file.path(out_dir, "samples.csv"),
  row.names = FALSE,
  quote = FALSE
)

metadata <- c(
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  sgsr_version = as.character(utils::packageVersion("sgsR")),
  sgsr_licence = "GPL (>= 3)",
  terra_version = as.character(utils::packageVersion("terra")),
  sf_version = as.character(utils::packageVersion("sf")),
  seed = seed,
  n_samples_requested = n_samples,
  n_samples_returned = nrow(samples),
  n_strata = length(unique(strata_values)),
  allocation = allocation,
  method = method
)
write.table(
  data.frame(key = names(metadata), value = unname(metadata)),
  file = file.path(out_dir, "metadata.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = FALSE
)

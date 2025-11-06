#' @keywords internal
"_PACKAGE"

nemora <- NULL

.onLoad <- function(libname, pkgname) {
  nemora <<- reticulate::import("nemora", delay_load = TRUE)
}

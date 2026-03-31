# R Pipeline: Probability Normalization Evaluation
rm(list = ls())

# Function to calculate Implied Probabilities and Naive Normalization
GetProbNorm <- function(Q) {
  P <- 1 / Q
  Pnorm <- P / rowSums(P)
  return(Pnorm)
}

# Fetch Data dynamically
cat("Fetching Dataset...\n")
url <- "https://www.football-data.co.uk/mmz4281/2425/I1.csv"
V <- read.csv(url)
V <- V[!is.na(V$B365H) & !is.na(V$FTR), ]

# Outcome Matrix
O <- cbind((V$FTR == "H"), (V$FTR == "D"), (V$FTR == "A"))

# B365 Processing
Q365 <- cbind(V$B365H, V$B365D, V$B365A)
P365norm <- GetProbNorm(Q365)

# Calculate Brier Score (Quadratic)
LQUAD365 <- rowSums((P365norm - O)^2)
AVGLQUAD365 <- mean(LQUAD365)

cat("\n--- Evaluation Results ---\n")
cat(sprintf("Bet365 Naive Mean Brier Score: %.4f\n", AVGLQUAD365))
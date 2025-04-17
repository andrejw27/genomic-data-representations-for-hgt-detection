# Load the MatrixCorrelation package
library(MatrixCorrelation)
library(reticulate)
library(parallel)

# Use Python's pickle module
pickle <- import("pickle")

# Open the Pickle file path to the transformed data
data <- "benbow"
#file_path <- "dataset/data_representation/transformed_benbow.pkl"
file_path <- sprintf("dataset/data_representation/transformed_%s.pkl", data)
# Load the pickle file
data <- py_load_object(file_path)

# Define the string to exclude
exclude_string <- c("y_", "group_")

# Extract keys that do NOT start with the exclude_string
keys <- names(data)

filtered_items <- data[!sapply(names(data), function(key) any(startsWith(key, exclude_string)))]

# Initialize an empty dataframe to store results
results <- data.frame(
  Encoding1 = character(),
  Encoding2 = character(),
  Adjusted_RV = numeric(),
  stringsAsFactors = FALSE
)

# Compute adjusted RV coefficients for all pairs of items
keys <- names(filtered_items)
for (i in 1:(length(keys) - 1)) {
  for (j in (i + 1):length(keys)) {
    key1 <- keys[i]
    key2 <- keys[j]
#    if (key1 == "GC" | key2=="GC") {
    matrix1 <- filtered_items[[key1]]
    matrix2 <- filtered_items[[key2]]
    
    # Compute adjusted RV coefficient
    rv_result <- RVadj(matrix1, matrix2)
    
    # Store results in the dataframe
    results <- rbind(results, data.frame(
      Encoding1 = key1,
      Encoding2 = key2,
      Adjusted_RV = rv_result,
      stringsAsFactors = FALSE
    ))

  }
}

# Save the dataframe to a TSV file
write.table(results, file = "results_benbow.tsv", sep = "\t", row.names = FALSE)
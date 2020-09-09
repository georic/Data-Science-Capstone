# ----------------------------------------------------------------------------------------------
# Load Required Libraries from CRAN and Prepare RStudio Environment
# ----------------------------------------------------------------------------------------------

library(ggplot2)
library(NLP)
library(tm)
library(RWeka)
library(data.table)
library(dplyr)
library(bigmemory)
library(SnowballC)

# ----------------------------------------------------------------------------------------------
# Load training data from Course Dataset
# ----------------------------------------------------------------------------------------------

zip_URL <- "https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip"
zip_Data_File <- "data/Coursera-SwiftKey.zip"

if (!file.exists('data')) {
  dir.create('data')
}

if (!file.exists("data/final/en_US")) {
  tempFile <- tempfile()
  download.file(zip_URL, tempFile)
  unzip(tempFile, exdir = "data")
  unlink(tempFile)
  rm(tempFile)
}


# Blogs
blog <- readLines("data/final/en_US/en_US.blogs.txt",skipNul = TRUE, warn = TRUE)
print(paste0("Number of lines in Blogs file:", length(blog)))

# News
news <- readLines("data/final/en_US/en_US.news.txt",skipNul = TRUE, warn = TRUE)
print(paste0("Number of lines in News file:", length(news)))


# Twitter
twitter <- readLines("data/final/en_US/en_US.twitter.txt",skipNul = TRUE, warn = TRUE)
print(paste0("Number of lines in Twitter file:", length(twitter)))

# Load Summary
print("Training Data Loaded")
print(paste0("Total Number of lines in Training Dataset:   ", length(blog) + length(news) + length(twitter)))

# Remove variables to optimize computing memory usage
rm(zip_URL, zip_Data_File)


# Set seed for reproducibility and Assign sample size
set.seed(20000)
sample_size = 600 #Limited due to physical computing memory

sample_blog <- blog[sample(1:length(blog),sample_size)]
sample_news <- news[sample(1:length(news),sample_size)]
sample_twitter <- twitter[sample(1:length(twitter),sample_size)]

sample_data<-rbind(sample_blog,sample_news,sample_twitter)
rm(blog,news,twitter) # for optimizing the environment
print(paste0("Number of lines in Sample Data file:", length(sample_data)))

# Load list of bad words file
bad_words_URL <- "https://www.phorum.org/phorum5/file.php/63/2330/badwords.txt.zip"
bad_words_file <- "data/badwords.txt"
if (!file.exists('data')) {
  dir.create('data')
}
if (!file.exists(bad_words_file)) {
  tempFile <- tempfile()
  download.file(bad_words_URL, tempFile)
  unzip(tempFile, exdir = "data")
  unlink(tempFile)
  rm(tempFile)
}
bad <- file(bad_words_file, open = "r")
bad_words <- readLines(bad, encoding = "UTF-8", skipNul = TRUE)
bad_words <- iconv(bad_words, "latin1", "ASCII", sub = "")
close(bad)

# Remove Bad Words (Profanity)
sample_data <- removeWords(sample_data, bad_words)
print(paste0("Number of lines in Sample Data:", length(sample_data)))

# Remove URL, email addresses, Twitter handles and hash tags
sample_data <- gsub("(f|ht)tp(s?)://(.*)[.][a-z]+", "", sample_data, ignore.case = FALSE, perl = TRUE)
sample_data <- gsub("\\S+[@]\\S+", "", sample_data, ignore.case = FALSE, perl = TRUE)
sample_data <- gsub("@[^\\s]+", "", sample_data, ignore.case = FALSE, perl = TRUE)
sample_data <- gsub("#[^\\s]+", "", sample_data, ignore.case = FALSE, perl = TRUE)

# Convert Text to Lowercase
sample_data <- tolower(sample_data)

# Remove Punctuation, Special Characters and Strip White Space
sample_data <- gsub("[^\\p{L}'\\s]+", "", sample_data, ignore.case = FALSE, perl = TRUE)
sample_data  <- gsub("[^0-9A-Za-z///' ]","'" , sample_data ,ignore.case = TRUE)
sample_data <- gsub("''", "" , sample_data ,ignore.case = TRUE)
sample_data <- gsub("^\\s+|\\s+$", "", sample_data)
sample_data <- stripWhitespace(sample_data)

# Write sample data set to disk and optimize computing memory usage
sample_data_file <- "data/en_US.sample.txt"
sdf <- file(sample_data_file, open = "w")
writeLines(sample_data, sdf)
close(sdf)

# Optimize computing memory usage
rm(bad_words_URL, bad_words_file, sdf, sample_data_file)

print(paste0("Number of lines in Sample Data file:", length(sample_data)))

# ----------------------------------------------------------------------------------------------
# Build Corpus & NGram Frequencies  
# ----------------------------------------------------------------------------------------------
corpus<-VCorpus(VectorSource(sample_data))
corpus <- tm_map(corpus, content_transformer(tolower)) # Convert characters to lowercase
corpus <- tm_map(corpus, removePunctuation) # Remove punctuation
corpus <- tm_map(corpus, removeNumbers) # Remove Numbers
convert_spaces <- content_transformer(function(x, pattern) gsub(pattern, " ", x)) # Function to transform content
corpus <- tm_map(corpus, convert_spaces, "#/|@|\\|") #Replace Special Characters with Spaces
corpus <- tm_map(corpus, stripWhitespace) # Remove Multiple White Spaces



UniGram_Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
BiGram_Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
TriGram_Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
QuadGram_Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 4, max = 4))

Uni_T <- NGramTokenizer(corpus, Weka_control(min = 1, max = 1))
Uni_GM <- TermDocumentMatrix(corpus, control = list(tokenize = UniGram_Tokenizer))
Bi_GM <- TermDocumentMatrix(corpus, control = list(tokenize = BiGram_Tokenizer))
Tri_GM <- TermDocumentMatrix(corpus, control = list(tokenize = TriGram_Tokenizer))
Quad_GM <- TermDocumentMatrix(corpus, control = list(tokenize = QuadGram_Tokenizer))

# UniGram
Frequency_Terms1 <- findFreqTerms(Uni_GM, lowfreq = 4)
Freq_T1 <- rowSums(as.matrix(Uni_GM[Frequency_Terms1,]))
Freq_T1 <- data.frame(Unigram=names(Freq_T1), frequency=Freq_T1)
Freq_T1 <- Freq_T1[order(-Freq_T1$frequency),]
unigramlist <- setDT(Freq_T1)
save(unigramlist,file="Unigram.Rda")

# BiGram
Frequency_Terms2 <- findFreqTerms(Bi_GM, lowfreq = 3)
Freq_T2 <- rowSums(as.matrix(Bi_GM[Frequency_Terms2,]))
Freq_T2 <- data.frame(Bigram=names(Freq_T2), frequency=Freq_T2)
Freq_T2 <- Freq_T2[order(-Freq_T2$frequency),]
bigramlist <- setDT(Freq_T2)
save(bigramlist,file="Bigram.Rda")

# TriGram
Frequency_Terms3 <- findFreqTerms(Tri_GM, lowfreq = 2)
Freq_T3 <- rowSums(as.matrix(Tri_GM[Frequency_Terms3,]))
Freq_T3 <- data.frame(Trigram=names(Freq_T3), frequency=Freq_T3)
trigramlist <- setDT(Freq_T3)
save(trigramlist,file="Trigram.Rda")

# QuadGram
Frequency_Terms4 <- findFreqTerms(Quad_GM, lowfreq = 1)
Freq_T4 <- rowSums(as.matrix(Quad_GM[Frequency_Terms4,]))
Freq_T4 <- data.frame(Quadgram=names(Freq_T4), frequency=Freq_T4)
quadgramlist <- setDT(Freq_T4)
save(quadgramlist,file="Quadgram.Rda")
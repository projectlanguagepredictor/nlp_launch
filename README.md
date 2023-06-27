# NLP_Launch: Natural Language Predictor

**Esayas Asefa and Alexia Lewis**

---

## Project Description:

* GitHub is a version control system platform that projects are tracked and saved on. It acts a resource for professionals and students. README files provide summaries of those projects and are typically the first thing people see when visiting a repository. Predicting the programming language based on README files can allow for more efficient research capabilities and greater utility by users using specific languages.
    
## Project Goal:

* Predict programming languages based on words used in README.md files on GitHub.
    
## Initial Thoughts 

* Many coding languages have overlap in use but each language may have a function that it's more often used for or better for than others which would means words in README's will vary.
   
## The Plan

### Wrangle

* Acquire data from GitHub
    * Use env.py credentials to acquire data from GitHub

* Prepare data
    * 220 rows × 1 columns *before* cleaning
    * 218 rows × __ columns *after* cleaning
    * 2 duplicates found and removed
        * 'awesome-swift'
        * 'awesome-privacy'
    * Dropped columns
        * insert
    * Renamed columns
        * column 0: 'repo_name'
    * Created new columns
        * insert
        * insert
        * insert
        * insert
    * No nulls
    * Cleaned text
        * token
        * lemma
        * stopwords
    
   
    * Split into Train, Validate, Test

### Explore
1. What are the most common words in READMEs?
2. Does the length of the README vary by programming language?
3. Do different programming languages use a different number of unique words?
4. Are there any words that uniquely identify a programming language?

### Model

* Model
    * ___ Models Selected ___
    * Choose Evaluation Metric
    * Baseline eval metric
    * Fit and score model with Train
    * Score model with Validate
    * Select best model
    * Score model with Test
    
## Data Dictionary  

| Feature | Definition|
|:--------|:-----------|
|| |
|| |
|| |
|| |
|| |
|| |
|| |
|| |
|| |
|| |
|| |
|| |    


## Steps to Reproduce
* Clone this repo
* Create env.py file containing your unqiue: **github_token** and **github_username** to access
* Run notebook

## Takeaways and Conclusions


## Recommendations/Next Steps


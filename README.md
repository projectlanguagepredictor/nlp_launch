# NLP_Launch: Natural Language Predictor

**Esayas Asefa and Alexia Lewis**

---

## Project Description:

* GitHub is a version control system platform that projects are tracked and saved on. It acts a resource for professionals and students. README files provide summaries of those projects and are typically the first thing people see when visiting a repository. Predicting the programming language based on README files can allow for more efficient research capabilities and greater utility by users using specific languages.
    
## Project Goal:

* Predict programming languages based on words used in README.md files on GitHub.
    
## Initial Thoughts 

* Many coding languages have overlap in use but each language may have a function that it's more often used for or better for than others which would mean words in README's will vary.
   
## The Plan

* Imports Used:
    * pandas
    * numpy
    * os
    * ...

### Wrangle

* Acquire data from GitHub
    * Use env.py credentials to acquire data from GitHub
    * Each column is a feature of the repository
    * Each row is a repository

* Prepare data
    * 740 rows × 3 columns *before* cleaning
    * 740 rows × 5 columns *after* cleaning
    * No duplicates
    * Created new columns
        * clean (for clean readme contents)
        * lemma (for lemmatized clean readme contents)
    * No nulls
    * Changed languages that are not 'Python', 'JavaScript', 'HTML', 'Shell', 'Java', 'Go' to 'other'
        * other = 599

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
|repo| The 'author name/repo name' of the repo|
|language| The repo language used|
|readme_contents| The contents of the entire README from each site|
|clean| The cleaned version of readme_contents|
|lemma| The lemmatized version of the clean column|

## Steps to Reproduce
* Clone this repo
* Create env.py file containing your unqiue: **github_token** and **github_username** to access
* Run notebook

## Takeaways and Conclusions


## Recommendations/Next Steps

Create a funciton on github where individuals have to identify the subject that their project is looking into. Choose 

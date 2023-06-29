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

* Acquire data from repository list in Github
    * Create function using Selenium to pull the readmes from each repo
    * Create dataframe of repo names, content, and language
* Prepare data
    * Identify and handle Nulls, duplicate data, and outliers
    * Change column names to make them readable
* Explore data in search of drivers of wine quality
    * Split into Train, Validate, Test
    * Start Visualizing Data
    * Select most relevant and interesting visualizations for final report
    * Find out which features have the strongest relationship with the target variable
* Answer the following initial questions
    1. What are the most common words in READMEs?
    2. Does the length of the README vary by programming language?
    3. Do different programming languages use a different number of unique words?
    4. Are there any words that uniquely identify a programming language?
* Develop a model to predict language
    * Choose Evaluation Metric
    * Evaluate model with Train and Validate
    * Select best model
    * Evaluate best model with Test
* Draw conclusions
    

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

We decided to use accuracy as our evaluation metric and KNN for our Test model since it was the best after training on. We were able to accomplish an prediction accuracy of 81% beating our baseline prediction of 80%.

* Model
    * KNN with an n_neighbor of 9
    * Evaluation Metric: Accuracy
    * Baseline .80
    * Fit and score model with Train
    * Score model with Validate
    * KNN was our best model giving us an accuracy of .83
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

The words used in readme's can be used with a great deal of proficiency in predicting the language of that project. 

## Recommendations/Next Steps

We would create a column identifying the likely subject of each repo. Create a feature on github where individuals have to identify the subject that their project is looking into. This may User Experience and Interface engineers.

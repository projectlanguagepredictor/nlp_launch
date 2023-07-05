# NLP_Launch: Natural Language Predictor

---

## Project Description:

* GitHub is a version control system platform that projects are tracked and saved on. It acts a resource for professionals and students. README files provide summaries of those projects and are typically the first thing people see when visiting a repository. Predicting the programming language based on README files can allow for more efficient research capabilities and greater utility by users using specific languages.

* Libraries Used in a JupyterLab Notebook:
    * Pandas, Numpy, Selenium, Re, NLTK, Scipy, Requests, BeautifulSoup, Seaborn, Matplotlib, SKLearn, typing, itertools
    
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
* Explore data in search of predictors of programming language
    * Split into Train, Validate, Test
    * Start Visualizing Data
    * Select most relevant and interesting visualizations for final report
    * Find out which features have the strongest relationship with the target variable
* Develop a model to predict language
    * Choose Evaluation Metric
    * Evaluate model with Train and Validate
    * Select best model
    * Evaluate best model with Test
* Draw conclusions
    

### Wrangle

* Acquire data from GitHub
    * Data acquired from GitHub using Selenium
    * Use env.py credentials
    * Each column is a feature of the repository
    * Each row is a repository

* Prepare data
    * 740 rows × 1 column before cleaning
    * 740 rows × 4 columns after cleaning and acquiring READMEs
    * No duplicates
    * No nulls
    * Changed languages that are not 'Python', 'JavaScript', 'HTML', 'Shell', 'Java', or 'Go' to 'other'
        * other = 599
    * Cleaned all text using
        * prepare.py functions
    * Additional Stopwords used to account for all word fractions leftover from the cleaning process to get as close as possible to all true words
    * No outliers removed

### Explore

1. What are the most common words in READMEs?
    * http is the highest, with more time we would also remove that word during cleaning to be able to interpret the other words better as it is an outlier. img came in as the second highest word
        * The rest of the top 20 words: code, awesome, source, star, data, web, tool, library, href, shield, open, go, svg, html, style, badge, python, image
        
2. Does the length of the README vary by programming language?
    * Go seemed to have a spike though, but not significant.

3. Do different programming languages use a different number of unique words?
    * Python has the highest with 272
    * Java has the least with 125.
 
4. Are there any words that uniquely identify a programming language?
    * Python contains library and api
    * Python contains image, python, data, and code
    * JavaScript contains j, style, svg, and star
    * HTML contains python and html
    * Shell contains open and code
    * Java contains style, svg, and shield
    * Unsurprisingly, Go contains go
    
### Model

We decided to use accuracy as our evaluation metric and KNN for our test model since it was the best after training on. We were able to accomplish an prediction accuracy of 82% beating our baseline prediction of 81%.

* Model
    * KNN with 9 neighbors
    * Evaluation Metric: Accuracy
        * **Baseline 80.86**
    * Fit and score model with train
    * Score model with validate
    * KNN was our best model:
        * **Test Accuracy of 82.00**
    
## Data Dictionary  

| Feature | Definition|
|:--------|:-----------|
|repo| The 'author name/repo name' of the repo|
|language| The repo language used|
|readme| The cleaned contents of unqiue READMEs|

## Steps to Reproduce
* Clone this repo
* Create env.py file containing your unqiue: **github_token** and **github_username** to access
* Run notebook

## Takeaways and Conclusions

The words used in readme's can be used with a great deal of proficiency in predicting the language of that project. 

## Recommendations/Next Steps

We would create a column identifying the project type of each repo. Create a feature on github where individuals have to identify the subject that their project is looking into. This may involve user experience and interface engineers.

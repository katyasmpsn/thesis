# “Obama never said that”: Evaluating fact-checks for topical consistency and quality
## Code that goes with Katya Simpson's MS Thesis 

! Work in Progress ! I'm currently refactoring this repo to make it easier to use

* `python src/main.py` creates word type clusters over several hyperparameters. Note that this can take a while to
run! It returns the top ten word types per cluster for evaluation
*  `python src/create_clusters_with_params.py` creates all word type embeddings with your chosen parameters
* `src/docs_to_notes_2.py` returns document topic distributions
* `scripts/analysis.ipynb` is a notebook with the regression analysis
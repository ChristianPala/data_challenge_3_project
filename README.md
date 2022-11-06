# Data Challenge 3 project, SUPSI

Repository for the first data challenge 3 project, SUPSI DS &amp; AI course.

### Important notes

**We recommend to use this project with the python version 3.10.**
There is the possibility to run the files manually, or to run a main file and wait to complete. If you want to do the latter,
please run the file named "main.py" located in the "main_with_functions" branch.

Please check that all the required libraries are installed (chapter at the end of this file).
For the correct functioning of this project please copy the file located in the ./main/mod directory called "
features.py"
to the source files of the TSFEL library.
The default TSFEL library path is:

Windows: `C:\Users\<user name>\anaconda3\envs\<environment name>\Lib\site-packages\tsfel\feature_extraction\`

Mac:`~/anaconda3/envs/<environment name>/lib/python3.10/site-packages/tsfel/feature_extraction/`

Linux: `~/anaconda3/envs/<environment name>/lib/python3.10/site-packages/tsfel/feature_extraction/`

#### Libraries

Run these commands to install all the required libraries:

If you have mambaforge installed we recommend to use that, as it is much faster than conda.
`mamba install scikit-learn xgboost pandas matplotlib seaborn hyperopt tabulate numpy nltk scipy igraph tqdm networkx karateclub shap mlxtend openpyxl`
Other libraries not available with conda or mamba:
`pip install tsfel imblearn`
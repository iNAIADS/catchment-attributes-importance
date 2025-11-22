# Catchment attributes importance

## Installation

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Instructions

Copy the content of the Google Drive folder: https://drive.google.com/drive/folders/11yQuQLf0QhT1z-i2yAd3fwKWAucSfkFL in the same directory of the git.
Unzip all the zipped files.
These contains the information already computed about RFE, hyperparameter optimization, ...
Recomputing all these information for the entire GAGES-II dataset will take more than one day.
Recomputation is triggered by the lack of the correspondent files (the ones in the zipped files), otherwise the info in the files is used and the final figure will be generated.

In order to produce the final figures:
1) open the `clean_version.ipynb`
2) run the first two cells (import libraries and compile functions)
3) run the following line: `wrapper(<DATASET_NAME>, <HYDROLOGICAL_INDEX>, filtering_dict=<DICTIONARY>, n_iter_randomSearch=100, n_repeats_shap=6, n_repeats_RFECV=6)`

E.g. `wrapper("GAGES", "mean_q", filtering_dict=None, n_iter_randomSearch=100, n_repeats_shap=6, n_repeats_RFECV=6)` to produce the results for average discharge in the entire GAGES-II dataset.
Other examples are present in the code and are self explanatory.
Do not modify the values of the other parameters of the wrapper function since this will trigger new calculations

## Additional Information

For catchment cluster interpretation see file SI_2.pdf in https://data.ess-dive.lbl.gov/view/doi:10.15485/1987555

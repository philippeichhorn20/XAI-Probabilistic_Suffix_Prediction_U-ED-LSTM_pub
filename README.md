# README

This Repository is a fork of the probabilistic suffix predictor with an extra folder for explainability analysis. 


## Setting Up the Python Environment with Pipenv

This project uses `pipenv` for managing Python dependencies. Follow the steps below to set up the virtual environment and install the necessary packages using the provided `Pipfile`.

### Prerequisites
Make sure you have Python and Pipenv installed.

### Setup Instructions

1. **Create the Virtual Environment**:
    
    ```bash
    pipenv install
    ```

2. **Activate the Virtual Environment**:
    
    ```bash
    pipenv shell
    ```

3. **Run the Project**: Inside the virtual environment, you have the Python packages installed for running the code.


## Explainability

The explainability implementations can be found in the folder src/interpretability. They include a wide array of implementations that are not limited to the techniques that ended up in the thesis. For those elements that were used in the thesis, please see the next sections.

### Model Creation

In the folder ../improved_pipeline you have a four-layer folder structure: {model}/{dataset}/{old, \[improved\]}/{Loader, Training} that provide a loading and training notebook. First run the loading notebook, then the training. The resulting pkl will be stored either in the same folder or in a separate pkl folder.

### Configurations

To run the analysis notebooks, reference the model and data .pkl's that you have produced in the previous step in the configuration files of the respective data set in the ../config folder. Then you have the ability to set an array of models that can then be selected in the analysis notebooks to quickly switch between different models. (Usually one for Camargo and two, one improved and one old, for Henryk).

### Analysis

The content used for the analysis section in the thesis can be found under ../notebooks.
Each of the analyzed models have their own subfolder. ../henryk refers to the U-ED-LSTM and ../camargo refers to the model by Camargo et al. Within those are subfolders for each of the data sets. Within these folders are the notebooks for different parts of the analysis.

They can be run independently of each other. Only the gateway notebook \*_gateway_decision_analysis.ipynb needs to be run before the other \*_gateway_\* notebooks.







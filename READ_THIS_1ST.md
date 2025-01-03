
# A Novel Plug-in Module for Fine-Grained Visual Classification
#Code adapted by Eduardo Bedin

The original implementation was provided by Po-Yung Chou, as detailed in the paper: A Novel Plug-in Module for Fine-Grained Visual Classification.

GitHub Repository: https://github.com/chou141253/FGVC-PIM

For comprehensive information on the original code and its operational principles, please refer to the paper and the `readme_original.md` file.

In this adapted version, several modifications have been implemented to enhance functionality:

* **Cross-Validation Integration**: A cross-validation technique has been incorporated to partition the dataset. The code now automatically segregates the data into training, validation, and test sets.

**Instructions for Execution**:

1. **Data Preparation**: Organize your dataset by placing the images in the ./data/all directory, ensuring that each class has its own subdirectory.

2. **Configuration Settings**:

* In the CUB200_SwinT.yaml configuration file, set the num_fold variable to the desired number of folds for cross-validation.

* Ad*just the val_ratio variable to define the percentage of data allocated for the validation set.

3. **Generating Folds**: Execute the following command in the terminal to create the data folds:

`python3 Split1.py --c ../configs/CUB200_SwinT.yaml`

This command constructs the folds, organizing the class folders with their respective image allocations.

4. **Pre-Execution Cleanup**: Before each run, ensure that the following directories are deleted if they exist: `folds, train, test, val, and results`.

For detailed explanations of each variable, please consult the comments within the `configs/CUB200_SwinT.yaml` file.

Note: The comments are provided in Portuguese.



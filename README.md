# derivatives_template_code

Scripts for preprocessing (longitudinal) fMRI data

## Installation

### Prerequisites

* A BIDS dataset in [DataLad dataset](http://docs.datalad.org/en/stable/generated/datalad.api.Dataset.html), which you can create using our [BIDS template](https://github.com/SkeideLab/bids_template) or download from [OpenNeuro](https://openneuro.org) (currently untested)

### If you already have a `derivatives` sub-dataset

* If using our BIDS template, your BIDS dataset probably already has a sub-dataset called `derivatives` installed
* You can install the scripts from this repository using the following command from your main BIDS directory:

    ```bash
    datalad install -d . -s https://github.com/SkeideLab/derivatives_template_code.git derivatives/code
    ```

### If you don't have a `derivatives` sub-dataset

* If you don't have a `derivatives` sub-dataset installed in your BIDS dataset, you can do so using the following command from your main BIDS directory:

    ```bash
    datalad create -d . -c text2git derivatives
    ```

* Next, you can install the scripts from this repository using the following command:

    ```bash
    datalad install -d . -s https://github.com/SkeideLab/derivatives_template_code.git derivatives/code
    ```

## Usage

...

## Processing details

...

# Selection of relevant features for alpha rhythm classification

In this work, we implement a feature selection technique utilizing $R^2$ feature maps to identify the most significant frequencies and channels used for classification of alpha rhythms.  Our methodology was applied to data from a subject in the EEG MMIDB database, the resulting $R^2$ map is visualized and analysed. We show that the most informative set of frequencies lie around the frequency of $9.9$Hz in the visual cortex of the brain. 

For more detail please read: [Alpha rhythm feature selection](./feature_selection_eeg.pdf)


## Setup

In order to run the program user must do the following:

1. Create a virtual environment:

    `python3 -m venv venv`

2. Activate virtual environment:

    `. venv/bin/activate`

3. Install dependencies:

    `pip install -r requirements.txt`


## Usage

You can run the code using:

`python3 main.py`.

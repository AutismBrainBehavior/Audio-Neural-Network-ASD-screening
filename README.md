[![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg

## ADOS Audio Neural Network for ASD Screening

This repository consists of the code used for carrying out classification of Autism Spectrum Disorder (ASD) using audio samples extracted from ADOS clinical examination recordings ([Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308388)). Conversion of ADOS recordings into purely audio files was done using `utilities/vac.py` file which requires `ffmpeg`. The neural network's code was inspired from [CNN Audio Neural Network Classifier](https://github.com/AmritK10/Urban-Sound-Classification) and modified for training and prediction over audio dataset extracted from ADOS clinical examination recordings. The neural network was originally used for classification of [Urban-8K](https://urbansounddataset.weebly.com/urbansound8k.html) audio dataset.

An illustration of Audio Feature extracted from each of the audio clip can be observed as follows:
<p align="center">
<img src="https://github.com/nshreyasvi/Audio-Neural-Network-ASD-screening/blob/main/illustrations/Figure_1.png" width="50%" height="50%">
</p>

## Usage with Anaconda3
The following instructions can be used to run the neural network using [Anaconda](https://www.anaconda.com/):
- Create a new anaconda environment using `conda create -n audio_nn python=3.6 -y`
- Install conda dependencies using `conda install librosa ffmpeg pandas tensorflow-gpu==1.15.0 seaborn matplotlib keras`
- Install pip dependencies using `pip install opencv-contrib-python==4.1.2.30 imutils`
- In order to train the neural network all of the audio files must be stored in the folder `very_large_data`.
- The labels for each of the audio files stored in the folder `very_large_data` are to be stored in a `training.csv` or `testing.csv` file in the folder `labels_very_large_data`. A sample format of the same can be found in the `labels_very_large_data` folder.
- In order to train the neural network run `python train_net.py`.
- The extracted features are stored in `.npy` format and the trained model can be seen saved as `trained_model.h5` in the same folder.
- In order to carry out prediction, use `python prediction.py` which will generate 3 `.csv` files namely `class_preds.csv`, `pred.csv` and `truth.csv`.
- Open the `testing.csv` labels file and sort them in ascending order. The `class_preds.csv` file consists of prediction class for the respective labels in the `testing.csv` file, `pred.csv` contains the columns `prob_ASD` (Prediction confidence of ASD class) and `prob_TD` (Prediction Confidence of TD class) and `truth.csv` containing truth labels (dummy labels) given to the testing sample.

In our study we carried out prediction by splitting each of the audio samples into 10-second clips to ensure better training and less overfitting. We then carried out prediction over each of these 10-second clips and aggregated them for each of the audio files of subjects to obtain the final prediction. In order to check for the overall prediction class, we checked the mean prediction confidence over the entire audio clip of a subject and checking if it was higher for ASD or TD.

An example of the training log for the given neural network can be observed as follows:

<p align="center">
<img src="https://github.com/nshreyasvi/Audio-Neural-Network-ASD-screening/blob/main/illustrations/history.png" width="50%" height="50%">
</p>

Illustrations for different audio features used for carrying out training and prediction of audio samples can be done by using `audio_features.py` file inside `utilities` folder by defining the path to the audio file whose different features have to be illustrated.

## Citation
```
@article{10.1371/journal.pone.0308388,
    doi = {10.1371/journal.pone.0308388},
    author = {Natraj, Shreyasvi AND Kojovic, Nada AND Maillart, Thomas AND Schaer, Marie},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Video-audio neural network ensemble for comprehensive screening of autism spectrum disorder in young children},
    year = {2024},
    month = {10},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0308388},
    pages = {1-20},
    abstract = {A timely diagnosis of autism is paramount to allow early therapeutic intervention in preschoolers. Deep Learning tools have been increasingly used to identify specific autistic symptoms. But they also offer opportunities for broad automated detection of autism at an early age. Here, we leverage a multi-modal approach by combining two neural networks trained on video and audio features of semi-standardized social interactions in a sample of 160 children aged 1 to 5 years old. Our ensemble model performs with an accuracy of 82.5% (F1 score: 0.816, Precision: 0.775, Recall: 0.861) for screening Autism Spectrum Disorders (ASD). Additional combinations of our model were developed to achieve higher specificity (92.5%, i.e., few false negatives) or sensitivity (90%, i.e. few false positives). Finally, we found a relationship between the neural network modalities and specific audio versus video ASD characteristics, bringing evidence that our neural network implementation was effective in taking into account different features that are currently standardized under the gold standard ASD assessment.},
    number = {10},
}
```

```
@article{Kojovic2021,
	title        = {Using 2D video-based pose estimation for automated prediction of autism spectrum disorders in young children},
	author       = {Kojovic, Nada and Natraj, Shreyasvi and Mohanty, Sharada Prasanna and Maillart, Thomas and Schaer, Marie},
	year         = 2021,
	month        = {Jul},
	day          = 23,
	journal      = {Scientific Reports},
	volume       = 11,
	number       = 1,
	pages        = 15069,
	doi          = {10.1038/s41598-021-94378-z},
	issn         = {2045-2322},
	url          = {https://doi.org/10.1038/s41598-021-94378-z}
}
```

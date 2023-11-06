# Datathon Repository

Welcome to the Datathon repository by Data Driven Swifties!

## About

We are a team of undergraduate students participating in the United Nations Datathon. Our goal is to leverage data to address pressing global issues and contribute to the Sustainable Development Goals (SDGs).

This project aims to explore the influence of social variables on salary levels in Uruguay, employing interpretable machine learning models trained on data from the Instituto Nacional de Estad ́ıstica (INE). This analysis is not solely an academic exercise but a strategic approach aimed at equipping policymakers with data-driven insights. Through the identification and understanding of key social variables. The goal is to pave the way for interventions that not only mitigate the symptoms of economic inequality but also target and ameliorate its root causes. Thereby, enhancing economic equity becomes not just an aspirational target but an achievable milestone through informed policy-making.

## Repository Structure

The repository includes several folders and files. Here's a brief overview of the most important ones:

- `data`: This directory contains all the datasets in zip format for space efficiency. For the script to run, all data should be extracted and present in this repository as .csv files.
- `main.py`: The main Python script for the project. It includes two models, one based on TabNet (transformer) and another based on Random Forest with AdaBoost. To train a specific model, one should toggle the respective function call within the script.
- `model_training.py`: A Python script for training machine learning models.
- `data_preprocessing.py`: A Python script responsible for preprocessing the data to be used in the models.
- `plotting_map.py`: This script uses GeoJSON to link data to department coordinates for visualization purposes.
- `model_weights`: This directory contains saved model weights from both the TabNet and the tree-based models. The weights for the tree model are illustrative, as the actual weights were too large(+600 MB).
- `requirements-dev.txt`: A text file listing the development dependencies for the project.
- `requirements.txt`: A text file listing the dependencies required to run the project.

## Usage

To use this repository, you will need to install the required dependencies listed in `requirements.txt`. If you are contributing to the development, you should also install the dependencies in `requirements-dev.txt`.

```bashmight
pip install -r requirements.txt
# If developing:
pip install -r requirements-dev.txt
```

Then, you can run the main script using Python:

```bash
python main.py tabnet
```
or 

```bash
python main.py trees
```

# Contact

For more information or if you have any questions regarding this project, feel free to watch the repository for updates or reach out directly. You can contact us at our personal emails:

- Federico de Bello: [fe.debello13@gmail.com](mailto:fe.debello13@gmail.com)
- Juan Pablo Sotelo Silva: [jpsotelosilva@gmail.com](mailto:jpsotelosilva@gmail.com).

We welcome any inquiries or suggestions and look forward to engaging with fellow data enthusiasts and contributors.

# Acknowledgements

We extend our heartfelt gratitude to the United Nations and the organizers of the Datathon for providing us with this incredible opportunity. The event has not only allowed us to apply our knowledge and skills to real-world problems but also to learn and grow as future data scientists. We appreciate the hard work and dedication that went into coordinating this event, which has brought together like-minded individuals from around the globe to contribute towards the Sustainable Development Goals (SDGs). Our participation in this datathon has been an enriching experience, and we are thankful for the support and resources provided to us throughout this journey.

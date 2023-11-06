# Datathon Repository

Welcome to the Datathon repository by Data Driven Swifties!

## About

We are a team of undergraduate students participating in the United Nations Datathon. Our goal is to leverage data to address pressing global issues and contribute to the Sustainable Development Goals (SDGs). 


## Repository Structure

The repository includes several folders and files. Here's a brief overview of the most important ones:

- `data`: This directory has all the dataset. They are in zip format for space reasons. In order for the script to run all the data should be in dis repository as a .csv file.
- `main.py`: The main Python script for the project. It includes two models, one based on transformer and another based on Random Forest and ADABooster. In order to train each model one should change which function from 
- `model_training.py`: A Python script for training machine learning models.
- `model_weights`: This directory contain saved model weights from the transformer and the tree model. The weights from the tree model are just toy weights, since the actual weights were too heavy.
- `requirements-dev.txt`: A text file listing the development dependencies for the project.
- `requirements.txt`: A text file listing the dependencies required to run the project.

## Usage

To use this repository, you would typically need to install the required dependencies listed in `requirements.txt` and possibly `requirements-dev.txt` if you are developing further.

```bashmight
pip install -r requirements.txt
# If developing:
pip install -r requirements-dev.txt
```

Then, you can run the main script using Python:

```bash
python main.py
```

# Contact

For more information or if you have any questions regarding this project, feel free to watch the repository for updates or reach out directly. You can contact me at my personal email: [fe.debello13@gmail.com](mailto:fe.debello13@gmail.com).

We welcome any inquiries or suggestions and look forward to engaging with fellow data enthusiasts and contributors.


# Acknowledgements

We extend our heartfelt gratitude to the United Nations and the organizers of the Datathon for providing us with this incredible opportunity. The event has not only allowed us to apply our knowledge and skills to real-world problems but also to learn and grow as future data scientists. We appreciate the hard work and dedication that went into coordinating this event, which has brought together like-minded individuals from around the globe to contribute towards the Sustainable Development Goals (SDGs). Our participation in this datathon has been an enriching experience, and we are thankful for the support and resources provided to us throughout this journey.

# Movie Success Analysis and Prediction Tool

This project explores the factors influencing a movie's success, both critically and commercially, using statistical analysis and machine learning techniques. The goal is to provide insights into what makes a movie successful in terms of IMDb ratings and box office revenue, and to offer a prediction tool that estimates these outcomes based on key features like budget, genre, and cast.

## Table of Contents
- [Project Description](#project-description)
- [Methodology](#methodology)
- [Features](#features)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Installation](#installation)

## Methodology
Our analysis involves collecting data from a survey on favorite movies, enriched with detailed information from the OMDb and TMDb APIs. We used statistical methods like correlation tests, linear regression, and ANOVA to identify key factors influencing IMDb ratings and box office revenue. The prediction tool leverages machine learning algorithms to forecast these outcomes based on input parameters.

## Features
- Predict IMDb ratings and box office revenue
- Analyze factors like budget, genre, cast, and director impact
- Visualize data trends and predictions

## Usage
To use the prediction tool:
1. Clone the repository to your local machine.
2. Run the tool using the provided scripts or Jupyter notebooks.
3. Input details like budget, genre, director, and actors.
4. View the predicted IMDb rating and box office revenue.

## Repository Structure
- `/src`: Analysis scripts
- `/data`: Datasets (excluded from Git with `.gitignore`)
- `/notebooks`: Jupyter Notebooks
- `/docs`: Documentation and methodology
- `/slides`: Presentation slides
- `/tools`: Prediction tool files

## Installation
To set up the project locally, follow these steps:
1. Clone the repository: `git clone https://github.com/elorakerboua/MovieSuccessAnalysis.git`
2. Navigate to the project directory: `cd MovieSuccessAnalysis`
3. Install the required dependencies: `pip install -r requirements.txt`

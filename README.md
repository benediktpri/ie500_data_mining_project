# IE500 Data Mining Project - Diabetes Risk Prediction

This repository contains all work for the student project as part of the **IE500 Data Mining Course** at **University of Mannheim** in 2024.
The project involves the analysis of the **Diabetes Health Indicators Dataset** published on [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data). The [original data](https://www.cdc.gov/brfss/annual_data/annual_2015.html) originates from the Behavioral Risk Factor Surveillance System (BRFSS), an annual health-related survey conducted by the CDC. The analysis focuses on identifying key risk factors associated with diabetes and developing predictive models to assess diabetes risk based on survey responses.

Given the high prevalence of diabetes in the United States, the creation of accurate predictive models can help with early detection and inform public health strategies, potentially mitigating the serious health complications associated with the disease. Using various machine learning techniques, this project aims to explore how survey data can be leveraged to enhance diabetes prediction and risk assessment.


Team 11: *name tbd*

Group members:
Philipp Gänz,
Salome Heckenthaler,
Patricia Paskuda,
Benedikt Prisett,
Matthias Fast


## Installation

### Prerequisites
- Python 3.9+

### Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/benediktpri/ie500_data_mining_project.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd ie500_data_mining_project
    ```

3. **Create a virtual environment** (optional, but recommended):
    ```bash
    python -m venv venv
    ```

4. **Activate the virtual environment**:
    ```bash
    source venv/bin/activate # On macOS/Linux
    venv\Scripts\activate # On Windows
    ```

5. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

6. **Set up pre-commit hooks**:
Set up the Git hooks defined in the `.pre-commit-config.yaml` file by running:
    ```bash
    pre-commit install
    ```

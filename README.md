# Movie Data Pipeline

This repository implements a movie data processing pipeline using `Spark`, `Delta Lake`, and `MLflow` for experiments tracking. The pipeline consists of three key steps:

1. [Download](https://drive.google.com/file/d/1r5NaERzCJXFg14J2Dvnyk4dwdY11Nikn/view?usp=drive_link) data, put to `src/data`, read and write to the `Bronze` layer. 
2. Clean and transform data, then store it in the `Silver` layer.
3. Train a Logistic Regression model using a Spark ML Pipeline.

## Prerequisites

- Docker
- Docker Compose

## Setup and Run

1. Build the Docker image:
   ```bash
   make build
   ```

2. Start containers:
   ```bash
   make up
   ```

3. Run the pipeline:
   ```bash
   make run_all
   ```

4. Stop containers when done:
   ```bash
   make down
   ```

## MLflow UI

You can open your browser and navigate to [http://localhost:5001](http://localhost:5001) to view the `MLflow` UI server.

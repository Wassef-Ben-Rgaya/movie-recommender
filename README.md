# Movie Recommender System

A machine learning-powered movie recommendation platform built with Python, FastAPI, PostgreSQL, and Docker. It generates personalized movie recommendations using collaborative filtering and exposes them through a REST API.

## Features

- Personalized movie recommendations based on user ratings
- Three recommendation approaches: SVD, ALS, and Neural Collaborative Filtering
- REST API built with FastAPI and interactive Swagger documentation
- PostgreSQL database for users, movies, and ratings
- Redis caching for recommendations
- Kafka-based interaction streaming
- Airflow orchestration for machine learning workflows
- MLflow experiment tracking
- Interactive Streamlit dashboard
- Docker Compose environment
- AWS Lambda, API Gateway, and S3 integration
- Automated tests with pytest

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python 3.12 |
| API | FastAPI, Uvicorn |
| Database | PostgreSQL |
| Data processing | pandas, NumPy |
| Recommendation models | Surprise SVD, ALS, Neural CF |
| Evaluation | RMSE, MAE |
| Caching | Redis |
| Streaming | Apache Kafka |
| Orchestration | Apache Airflow |
| Experiment tracking | MLflow |
| Dashboard | Streamlit |
| Deployment | Docker, Docker Compose, AWS |
| Testing | pytest |

## Dataset

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).

- **943 users**
- **1,682 movies**
- **100,000 ratings**

Download and extract the dataset according to the ingestion script's expected directory structure. The dataset and generated model artifacts are not included in this repository.

## Architecture

```text
MovieLens 100K Dataset
          |
          v
    Data Ingestion
          |
          v
      PostgreSQL
          |
          v
  Model Training & Evaluation
     SVD / ALS / Neural CF
          |
          v
    FastAPI REST API
       |         |
       v         v
     Redis    Recommendations
       Cache      |
                  v
            Streamlit Dashboard

Additional components:
Airflow | MLflow | Kafka | AWS Lambda | API Gateway | S3
```

## Getting Started

### Prerequisites

- Python 3.12
- Git
- Docker Desktop with Docker Compose
- A `.env` configuration based on `.env.example`

### 1. Clone the repository

```bash
git clone https://github.com/Wassef-Ben-Rgaya/movie-recommender.git
cd movie-recommender
```

### 2. Configure environment variables

Create a local `.env` file using `.env.example` as a reference.

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and provide appropriate values for the required database credentials and application secrets. Use unique secret values for local services.

**Never commit `.env` or publish passwords, access keys, or other secrets.**

### 3. Start the services

Build and start the Docker Compose environment:

```powershell
docker compose up -d --build
```

Check service status:

```powershell
docker compose ps
```

View logs if a service fails to start:

```powershell
docker compose logs --tail 100
```

Some services may require the dataset, model artifacts, or additional initialization steps before all features are available. Follow the relevant project configuration and scripts.

### 4. Access the application

| Service | Local URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |
| Streamlit dashboard | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

Access a service only when its corresponding container is running and its port is configured in Docker Compose.

## API Usage

### Get personalized recommendations

**Endpoint**

```http
GET /recommendations/{user_id}?n=10
```

Example using PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/recommendations/1?n=10"
```

Example response:

```json
{
  "user_id": 1,
  "count": 2,
  "recommendations": [
    {
      "movie_id": 475,
      "title": "Trainspotting (1996)",
      "predicted_rating": 4.76
    },
    {
      "movie_id": 483,
      "title": "Casablanca (1942)",
      "predicted_rating": 4.74
    }
  ]
}
```

This response is illustrative. The actual number and content of recommendations depend on the requested limit, available model artifacts, and the API implementation.

### Health check

```http
GET /health
```

Expected response:

```json
{
  "status": "healthy"
}
```

## Model Evaluation

The current SVD evaluation reports the following results:

| Metric | Result |
|---|---:|
| RMSE | 0.9365 |
| MAE | 0.7358 |

These are preliminary results from the current evaluation workflow. For a reliable comparison, ensure that the test set is excluded from model training and compare SVD, ALS, and Neural CF using consistent data splits and evaluation metrics.

## Automated Testing

Run the test suite from the project root:

```powershell
python -m pytest tests/ -v
```

**Latest verified result:** 10 tests passed, with one deprecation warning.

The tests cover API behavior, recommendation responses, request validation, and model-related functionality.

## Project Structure

```text
movie-recommender/
├── airflow/
│   └── dags/
├── aws/
├── data/                  # Local data and model artifacts
├── docker/
├── sql/
├── src/
│   ├── api/
│   ├── cache/
│   ├── dashboard/
│   ├── data/
│   ├── kafka/
│   └── models/
├── tests/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Security Notes

- Keep `.env` and all credentials out of version control.
- Use least-privilege AWS IAM policies.
- Configure unique secrets for local and deployed services.
- Do not expose development credentials in production.
- Keep datasets, trained artifacts, and generated deployment archives out of Git unless intentionally required.

## Future Improvements

- Establish a leakage-free evaluation pipeline.
- Compare SVD, ALS, and Neural CF on consistent test splits.
- Add ranking metrics such as Precision@K and Recall@K.
- Expand API integration and automated test coverage.
- Add monitoring, logging, and performance benchmarks.
- Document dataset acquisition, model training, and AWS deployment in detail.

## License

No license has been specified yet. Add a `LICENSE` file before allowing others to reuse, modify, or redistribute the project.

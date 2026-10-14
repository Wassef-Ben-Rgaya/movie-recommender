# Movie Recommender System

A movie recommendation platform built with Python, FastAPI, PostgreSQL, and machine learning. The system generates personalized movie recommendations based on user ratings and exposes them through a REST API.

## Features

- Personalized movie recommendations for individual users
- REST API built with FastAPI
- PostgreSQL database for users, movies, and ratings
- Collaborative filtering models: SVD, ALS, and Neural Collaborative Filtering
- Docker Compose deployment for the API and database
- Interactive API documentation with Swagger UI
- Automated tests with pytest

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| API | FastAPI, Uvicorn |
| Database | PostgreSQL |
| Data Processing | pandas, NumPy |
| Recommendation Models | Surprise SVD, ALS, Neural Collaborative Filtering |
| Machine Learning Evaluation | RMSE, MAE |
| Containers | Docker, Docker Compose |
| Testing | pytest |
| API Documentation | Swagger UI |

## Dataset

The project uses the MovieLens 100K dataset.

- **Users:** 943
- **Movies:** 1,682
- **Ratings:** 100,000

## Architecture

```text
MovieLens Dataset
       |
       v
Data Ingestion
       |
       v
PostgreSQL Database
       |
       v
Recommendation Models
(SVD / ALS / Neural CF)
       |
       v
FastAPI REST API
       |
       v
Personalized Movie Recommendations
```

## Getting Started

### Prerequisites

- Python 3.12
- Docker Desktop
- Docker Compose
- Git

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd movie-recommender
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the actual URL of your repository.

### 2. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

Configure the PostgreSQL database name, username, password, host, and port according to your local setup.

Do not commit your `.env` file or expose database credentials.

### 3. Start the application

```bash
docker compose up -d --build
```

Check the running containers:

```bash
docker compose ps
```

### 4. Open the API documentation

Swagger UI:

http://localhost:8000/docs

Health endpoint:

http://localhost:8000/health

## API Usage

### Get personalized recommendations

**Endpoint**

```http
GET /recommendations/{user_id}?n=10
```

Example:

```bash
curl "http://localhost:8000/recommendations/1?n=10"
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

The response above is illustrative; the number of returned movies depends on the requested limit and available recommendations.

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

The SVD model is evaluated using:

- **RMSE:** 0.9365
- **MAE:** 0.7358

These metrics are preliminary results from the current evaluation script. A strictly independent evaluation should ensure that the test ratings were not used to train the saved model.

## Automated Testing

Run the test suite:

```bash
python -m pytest -v
```

Latest verified result:

```text
10 passed, 1 warning
```

The tests cover API endpoints, recommendation responses, invalid request limits, model artifacts, and SVD prediction validity.

## Project Structure

```text
movie-recommender/
├── data/
│   ├── raw/
│   └── processed/
│       └── models/
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
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Future Improvements

- Establish a leakage-free evaluation pipeline
- Compare SVD, ALS, and Neural Collaborative Filtering using consistent metrics
- Add recommendation quality metrics such as Precision@K and Recall@K
- Improve API integration and automated test coverage
- Add monitoring and performance benchmarks
- Enhance the user interface and deployment documentation

## License

Add a license file if you intend to distribute this project publicly.

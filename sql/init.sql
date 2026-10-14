
-- =========================================
-- Movie Recommender System
-- PostgreSQL 16
-- =========================================

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    age INTEGER CHECK (age >= 0),
    gender VARCHAR(10),
    occupation VARCHAR(100),
    zip_code VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    release_date DATE,
    genres TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ratings (
    rating_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    movie_id INTEGER NOT NULL REFERENCES movies(movie_id),
    rating NUMERIC(2,1) NOT NULL CHECK (rating BETWEEN 1 AND 5),
    rated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    movie_id INTEGER NOT NULL REFERENCES movies(movie_id),
    score NUMERIC(10,6) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, movie_id, model_name)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ratings_user
    ON ratings(user_id);

CREATE INDEX IF NOT EXISTS idx_ratings_movie
    ON ratings(movie_id);

CREATE INDEX IF NOT EXISTS idx_ratings_score
    ON ratings(rating);

CREATE INDEX IF NOT EXISTS idx_recommendations_user_score
    ON recommendations(user_id, score DESC);

CREATE INDEX IF NOT EXISTS idx_movies_title
    ON movies(title);

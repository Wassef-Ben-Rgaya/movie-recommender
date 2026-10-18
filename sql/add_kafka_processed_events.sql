
-- Durable Kafka event deduplication.
-- PostgreSQL remains the source of truth.

CREATE TABLE IF NOT EXISTS kafka_processed_events (
    event_id       VARCHAR(64) PRIMARY KEY,
    topic          VARCHAR(255) NOT NULL,
    partition_id   INTEGER NOT NULL,
    message_offset BIGINT NOT NULL,
    user_id        INTEGER NOT NULL REFERENCES users(user_id),
    movie_id       INTEGER NOT NULL REFERENCES movies(movie_id),
    processed_at   TIMESTAMP WITHOUT TIME ZONE NOT NULL
                   DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT kafka_processed_events_partition_check
        CHECK (partition_id >= 0),

    CONSTRAINT kafka_processed_events_offset_check
        CHECK (message_offset >= 0),

    CONSTRAINT kafka_processed_events_coordinates_key
        UNIQUE (topic, partition_id, message_offset)
);

CREATE INDEX IF NOT EXISTS idx_kafka_processed_events_processed_at
    ON kafka_processed_events (processed_at);

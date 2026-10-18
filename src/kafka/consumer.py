
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import redis
from kafka import KafkaConsumer, TopicPartition
from kafka.errors import KafkaError
from kafka.structs import OffsetAndMetadata
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.cache.redis_client import (
    get_redis_client,
    invalidate_user_recommendations,
)
from src.data.db import get_engine


BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"
)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

TOPIC = "user_interactions"
GROUP_ID = "movie-recommender-consumer"
MAX_HISTORY_ITEMS = 100

INSERT_PROCESSED_EVENT = text("""
    INSERT INTO kafka_processed_events (
        event_id,
        topic,
        partition_id,
        message_offset,
        user_id,
        movie_id
    )
    VALUES (
        :event_id,
        :topic,
        :partition_id,
        :message_offset,
        :user_id,
        :movie_id
    )
    ON CONFLICT DO NOTHING
    RETURNING event_id
""")

INSERT_RATING = text("""
    INSERT INTO ratings (user_id, movie_id, rating, rated_at)
    VALUES (:user_id, :movie_id, :rating, :rated_at)
    ON CONFLICT (user_id, movie_id)
    DO UPDATE SET
        rating = EXCLUDED.rating,
        rated_at = EXCLUDED.rated_at
""")

CHECK_USER = text("""
    SELECT EXISTS (
        SELECT 1 FROM users WHERE user_id = :user_id
    )
""")

CHECK_MOVIE = text("""
    SELECT EXISTS (
        SELECT 1 FROM movies WHERE movie_id = :movie_id
    )
""")

# This Redis script deduplicates history entries and updates the
# latest interaction atomically. PostgreSQL remains authoritative
# for durable event deduplication.
RECORD_INTERACTION_SCRIPT = """
if redis.call('SET', KEYS[1], '1', 'NX') then
    redis.call('SET', KEYS[2], ARGV[1])
    redis.call('LPUSH', KEYS[3], ARGV[2])
    redis.call('LTRIM', KEYS[3], 0, tonumber(ARGV[3]) - 1)
    return 1
end

return 0
"""


def validate_event(event):
    """Validate and normalize a Kafka interaction."""
    if not isinstance(event, dict):
        raise ValueError("L'événement doit être un objet JSON.")

    user_id = event.get("user_id")
    movie_id = event.get("movie_id")
    raw_rating = event.get("rating")

    if (
        isinstance(user_id, bool)
        or not isinstance(user_id, int)
        or user_id <= 0
    ):
        raise ValueError("user_id doit être un entier positif.")

    if (
        isinstance(movie_id, bool)
        or not isinstance(movie_id, int)
        or movie_id <= 0
    ):
        raise ValueError("movie_id doit être un entier positif.")

    if isinstance(raw_rating, bool):
        raise ValueError("La note doit être numérique.")

    try:
        rating = float(raw_rating)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("La note doit être numérique.") from exc

    if not math.isfinite(rating) or not 0.5 <= rating <= 5.0:
        raise ValueError("La note doit être comprise entre 0.5 et 5.0.")

    if not math.isclose(rating * 2, round(rating * 2), abs_tol=1e-9):
        raise ValueError("La note doit être un multiple de 0.5.")

    return {
        "user_id": user_id,
        "movie_id": movie_id,
        "rating": rating,
    }


def make_event_identity(event, message):
    """Create a stable identity for an event."""
    supplied_id = event.get("event_id")

    if supplied_id is not None:
        if (
            isinstance(supplied_id, bool)
            or not isinstance(supplied_id, (str, int))
            or not str(supplied_id).strip()
        ):
            raise ValueError(
                "event_id doit être une chaîne ou un entier non vide."
            )
        identity = f"event-id:{supplied_id}"
    else:
        identity = (
            f"kafka:{message.topic}:"
            f"{message.partition}:{message.offset}"
        )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def commit_message_offset(consumer, message):
    """Commit only the next offset for this partition."""
    partition = TopicPartition(message.topic, message.partition)
    consumer.commit(
        offsets={
            partition: OffsetAndMetadata(message.offset + 1, "")
        }
    )


def process_event(engine, cache, event, message):
    """Persist an event once, then update Redis."""
    interaction = validate_event(event)
    identity = make_event_identity(event, message)

    user_id = interaction["user_id"]
    movie_id = interaction["movie_id"]

    # Validate JSON serializability before making durable changes.
    normalized_event = {
        "user_id": user_id,
        "movie_id": movie_id,
        "rating": interaction["rating"],
    }

    for field in ("event_id", "timestamp", "event_type"):
        if field in event:
            value = event[field]
            if value is not None and not isinstance(
                value, (str, int, float, bool)
            ):
                raise ValueError(f"{field} a un type invalide.")
            normalized_event[field] = value

    serialized_event = json.dumps(
        normalized_event,
        allow_nan=False,
    )

    history_entry = {
        **normalized_event,
        "_consumer_event_id": identity,
    }
    serialized_history_entry = json.dumps(
        history_entry,
        allow_nan=False,
    )

    rated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    # The event marker and rating upsert are committed together.
    # If the marker already exists, ratings is not modified.
    with engine.begin() as connection:
        user_exists = connection.execute(
            CHECK_USER, {"user_id": user_id}
        ).scalar_one()

        movie_exists = connection.execute(
            CHECK_MOVIE, {"movie_id": movie_id}
        ).scalar_one()

        if not user_exists:
            raise ValueError(f"Utilisateur inconnu : {user_id}.")

        if not movie_exists:
            raise ValueError(f"Film inconnu : {movie_id}.")

        inserted = connection.execute(
            INSERT_PROCESSED_EVENT,
            {
                "event_id": identity,
                "topic": message.topic,
                "partition_id": message.partition,
                "message_offset": message.offset,
                "user_id": user_id,
                "movie_id": movie_id,
            },
        ).scalar_one_or_none()

        is_new_event = inserted is not None

        if is_new_event:
            connection.execute(
                INSERT_RATING,
                {
                    **interaction,
                    "rated_at": rated_at,
                },
            )

    # A durable duplicate does not rewrite the rating, but Redis
    # may still need a retry after an earlier Redis failure.
    # The marker below prevents repeated history insertion.
    marker_key = f"kafka:processed:{identity}"
    interaction_key = f"interaction:{user_id}:{movie_id}"
    history_key = f"user:{user_id}:interactions"

    cache.eval(
        RECORD_INTERACTION_SCRIPT,
        3,
        marker_key,
        interaction_key,
        history_key,
        serialized_event,
        serialized_history_entry,
        MAX_HISTORY_ITEMS,
    )

    deleted = invalidate_user_recommendations(user_id)

    return (
        user_id,
        movie_id,
        interaction["rating"],
        int(deleted),
        is_new_event,
    )


def main():
    engine = get_engine()
    cache = get_redis_client()
    consumer = None

    try:
        cache.ping()
        print(f"Redis connecté : {REDIS_HOST}:{REDIS_PORT}")

        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            group_id=GROUP_ID,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=None,
        )

        print(
            f"Consumer démarré sur {TOPIC}. "
            "En attente des interactions..."
        )

        for message in consumer:
            try:
                if message.value is None:
                    raise ValueError("Message Kafka vide (tombstone).")

                try:
                    event = json.loads(message.value.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Message JSON invalide : {exc}"
                    ) from exc

                result = process_event(engine, cache, event, message)
                user_id, movie_id, rating, deleted, is_new = result

            except (ValueError, TypeError, KeyError) as exc:
                # Invalid records are skipped deliberately to avoid
                # blocking the partition forever.
                print(
                    f"Événement invalide ignoré "
                    f"(partition={message.partition}, "
                    f"offset={message.offset}) : {exc}"
                )
                try:
                    commit_message_offset(consumer, message)
                except KafkaError as commit_exc:
                    print(
                        f"Échec du commit de l'événement invalide : "
                        f"{commit_exc}. Arrêt du consumer."
                    )
                    break
                continue

            except (redis.RedisError, SQLAlchemyError, KafkaError) as exc:
                print(
                    f"Erreur de traitement "
                    f"(partition={message.partition}, "
                    f"offset={message.offset}) : "
                    f"{type(exc).__name__}: {exc}. "
                    "Offset non confirmé ; arrêt du consumer."
                )
                break

            except Exception as exc:
                print(
                    f"Erreur inattendue "
                    f"(partition={message.partition}, "
                    f"offset={message.offset}) : "
                    f"{type(exc).__name__}: {exc}. "
                    "Offset non confirmé ; arrêt du consumer."
                )
                break

            try:
                commit_message_offset(consumer, message)
            except KafkaError as exc:
                print(
                    f"Échec du commit pour partition={message.partition}, "
                    f"offset={message.offset}: {exc}. "
                    "Le message pourra être rejoué."
                )
                break

            status = "nouvel événement" if is_new else "doublon PostgreSQL"
            print(
                f"Interaction traitée ({status}) : "
                f"user={user_id}, movie={movie_id}, rating={rating}; "
                f"cache supprimé={deleted}; "
                f"partition={message.partition}, offset={message.offset}"
            )

    except KeyboardInterrupt:
        print("Arrêt demandé par l'utilisateur.")

    finally:
        if consumer is not None:
            consumer.close()
        engine.dispose()
        print("Consumer arrêté.")


if __name__ == "__main__":
    main()

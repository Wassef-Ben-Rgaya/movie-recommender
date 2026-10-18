
import json
import os
from datetime import datetime, timezone

from kafka import KafkaProducer

BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"
)


def send_interaction(user_id: int, movie_id: int, rating: float) -> dict:
    if user_id < 1 or movie_id < 1:
        raise ValueError("user_id et movie_id doivent être positifs")

    if not 0.5 <= rating <= 5.0:
        raise ValueError("La note doit être comprise entre 0.5 et 5.0")

    event = {
        "user_id": user_id,
        "movie_id": movie_id,
        "rating": rating,
        "event_type": "rating",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        key_serializer=lambda key: str(key).encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )

    try:
        result = producer.send(
            "user_interactions",
            key=user_id,
            value=event,
        ).get(timeout=15)

        producer.flush()

        print(
            f"Interaction envoyée : user={user_id}, "
            f"movie={movie_id}, rating={rating}, "
            f"partition={result.partition}, offset={result.offset}"
        )
        return event
    finally:
        producer.close()


if __name__ == "__main__":
    send_interaction(user_id=1, movie_id=1, rating=5.0)

.PHONY: build up down ingest preprocess pipeline run_all logs

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

ingest:
	docker-compose exec spark python /app/src/ingest.py

preprocess:
	docker-compose exec spark python /app/src/preprocessing.py

pipeline:
	docker-compose exec spark python /app/src/pipeline.py

run_all:
	docker-compose exec spark bash -c "python /app/src/ingest.py && python /app/src/preprocessing.py && python /app/src/pipeline.py"

logs:
	docker-compose logs spark

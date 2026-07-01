.PHONY: help install db-up db-down db-reset run clean

UV = uv
DOCKER_COMPOSE = docker compose

help:
	@echo "Available commands:"
	@echo "  install : Install dependencies using uv"
	@echo "  db-up   : Start PostgreSQL + Qdrant"
	@echo "  db-down : Stop all services"
	@echo "  db-reset: Stop all services and delete all data"
	@echo "  run     : Run FastAPI backend"
	@echo "  clean   : Clean up temporary files"

install:
	$(UV) sync

db-up:
	$(DOCKER_COMPOSE) up -d

db-down:
	$(DOCKER_COMPOSE) down

db-reset:
	@read -p "Xoá toàn bộ data? [y/N] " c && [ "$$c" = "y" ] && $(DOCKER_COMPOSE) down -v

run:
	$(UV) run main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

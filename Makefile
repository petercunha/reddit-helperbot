.PHONY: help dev watch deploy deploy-fresh logs ps stop start down

help:
	@echo "Available targets:"
	@echo "  make dev          # local dev: build and run in foreground"
	@echo "  make watch        # local dev: auto-rebuild on file changes"
	@echo "  make deploy       # server deploy: build and run detached"
	@echo "  make deploy-fresh # force recreate container on deploy"
	@echo "  make logs         # follow helperbot logs"
	@echo "  make ps           # show compose status"
	@echo "  make stop         # stop services"
	@echo "  make start        # start stopped services"
	@echo "  make down         # stop and remove services"

dev:
	docker compose up --build

watch:
	docker compose up --watch

deploy:
	docker compose up -d --build

deploy-fresh:
	docker compose up -d --build --force-recreate

logs:
	docker compose logs -f helperbot

ps:
	docker compose ps

stop:
	docker compose stop

start:
	docker compose start

down:
	docker compose down

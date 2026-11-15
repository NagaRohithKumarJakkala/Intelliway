# Intelliway Chat App

Small standalone web chat for Intelliway. It has no frontend dependency or
Python package dependency and runs separately from the main Intelliway
Compose project.

## Run with Docker

Start Intelliway first so LiteLLM and Semantic Router are available:

```bash
cd ..
docker compose up -d
```

Then start this app from this directory:

```bash
cd chat-app
export LITELLM_MASTER_KEY="$(grep '^LITELLM_MASTER_KEY=' ../.env | cut -d= -f2-)"
docker compose up -d --build
```

Open <http://localhost:3000>.

The app offers automatic routing, local aliases, and external GLM aliases.
It sends LiteLLM requests with the key server-side, so the key is not exposed
in browser JavaScript. Each response displays the model returned by LiteLLM or
Semantic Router, plus prompt, completion, and total token counts when the
upstream provider returns usage data.

## Stop

```bash
docker compose down
```

## Configuration

The defaults assume the main Intelliway services run on the host:

```dotenv
CHAT_LITELLM_BASE_URL=http://host.docker.internal:4000
CHAT_SEMANTIC_ROUTER_BASE_URL=http://host.docker.internal:8899
CHAT_MAX_TOKENS=4096
```

Override these values if the services use different addresses. The main
Intelliway Docker files are not modified by this app.

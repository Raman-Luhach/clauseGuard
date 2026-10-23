# Deploying ClauseGuard

The serving image contains the web UI, the REST API and the packaged models (TF-IDF and/or
ONNX). It does not contain PyTorch, so it is small and runs on a free CPU host.

## Build and run locally

```bash
make package                      # served run of every module -> deploy/models
docker build -t clauseguard .
docker run --rm -p 7860:7860 clauseguard
# open http://localhost:7860 ; API docs at http://localhost:7860/docs
```

## Hugging Face Spaces (free CPU)

1. Create a Space at https://huggingface.co/new-space with **SDK: Docker** (blank template).
2. Clone it and copy in: `Dockerfile`, `.dockerignore`, `pyproject.toml`, `uv.lock`, `src/`,
   `configs/`, `deploy/models/`, and a `README.md` that starts with:

   ```yaml
   ---
   title: ClauseGuard
   emoji: ⚖️
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   ---
   ```

3. Model files are large: run `git lfs install` and `git lfs track "*.onnx" "*.joblib"`
   before committing, then push. The Space builds the image and starts the app.

## Optional: LLM explanations

Explanations come from templates by default. To use Claude instead, set in the Space settings
(or with `docker run -e ...`): the secret `ANTHROPIC_API_KEY`, and the variable
`CLAUSEGUARD_LLM=anthropic`. Optional: `CLAUSEGUARD_LLM_MODEL` (default `claude-opus-5-5`),
`CLAUSEGUARD_LLM_EFFORT` (default `low`). Only the flagged clauses are sent, never the whole
document; if the API fails, the templates are used.

## Notes

- **Licence:** the privacy model is trained on OPP-115, which allows research, teaching and
  scholarship only. Keep the deployment non-commercial and say so on the Space.
- **Logs:** request logs (`/app/logs/events.jsonl`, metadata only) are lost when a Space
  restarts. `clauseguard monitor` summarises them where they persist.
- **Health:** `GET /health` returns 503 until at least one model is available; Docker's
  `HEALTHCHECK` uses it.

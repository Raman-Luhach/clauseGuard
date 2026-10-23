# ClauseGuard common tasks. Everything here runs on a laptop CPU; GPU training comes later.
MODULES := tos privacy contract

.PHONY: setup lint format test test-fast data train-cpu evaluate cv smoke example check \
	train-gpu train-gpu-smoke evaluate-final export-onnx robustness benchmark package after-gpu serve

setup:  ## install Python 3.12 and all dependencies
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:  ## full test suite (includes a tiny real transformer and the real ToS data)
	uv run pytest -q

test-fast:  ## skip tests that download a model or read the real datasets
	uv run pytest -q -m "not slow"

data:  ## download all datasets (checksum-verified) and build unit tables + data cards
	uv run clauseguard data download all
	for m in $(MODULES); do uv run clauseguard data prepare $$m && uv run clauseguard data card $$m || exit 1; done

train-cpu:  ## the real majority and TF-IDF baselines (seconds)
	for m in $(MODULES); do uv run clauseguard train $$m --model majority && uv run clauseguard train $$m --model tfidf || exit 1; done

evaluate:  ## validation report for the served run of every module
	for m in $(MODULES); do uv run clauseguard evaluate $$m || exit 1; done

cv:  ## 5-fold cross-validation of the privacy TF-IDF baseline (small test set)
	uv run clauseguard cv privacy

smoke:  ## tiny CPU run of every configured model, including the transformers
	uv run clauseguard smoke $(MODULES)

example:  ## analyse the sample documents (one per document type)
	uv run clauseguard analyze examples/github-terms-of-service.txt --type tos
	uv run clauseguard analyze examples/github-privacy-statement.txt --type privacy
	uv run clauseguard analyze examples/cuad-test-contract.txt --type contract

check: lint test smoke train-gpu-smoke  ## everything that must pass before the GPU session

# ---- GPU session (Lightning AI); see docs/LIGHTNING.md
train-gpu:  ## every transformer for every module, best selected; resumable
	uv run clauseguard train-gpu $(MODULES)

train-gpu-smoke:  ## the whole GPU session in miniature, on the CPU
	uv run clauseguard train-gpu $(MODULES) --smoke

evaluate-final:  ## results table with the test split (run once, after the GPU session)
	uv run clauseguard final-report $(MODULES) --final

# ---- after the GPU session, back on the Mac
export-onnx:  ## ONNX + INT8 export of every module's served transformer
	for m in $(MODULES); do uv run clauseguard export-onnx $$m || exit 1; done

robustness:
	for m in $(MODULES); do uv run clauseguard robustness $$m || exit 1; done

benchmark:
	for m in $(MODULES); do uv run clauseguard benchmark $$m || exit 1; done

package:  ## copy the served models to deploy/models for the Docker image
	rm -rf deploy/models && uv run clauseguard package-models deploy/models

after-gpu: evaluate-final export-onnx robustness benchmark package  ## all post-training steps

serve:  ## web UI + API on http://127.0.0.1:8000
	uv run clauseguard serve

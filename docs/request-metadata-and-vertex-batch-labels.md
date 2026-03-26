# Request metadata, Vertex labels, and batch `RequestLabels`

This document summarizes the **request `Metadata`** field on unified completions, how each provider uses it, **Vertex-only request labels**, and **echoed labels on Vertex batch results** (`RequestLabels`). It also covers **tests**, **Makefile targets**, and **editor/CI notes** added alongside these features.

---

## 1. `CompletionRequest.Metadata`

**Location:** [`pkg/types/request.go`](../pkg/types/request.go)

- **Type:** `map[string]string` with JSON name `metadata`.
- **Purpose:** Optional string key–value data for providers that support tracing, billing labels, or dashboard metadata.
- **Not merged with** `Extra` (`map[string]any`); those remain separate escape hatches.

### Provider mapping

| Provider | Behavior |
|----------|----------|
| **OpenAI** | Copied to the chat completions body as `metadata` ([`pkg/provider/openai`](../pkg/provider/openai/)). |
| **Vertex AI** | Merged into Gemini `generateContent` **`labels`** (see §2). Applies to **Complete**, **Stream**, and **batch** input lines. |
| **Google (Generative Language / AI Studio)** | **Not sent.** That API rejects unknown fields such as `labels`; `Metadata` is ignored so requests stay valid. |
| **Anthropic** | Only the key **`user_id`** is forwarded to `metadata.user_id`. Other keys are ignored. |

Vertex batch continues to set **`labels["custom_id"]`** from the batch `CustomID`; that is merged with user `Metadata` (see [`pkg/provider/vertex/batch.go`](../pkg/provider/vertex/batch.go)).

---

## 2. Vertex vs Google: `ApplyMetadataAsLabels`

**Location:** [`pkg/provider/google/transform.go`](../pkg/provider/google/transform.go)

- **`TransformRequest`** builds a `GenerateContentRequest` **without** setting `Labels` (shared by both Google and Vertex clients).
- **`ApplyMetadataAsLabels(gReq, metadata)`** copies `metadata` into `gReq.Labels`.

**Call sites (Vertex only):**

- [`pkg/provider/vertex/client.go`](../pkg/provider/vertex/client.go) — after `TransformRequest` in **Complete** and **Stream**.
- [`pkg/provider/vertex/batch.go`](../pkg/provider/vertex/batch.go) — after `TransformRequest` for each batch line, before `custom_id` is added.

The **Google** client ([`pkg/provider/google/client.go`](../pkg/provider/google/client.go)) does **not** call `ApplyMetadataAsLabels`.

**Label constraints:** GCP/Vertex label keys and values must follow [Google’s label rules](https://cloud.google.com/resource-manager/docs/creating-managing-labels) (e.g. lowercase-oriented keys). Invalid labels surface as API errors from Vertex.

---

## 3. Batch results: `RequestLabels`

When Vertex batch output JSONL **echoes** the original request, each line can include `request.labels`. The router now exposes the full map on results.

| Type | Field | JSON (when serialized) |
|------|--------|-------------------------|
| [`provider.BatchResult`](../pkg/provider/provider.go) | `RequestLabels` | `request_labels` |
| [`batch.Result`](../pkg/batch/batch.go) | `RequestLabels` | `request_labels` |

**Population:** [`pkg/provider/vertex/batch.go`](../pkg/provider/vertex/batch.go) (`downloadBatchResults`) copies `line.Request.Labels` into `BatchResult.RequestLabels` and still sets `CustomID` from `labels["custom_id"]`.

Other batch providers may leave `RequestLabels` nil.

---

## 4. Tests

### Unit tests (`go test ./pkg/...`)

| Area | File(s) |
|------|---------|
| `ApplyMetadataAsLabels` / no labels in `TransformRequest` | [`pkg/provider/google/transform_test.go`](../pkg/provider/google/transform_test.go) |
| OpenAI `metadata` in JSON | [`pkg/provider/openai/transform_test.go`](../pkg/provider/openai/transform_test.go) |
| Anthropic `metadata.user_id` | [`pkg/provider/anthropic/transform_test.go`](../pkg/provider/anthropic/transform_test.go) |
| Vertex batch label merge (metadata + `custom_id`) | [`pkg/provider/vertex/batch_metadata_test.go`](../pkg/provider/vertex/batch_metadata_test.go) |
| `GetBatchResults` / `RequestLabels` (mocked GCS) | [`pkg/provider/vertex/client_test.go`](../pkg/provider/vertex/client_test.go) (`TestDownloadBatchResults_*`) |

### Integration tests (`-tags=integration`)

**Package:** [`tests/`](../tests/)

| File | Notes |
|------|--------|
| [`tests/doc.go`](../tests/doc.go) | Always-built `package tests` so `go list ./tests` works without the `integration` tag. |
| [`tests/metadata_integration_test.go`](../tests/metadata_integration_test.go) | OpenAI/Anthropic/Vertex/Google metadata behavior; Vertex batch **create** with metadata; optional slow **wait + GetResults** test. |
| [`tests/integration_test.go`](../tests/integration_test.go) | General integration suite; package comment documents `VERTEX_BATCH_WAIT_RESULTS`. |

**Slow Vertex batch end-to-end (echoed labels):**

- Test: `TestVertex_BatchMetadata_GetResultsIncludesEchoedLabels`
- **Requires:** `VERTEX_BATCH_WAIT_RESULTS=1`, Vertex auth, `VERTEX_BATCH_BUCKET`, etc.
- **Convenience:** `make test-vertex-batch-results` sets the env var and runs that test.

**Google integration:** `TestGoogle_CompletionWithMetadata_NotForwarded` ensures setting `Metadata` does not break the AI Studio API (labels are not sent).

---

## 5. Makefile and environment

| Target | Purpose |
|--------|---------|
| `make test-metadata` | Integration tests matching `Metadata` or `BatchMetadata_GetResults` (slow batch test skips unless `VERTEX_BATCH_WAIT_RESULTS=1`). |
| `make test-vertex-batch-results` | Runs `TestVertex_BatchMetadata_GetResultsIncludesEchoedLabels` with `VERTEX_BATCH_WAIT_RESULTS=1`. |

**Optional env (documented in `make help`):**

- `VERTEX_BATCH_WAIT_RESULTS=1` — enable the long-running Vertex batch **GetResults** / **RequestLabels** integration test.

---

## 6. Editor: gopls and the `integration` build tag

Integration sources use `//go:build integration`, so by default **no Go files** matched and tools could report “no packages” for `tests/`.

**Repo fixes:**

- [`tests/doc.go`](../tests/doc.go) — untagged file in `package tests`.
- [`.vscode/settings.json`](../.vscode/settings.json) — `gopls` `build.buildFlags`: `["-tags=integration"]` so tagged test files type-check in the IDE.

Reload the editor after cloning if diagnostics are missing.

---

## 7. File index (primary touchpoints)

- Types: [`pkg/types/request.go`](../pkg/types/request.go)
- Provider interface / batch types: [`pkg/provider/provider.go`](../pkg/provider/provider.go)
- Google transform + `ApplyMetadataAsLabels`: [`pkg/provider/google/transform.go`](../pkg/provider/google/transform.go)
- OpenAI: [`pkg/provider/openai/types.go`](../pkg/provider/openai/types.go), [`transform.go`](../pkg/provider/openai/transform.go)
- Anthropic: [`pkg/provider/anthropic/transform.go`](../pkg/provider/anthropic/transform.go)
- Vertex client + batch: [`pkg/provider/vertex/client.go`](../pkg/provider/vertex/client.go), [`batch.go`](../pkg/provider/vertex/batch.go)
- Batch facade: [`pkg/batch/batch.go`](../pkg/batch/batch.go)

---

## 8. OpenAI metadata limits

OpenAI documents limits on chat **metadata** (e.g. number of pairs, key/value lengths). Values outside those limits may fail at the API. The router forwards the map as-is; validation is left to the provider unless added later.

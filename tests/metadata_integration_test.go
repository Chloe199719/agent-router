//go:build integration

package tests

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Chloe199719/agent-router/pkg/batch"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// metadataLabelKeys must satisfy Vertex / GCP label rules (lowercase, digits, underscores, hyphens).
const metadataLabelKeysIntegration = "agent_router_itest"

// ============================================================================
// Request Metadata → OpenAI metadata
// ============================================================================

func TestOpenAI_CompletionWithMetadata(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderOpenAI) {
		t.Skip("OpenAI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderOpenAI,
		Model:     openAIModel,
		MaxTokens: types.Ptr(40),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Reply with exactly: ok"),
		},
		Metadata: map[string]string{
			"agent_router_itest": "completion",
			"run_id":             "metadata-integration",
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}
	if strings.TrimSpace(resp.Text()) == "" {
		t.Error("expected non-empty text")
	}
	t.Logf("text: %q", resp.Text())
}

func TestOpenAI_StreamWithMetadata(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderOpenAI) {
		t.Skip("OpenAI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	stream, err := r.Stream(ctx, &types.CompletionRequest{
		Provider:  types.ProviderOpenAI,
		Model:     openAIModel,
		MaxTokens: types.Ptr(40),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Say hi in one short phrase."),
		},
		Metadata: map[string]string{
			"agent_router_itest": "stream",
		},
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}
	defer stream.Close()

	for {
		ev, err := stream.Next()
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if ev == nil {
			break
		}
	}
	if stream.Response() == nil {
		t.Error("expected accumulated response after stream")
	}
}

// ============================================================================
// Request Metadata → Anthropic metadata.user_id
// ============================================================================

func TestAnthropic_CompletionWithMetadataUserID(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderAnthropic) {
		t.Skip("Anthropic not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderAnthropic,
		Model:     anthropicModel,
		MaxTokens: types.Ptr(40),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Reply with exactly: ok"),
		},
		Metadata: map[string]string{
			"user_id":           "integration-test-user-metadata",
			"ignored_other_key": "should-not-break-request",
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}
	if strings.TrimSpace(resp.Text()) == "" {
		t.Error("expected non-empty text")
	}
	t.Logf("text: %q", resp.Text())
}

// ============================================================================
// Request Metadata → Google / Vertex labels
// ============================================================================

// TestGoogle_CompletionWithMetadata_NotForwarded checks that Metadata does not break the Google
// Generative Language API. That API does not support Vertex-style labels; the router omits them.
func TestGoogle_CompletionWithMetadata_NotForwarded(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderGoogle) {
		t.Skip("Google not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderGoogle,
		Model:     googleModel,
		MaxTokens: types.Ptr(40),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Reply with exactly: ok"),
		},
		Metadata: map[string]string{
			metadataLabelKeysIntegration: "would-be-vertex-only",
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}
	if strings.TrimSpace(resp.Text()) == "" {
		t.Error("expected non-empty text")
	}
	t.Logf("text: %q", resp.Text())
}

func TestVertex_CompletionWithMetadataLabels(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderVertex) {
		t.Skip("Vertex AI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderVertex,
		Model:     vertexModel,
		MaxTokens: types.Ptr(40),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Reply with exactly: ok"),
		},
		Metadata: map[string]string{
			metadataLabelKeysIntegration: "vertex-labels",
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}
	if strings.TrimSpace(resp.Text()) == "" {
		t.Error("expected non-empty text")
	}
	t.Logf("text: %q", resp.Text())
}

func TestVertex_StreamWithMetadataLabels(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderVertex) {
		t.Skip("Vertex AI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	stream, err := r.Stream(ctx, &types.CompletionRequest{
		Provider:  types.ProviderVertex,
		Model:     vertexModel,
		MaxTokens: types.Ptr(40),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Say hi in one short phrase."),
		},
		Metadata: map[string]string{
			metadataLabelKeysIntegration: "vertex-stream-labels",
		},
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}
	defer stream.Close()

	for {
		ev, err := stream.Next()
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if ev == nil {
			break
		}
	}
}

// ============================================================================
// Vertex batch: labels include both CustomID and request Metadata
// ============================================================================

func TestVertex_BatchWithRequestMetadata(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderVertex) {
		t.Skip("Vertex AI not configured")
	}
	if os.Getenv("VERTEX_BATCH_BUCKET") == "" {
		t.Skip("VERTEX_BATCH_BUCKET not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Versioned id required for Vertex batch (see TestVertex_BatchCreate).
	batchModel := "gemini-2.0-flash-001"

	job, err := r.Batch().Create(ctx, types.ProviderVertex, vertexBatchRequestsWithMetadata(batchModel))
	if err != nil {
		t.Fatalf("Batch create failed: %v", err)
	}
	if job.ID == "" {
		t.Fatal("empty batch job ID")
	}
	t.Logf("batch job %s created (metadata on inline requests is merged into labels by provider)", job.ID)
}

// TestVertex_BatchMetadata_GetResultsIncludesEchoedLabels waits for a Vertex batch job to finish, then checks
// GetResults returns RequestLabels echoing custom_id and CompletionRequest.Metadata (Vertex labels).
//
// Slow (often many minutes). Opt in: VERTEX_BATCH_WAIT_RESULTS=1
func TestVertex_BatchMetadata_GetResultsIncludesEchoedLabels(t *testing.T) {
	if os.Getenv("VERTEX_BATCH_WAIT_RESULTS") == "" {
		t.Skip("set VERTEX_BATCH_WAIT_RESULTS=1 to run this test (waits for Vertex batch completion; can take many minutes)")
	}

	r := getRouter(t)
	if !hasProvider(r, types.ProviderVertex) {
		t.Skip("Vertex AI not configured")
	}
	if os.Getenv("VERTEX_BATCH_BUCKET") == "" {
		t.Skip("VERTEX_BATCH_BUCKET not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	batchModel := "gemini-2.0-flash-001"
	job, err := r.Batch().Create(ctx, types.ProviderVertex, vertexBatchRequestsWithMetadata(batchModel))
	if err != nil {
		t.Fatalf("Batch create failed: %v", err)
	}
	if job.ID == "" {
		t.Fatal("empty batch job ID")
	}
	t.Logf("waiting for batch job %s", job.ID)

	finalJob, err := r.Batch().Wait(ctx, types.ProviderVertex, job.ID, 20*time.Second)
	if err != nil {
		t.Fatalf("Batch wait: %v", err)
	}
	if finalJob.Status != batch.StatusCompleted {
		t.Fatalf("batch finished with status %s (want %s)", finalJob.Status, batch.StatusCompleted)
	}

	results, err := r.Batch().GetResults(ctx, types.ProviderVertex, job.ID)
	if err != nil {
		t.Fatalf("GetResults: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	byCustom := make(map[string]batch.Result, len(results))
	for _, res := range results {
		byCustom[res.CustomID] = res
	}

	for _, id := range []string{"meta-itest-1", "meta-itest-2"} {
		res, ok := byCustom[id]
		if !ok {
			t.Fatalf("missing result for CustomID %q (have: %v)", id, byCustomKeys(byCustom))
		}
		if res.Error != nil {
			t.Fatalf("result %q: %v", id, res.Error)
		}
		labels := res.RequestLabels
		if labels == nil {
			t.Fatalf("result %q: nil RequestLabels (batch output should echo request labels)", id)
		}
		if labels["custom_id"] != id {
			t.Errorf("result %q: labels[custom_id]=%q", id, labels["custom_id"])
		}
		if labels[metadataLabelKeysIntegration] != "vertex-batch" {
			t.Errorf("result %q: labels[%s]=%q want vertex-batch", id, metadataLabelKeysIntegration, labels[metadataLabelKeysIntegration])
		}
		wantExtra := map[string]string{"meta-itest-1": "one", "meta-itest-2": "two"}[id]
		if labels["batch_extra_label"] != wantExtra {
			t.Errorf("result %q: batch_extra_label=%q want %q", id, labels["batch_extra_label"], wantExtra)
		}
		if len(labels) < 3 {
			t.Errorf("result %q: expected at least 3 labels (custom_id + 2 metadata), got %d: %v", id, len(labels), labels)
		}
		text := ""
		if res.Response != nil {
			text = res.Response.Text()
		}
		t.Logf("result %q: RequestLabels=%v text=%q", id, labels, text)
	}
}

func byCustomKeys(m map[string]batch.Result) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// vertexBatchRequestsWithMetadata mirrors TestVertex_BatchCreate with Metadata on each request (merged into labels with custom_id).
func vertexBatchRequestsWithMetadata(model string) []batch.Request {
	return []batch.Request{
		{
			CustomID: "meta-itest-1",
			Request: &types.CompletionRequest{
				Provider:  types.ProviderVertex,
				Model:     model,
				MaxTokens: types.Ptr(30),
				Messages: []types.Message{
					types.NewTextMessage(types.RoleUser, "Say 'a' only."),
				},
				Metadata: map[string]string{
					metadataLabelKeysIntegration: "vertex-batch",
					"batch_extra_label":          "one",
				},
			},
		},
		{
			CustomID: "meta-itest-2",
			Request: &types.CompletionRequest{
				Provider:  types.ProviderVertex,
				Model:     model,
				MaxTokens: types.Ptr(30),
				Messages: []types.Message{
					types.NewTextMessage(types.RoleUser, "Say 'b' only."),
				},
				Metadata: map[string]string{
					metadataLabelKeysIntegration: "vertex-batch",
					"batch_extra_label":          "two",
				},
			},
		},
	}
}

package vertex

import (
	"testing"

	googleProvider "github.com/Chloe199719/agent-router/pkg/provider/google"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// CreateBatch merges custom_id into labels after TransformRequest; user Metadata must remain.
func TestBatchInputLabelsMergeMetadataAndCustomID(t *testing.T) {
	transformer := googleProvider.NewTransformer()

	req := &types.CompletionRequest{
		Model:    "gemini-2.0-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hello")},
		Metadata: map[string]string{
			"trace_id": "trace-xyz",
		},
	}

	gReq := transformer.TransformRequest(req)
	googleProvider.ApplyMetadataAsLabels(gReq, req.Metadata)

	if gReq.Labels == nil {
		gReq.Labels = make(map[string]string)
	}
	gReq.Labels["custom_id"] = "request-alpha"

	if gReq.Labels["trace_id"] != "trace-xyz" {
		t.Errorf("expected trace_id preserved, got %q", gReq.Labels["trace_id"])
	}
	if gReq.Labels["custom_id"] != "request-alpha" {
		t.Errorf("expected custom_id from batch, got %q", gReq.Labels["custom_id"])
	}
	if len(gReq.Labels) != 2 {
		t.Errorf("expected 2 labels, got %d: %v", len(gReq.Labels), gReq.Labels)
	}
}

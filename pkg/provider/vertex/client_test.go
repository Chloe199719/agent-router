package vertex

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	googleProvider "github.com/Chloe199719/agent-router/pkg/provider/google"

	"github.com/Chloe199719/agent-router/pkg/provider"
	"github.com/Chloe199719/agent-router/pkg/types"
)

func TestNew_DefaultConfig(t *testing.T) {
	client := New("my-project", "us-central1", provider.WithAccessToken("test-token"))

	if client.projectID != "my-project" {
		t.Errorf("expected projectID 'my-project', got %q", client.projectID)
	}
	if client.location != "us-central1" {
		t.Errorf("expected location 'us-central1', got %q", client.location)
	}
	if client.baseURL != "https://us-central1-aiplatform.googleapis.com/v1" {
		t.Errorf("expected default base URL, got %q", client.baseURL)
	}
	if client.config.AccessToken != "test-token" {
		t.Errorf("expected access token 'test-token', got %q", client.config.AccessToken)
	}
}

func TestNew_CustomBaseURL(t *testing.T) {
	client := New("my-project", "us-central1",
		provider.WithAccessToken("test-token"),
		provider.WithBaseURL("https://custom-endpoint.example.com/v1"),
	)

	if client.baseURL != "https://custom-endpoint.example.com/v1" {
		t.Errorf("expected custom base URL, got %q", client.baseURL)
	}
}

func TestNew_APIKeyAuth(t *testing.T) {
	client := New("my-project", "us-central1", provider.WithAPIKey("test-api-key"))

	if client.config.APIKey != "test-api-key" {
		t.Errorf("expected API key 'test-api-key', got %q", client.config.APIKey)
	}
}

func TestNew_ProjectIDFromConfig(t *testing.T) {
	client := New("", "", provider.WithAccessToken("tok"), provider.WithProjectID("cfg-project"), provider.WithLocation("europe-west1"))

	if client.projectID != "cfg-project" {
		t.Errorf("expected projectID 'cfg-project', got %q", client.projectID)
	}
	if client.location != "europe-west1" {
		t.Errorf("expected location 'europe-west1', got %q", client.location)
	}
}

func TestName(t *testing.T) {
	client := New("proj", "loc", provider.WithAccessToken("tok"))
	if client.Name() != types.ProviderVertex {
		t.Errorf("expected provider name 'vertex', got %q", client.Name())
	}
}

func TestSupportsFeature(t *testing.T) {
	client := New("proj", "loc", provider.WithAccessToken("tok"))

	supported := []types.Feature{
		types.FeatureStreaming,
		types.FeatureStructuredOutput,
		types.FeatureTools,
		types.FeatureVision,
		types.FeatureJSON,
		types.FeatureBatch,
	}

	for _, f := range supported {
		if !client.SupportsFeature(f) {
			t.Errorf("expected feature %q to be supported", f)
		}
	}

	if client.SupportsFeature("unknown_feature") {
		t.Error("expected unknown feature to be unsupported")
	}
}

func TestModels(t *testing.T) {
	client := New("proj", "loc", provider.WithAccessToken("tok"))
	models := client.Models()

	if len(models) == 0 {
		t.Fatal("expected at least one model")
	}

	// Check that gemini-2.0-flash is included
	found := false
	for _, m := range models {
		if m == "gemini-2.0-flash" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected 'gemini-2.0-flash' in models list")
	}
}

func TestBuildURL_WithAccessToken(t *testing.T) {
	client := New("my-project", "us-central1", provider.WithAccessToken("test-token"))

	url := client.buildURL("gemini-2.0-flash", "generateContent")
	expected := "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent"

	if url != expected {
		t.Errorf("expected URL:\n  %s\ngot:\n  %s", expected, url)
	}
}

func TestBuildURL_WithAPIKey(t *testing.T) {
	client := New("my-project", "us-central1", provider.WithAPIKey("my-api-key"))

	url := client.buildURL("gemini-2.0-flash", "streamGenerateContent")
	expected := "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:streamGenerateContent?key=my-api-key"

	if url != expected {
		t.Errorf("expected URL:\n  %s\ngot:\n  %s", expected, url)
	}
}

func TestBuildURL_DifferentLocations(t *testing.T) {
	tests := []struct {
		location string
		expected string
	}{
		{"us-central1", "https://us-central1-aiplatform.googleapis.com/v1"},
		{"europe-west1", "https://europe-west1-aiplatform.googleapis.com/v1"},
		{"asia-southeast1", "https://asia-southeast1-aiplatform.googleapis.com/v1"},
	}

	for _, tt := range tests {
		client := New("proj", tt.location, provider.WithAccessToken("tok"))
		if client.baseURL != tt.expected {
			t.Errorf("location %q: expected base URL %q, got %q", tt.location, tt.expected, client.baseURL)
		}
	}
}

func TestSetHeaders_AccessToken(t *testing.T) {
	client := New("proj", "loc", provider.WithAccessToken("my-bearer-token"))

	req, _ := http.NewRequest("POST", "https://example.com", nil)
	client.setHeaders(req)

	if req.Header.Get("Content-Type") != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %q", req.Header.Get("Content-Type"))
	}
	if req.Header.Get("Authorization") != "Bearer my-bearer-token" {
		t.Errorf("expected Authorization 'Bearer my-bearer-token', got %q", req.Header.Get("Authorization"))
	}
}

func TestSetHeaders_APIKey(t *testing.T) {
	client := New("proj", "loc", provider.WithAPIKey("my-api-key"))

	req, _ := http.NewRequest("POST", "https://example.com", nil)
	client.setHeaders(req)

	if req.Header.Get("Content-Type") != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %q", req.Header.Get("Content-Type"))
	}
	// API key auth should NOT set Authorization header
	if req.Header.Get("Authorization") != "" {
		t.Errorf("expected no Authorization header for API key auth, got %q", req.Header.Get("Authorization"))
	}
}

func TestComplete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the URL pattern
		if !strings.Contains(r.URL.Path, "/projects/test-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent") {
			t.Errorf("unexpected URL path: %s", r.URL.Path)
		}

		// Verify Authorization header
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("expected Bearer token, got %q", r.Header.Get("Authorization"))
		}

		// Verify request body is valid
		body, _ := io.ReadAll(r.Body)
		var gReq googleProvider.GenerateContentRequest
		if err := json.Unmarshal(body, &gReq); err != nil {
			t.Errorf("failed to unmarshal request: %v", err)
		}

		// Return response
		resp := googleProvider.GenerateContentResponse{
			Candidates: []googleProvider.Candidate{
				{
					Content: &googleProvider.Content{
						Role: "model",
						Parts: []googleProvider.Part{
							{Text: "Hello from Vertex AI!"},
						},
					},
					FinishReason: "STOP",
				},
			},
			UsageMetadata: &googleProvider.UsageMetadata{
				PromptTokenCount:     5,
				CandidatesTokenCount: 10,
				TotalTokenCount:      15,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := New("test-project", "us-central1",
		provider.WithAccessToken("test-token"),
		provider.WithBaseURL(server.URL),
	)

	resp, err := client.Complete(context.Background(), &types.CompletionRequest{
		Provider: types.ProviderVertex,
		Model:    "gemini-2.0-flash",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Hello"),
		},
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Provider != types.ProviderVertex {
		t.Errorf("expected provider 'vertex', got %q", resp.Provider)
	}

	if resp.Model != "gemini-2.0-flash" {
		t.Errorf("expected model 'gemini-2.0-flash', got %q", resp.Model)
	}

	if resp.Text() != "Hello from Vertex AI!" {
		t.Errorf("expected text 'Hello from Vertex AI!', got %q", resp.Text())
	}

	if resp.StopReason != types.StopReasonEnd {
		t.Errorf("expected stop reason 'end', got %q", resp.StopReason)
	}

	if resp.Usage.InputTokens != 5 {
		t.Errorf("expected 5 input tokens, got %d", resp.Usage.InputTokens)
	}

	if resp.Usage.OutputTokens != 10 {
		t.Errorf("expected 10 output tokens, got %d", resp.Usage.OutputTokens)
	}

	if resp.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", resp.Usage.TotalTokens)
	}
}

func TestComplete_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(googleProvider.ErrorResponse{
			Error: &googleProvider.APIError{
				Code:    401,
				Message: "invalid credentials",
				Status:  "UNAUTHENTICATED",
			},
		})
	}))
	defer server.Close()

	client := New("test-project", "us-central1",
		provider.WithAccessToken("bad-token"),
		provider.WithBaseURL(server.URL),
	)

	_, err := client.Complete(context.Background(), &types.CompletionRequest{
		Provider: types.ProviderVertex,
		Model:    "gemini-2.0-flash",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Hello"),
		},
	})

	if err == nil {
		t.Fatal("expected error for unauthorized request")
	}

	if !strings.Contains(err.Error(), "invalid") {
		t.Errorf("expected error to mention 'invalid', got: %v", err)
	}
}

func TestComplete_RateLimitError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(googleProvider.ErrorResponse{
			Error: &googleProvider.APIError{
				Code:    429,
				Message: "rate limit exceeded",
				Status:  "RESOURCE_EXHAUSTED",
			},
		})
	}))
	defer server.Close()

	client := New("test-project", "us-central1",
		provider.WithAccessToken("tok"),
		provider.WithBaseURL(server.URL),
	)

	_, err := client.Complete(context.Background(), &types.CompletionRequest{
		Model:    "gemini-2.0-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hello")},
	})

	if err == nil {
		t.Fatal("expected rate limit error")
	}
	if !strings.Contains(err.Error(), "rate limit") {
		t.Errorf("expected error to mention 'rate limit', got: %v", err)
	}
}

func TestComplete_NotFoundError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(googleProvider.ErrorResponse{
			Error: &googleProvider.APIError{
				Code:    404,
				Message: "model not found",
				Status:  "NOT_FOUND",
			},
		})
	}))
	defer server.Close()

	client := New("test-project", "us-central1",
		provider.WithAccessToken("tok"),
		provider.WithBaseURL(server.URL),
	)

	_, err := client.Complete(context.Background(), &types.CompletionRequest{
		Model:    "nonexistent-model",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hello")},
	})

	if err == nil {
		t.Fatal("expected not found error")
	}
	if !strings.Contains(err.Error(), "model not found") {
		t.Errorf("expected error to mention 'model not found', got: %v", err)
	}
}

func TestComplete_ForbiddenError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(googleProvider.ErrorResponse{
			Error: &googleProvider.APIError{
				Code:    403,
				Message: "permission denied",
				Status:  "PERMISSION_DENIED",
			},
		})
	}))
	defer server.Close()

	client := New("test-project", "us-central1",
		provider.WithAccessToken("tok"),
		provider.WithBaseURL(server.URL),
	)

	_, err := client.Complete(context.Background(), &types.CompletionRequest{
		Model:    "gemini-2.0-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hello")},
	})

	if err == nil {
		t.Fatal("expected forbidden error")
	}
	if !strings.Contains(err.Error(), "permission denied") {
		t.Errorf("expected error to mention 'permission denied', got: %v", err)
	}
}

func TestComplete_ContextLengthError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(googleProvider.ErrorResponse{
			Error: &googleProvider.APIError{
				Code:    400,
				Message: "context length exceeded: too many tokens",
				Status:  "INVALID_ARGUMENT",
			},
		})
	}))
	defer server.Close()

	client := New("test-project", "us-central1",
		provider.WithAccessToken("tok"),
		provider.WithBaseURL(server.URL),
	)

	_, err := client.Complete(context.Background(), &types.CompletionRequest{
		Model:    "gemini-2.0-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hello")},
	})

	if err == nil {
		t.Fatal("expected context length error")
	}
	if !strings.Contains(err.Error(), "context") {
		t.Errorf("expected error to mention 'context', got: %v", err)
	}
}

func TestComplete_WithAPIKeyAuth(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify API key is in query param, NOT in Authorization header
		if r.URL.Query().Get("key") != "my-api-key" {
			t.Errorf("expected key query param 'my-api-key', got %q", r.URL.Query().Get("key"))
		}
		if r.Header.Get("Authorization") != "" {
			t.Errorf("expected no Authorization header with API key auth, got %q", r.Header.Get("Authorization"))
		}

		resp := googleProvider.GenerateContentResponse{
			Candidates: []googleProvider.Candidate{
				{
					Content: &googleProvider.Content{
						Role:  "model",
						Parts: []googleProvider.Part{{Text: "Hi"}},
					},
					FinishReason: "STOP",
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := New("test-project", "us-central1",
		provider.WithAPIKey("my-api-key"),
		provider.WithBaseURL(server.URL),
	)

	resp, err := client.Complete(context.Background(), &types.CompletionRequest{
		Model:    "gemini-2.0-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Text() != "Hi" {
		t.Errorf("expected 'Hi', got %q", resp.Text())
	}
}

func TestStream_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify streaming URL
		if !strings.Contains(r.URL.Path, "streamGenerateContent") {
			t.Errorf("expected streaming URL, got %s", r.URL.Path)
		}

		// Return JSON array stream format
		chunks := []googleProvider.StreamChunk{
			{
				Candidates: []googleProvider.Candidate{
					{
						Content: &googleProvider.Content{
							Role:  "model",
							Parts: []googleProvider.Part{{Text: "Hello "}},
						},
					},
				},
			},
			{
				Candidates: []googleProvider.Candidate{
					{
						Content: &googleProvider.Content{
							Role:  "model",
							Parts: []googleProvider.Part{{Text: "World!"}},
						},
						FinishReason: "STOP",
					},
				},
				UsageMetadata: &googleProvider.UsageMetadata{
					PromptTokenCount:     3,
					CandidatesTokenCount: 7,
					TotalTokenCount:      10,
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chunks)
	}))
	defer server.Close()

	client := New("test-project", "us-central1",
		provider.WithAccessToken("tok"),
		provider.WithBaseURL(server.URL),
	)

	reader, err := client.Stream(context.Background(), &types.CompletionRequest{
		Model:    "gemini-2.0-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer reader.Close()

	var events []*types.StreamEvent
	for {
		event, err := reader.Next()
		if err != nil {
			t.Fatalf("unexpected stream error: %v", err)
		}
		if event == nil {
			break
		}
		events = append(events, event)
	}

	// Should have: start, content delta "Hello ", content delta "World!", done
	if len(events) < 3 {
		t.Fatalf("expected at least 3 events, got %d", len(events))
	}

	// First event: start
	if events[0].Type != types.StreamEventStart {
		t.Errorf("expected first event type 'start', got %q", events[0].Type)
	}

	// Check accumulated response
	resp := reader.Response()
	if resp == nil {
		t.Fatal("expected non-nil response after stream")
	}
	if resp.Provider != types.ProviderVertex {
		t.Errorf("expected provider 'vertex', got %q", resp.Provider)
	}

	fullText := ""
	for _, block := range resp.Content {
		if block.Type == types.ContentTypeText {
			fullText += block.Text
		}
	}
	if fullText != "Hello World!" {
		t.Errorf("expected accumulated text 'Hello World!', got %q", fullText)
	}
}

func TestStream_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(googleProvider.ErrorResponse{
			Error: &googleProvider.APIError{
				Code:    500,
				Message: "internal error",
				Status:  "INTERNAL",
			},
		})
	}))
	defer server.Close()

	client := New("test-project", "us-central1",
		provider.WithAccessToken("tok"),
		provider.WithBaseURL(server.URL),
	)

	_, err := client.Stream(context.Background(), &types.CompletionRequest{
		Model:    "gemini-2.0-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
	})

	if err == nil {
		t.Fatal("expected error for server error")
	}
}

func TestContains(t *testing.T) {
	tests := []struct {
		s        string
		substrs  []string
		expected bool
	}{
		{"context length exceeded", []string{"context", "token"}, true},
		{"too many tokens in request", []string{"context", "token"}, true},
		{"invalid argument", []string{"context", "token"}, false},
		{"", []string{"context"}, false},
		{"test", []string{}, false},
	}

	for _, tt := range tests {
		result := contains(tt.s, tt.substrs...)
		if result != tt.expected {
			t.Errorf("contains(%q, %v) = %v, expected %v", tt.s, tt.substrs, result, tt.expected)
		}
	}
}

// TestProviderInterface verifies that Client implements the Provider interface.
func TestProviderInterface(t *testing.T) {
	var _ provider.Provider = (*Client)(nil)
}

// TestBatchProviderInterface verifies that Client implements the BatchProvider interface.
func TestBatchProviderInterface(t *testing.T) {
	var _ provider.BatchProvider = (*Client)(nil)
}

func TestComplete_WithToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := googleProvider.GenerateContentResponse{
			Candidates: []googleProvider.Candidate{
				{
					Content: &googleProvider.Content{
						Role: "model",
						Parts: []googleProvider.Part{
							{
								FunctionCall: &googleProvider.FunctionCall{
									Name: "get_weather",
									Args: map[string]any{"location": "Paris"},
								},
							},
						},
					},
					FinishReason: "STOP",
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := New("test-project", "us-central1",
		provider.WithAccessToken("tok"),
		provider.WithBaseURL(server.URL),
	)

	resp, err := client.Complete(context.Background(), &types.CompletionRequest{
		Model:    "gemini-2.0-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "What's the weather?")},
		Tools: []types.Tool{
			{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters: types.JSONSchema{
					Type:       "object",
					Properties: map[string]types.JSONSchema{"location": {Type: "string"}},
				},
			},
		},
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}

	if resp.ToolCalls[0].Name != "get_weather" {
		t.Errorf("expected tool call name 'get_weather', got %q", resp.ToolCalls[0].Name)
	}

	input, ok := resp.ToolCalls[0].Input.(map[string]any)
	if !ok {
		t.Fatal("expected input to be map")
	}
	if input["location"] != "Paris" {
		t.Errorf("expected location 'Paris', got %v", input["location"])
	}
}

func TestBatchConvertStatus(t *testing.T) {
	client := New("proj", "loc", provider.WithAccessToken("tok"))

	tests := []struct {
		batch    *googleProvider.BatchJob
		expected provider.BatchStatus
	}{
		{
			batch:    &googleProvider.BatchJob{Done: true},
			expected: provider.BatchStatusCompleted,
		},
		{
			batch:    &googleProvider.BatchJob{Done: true, Error: &googleProvider.StatusError{Code: 500, Message: "fail"}},
			expected: provider.BatchStatusFailed,
		},
		{
			batch:    &googleProvider.BatchJob{Metadata: &googleProvider.BatchMetadata{State: "JOB_STATE_PENDING"}},
			expected: provider.BatchStatusPending,
		},
		{
			batch:    &googleProvider.BatchJob{Metadata: &googleProvider.BatchMetadata{State: "JOB_STATE_RUNNING"}},
			expected: provider.BatchStatusInProgress,
		},
		{
			batch:    &googleProvider.BatchJob{Metadata: &googleProvider.BatchMetadata{State: "JOB_STATE_SUCCEEDED"}},
			expected: provider.BatchStatusCompleted,
		},
		{
			batch:    &googleProvider.BatchJob{Metadata: &googleProvider.BatchMetadata{State: "JOB_STATE_FAILED"}},
			expected: provider.BatchStatusFailed,
		},
		{
			batch:    &googleProvider.BatchJob{Metadata: &googleProvider.BatchMetadata{State: "JOB_STATE_CANCELLED"}},
			expected: provider.BatchStatusCancelled,
		},
		{
			batch:    &googleProvider.BatchJob{Metadata: &googleProvider.BatchMetadata{State: "JOB_STATE_EXPIRED"}},
			expected: provider.BatchStatusExpired,
		},
		{
			batch:    &googleProvider.BatchJob{Metadata: &googleProvider.BatchMetadata{State: "BATCH_STATE_RUNNING"}},
			expected: provider.BatchStatusInProgress,
		},
		{
			batch:    &googleProvider.BatchJob{},
			expected: provider.BatchStatusPending,
		},
	}

	for _, tt := range tests {
		result := client.convertBatchStatus(tt.batch)
		if result != tt.expected {
			state := ""
			if tt.batch.Metadata != nil {
				state = tt.batch.Metadata.State
			}
			t.Errorf("convertBatchStatus(done=%v, state=%q) = %q, expected %q",
				tt.batch.Done, state, result, tt.expected)
		}
	}
}

func TestBatchConvertJob(t *testing.T) {
	client := New("proj", "loc", provider.WithAccessToken("tok"))

	batch := &googleProvider.BatchJob{
		Name: "projects/proj/locations/loc/batchPredictionJobs/123",
		Metadata: &googleProvider.BatchMetadata{
			DisplayName: "test-batch",
			State:       "JOB_STATE_RUNNING",
			CreateTime:  "2025-01-15T10:00:00Z",
		},
	}

	job := client.convertBatchJob(batch, "gemini-2.0-flash")

	if job.ID != "projects/proj/locations/loc/batchPredictionJobs/123" {
		t.Errorf("unexpected job ID: %s", job.ID)
	}
	if job.Provider != types.ProviderVertex {
		t.Errorf("expected provider 'vertex', got %q", job.Provider)
	}
	if job.Status != provider.BatchStatusInProgress {
		t.Errorf("expected status 'in_progress', got %q", job.Status)
	}
	if job.Metadata["model"] != "gemini-2.0-flash" {
		t.Errorf("expected model in metadata, got %v", job.Metadata["model"])
	}
	if job.Metadata["display_name"] != "test-batch" {
		t.Errorf("expected display_name in metadata, got %v", job.Metadata["display_name"])
	}
}

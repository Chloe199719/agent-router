// Package tests provides integration tests for the agent router.
// These tests run against real APIs and require valid API keys.
//
// Run with: go test -v ./tests/... -tags=integration
//
// Required environment variables (or .env file):
//   - OPENAI_API_KEY
//   - ANTHROPIC_API_KEY
//   - GOOGLE_API_KEY (optional)
//
//go:build integration

package tests

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/joho/godotenv"

	router "github.com/Chloe199719/agent-router"
	"github.com/Chloe199719/agent-router/pkg/types"
)

func init() {
	// Load .env file if it exists (from project root)
	godotenv.Load("../.env")
	godotenv.Load(".env")
}

// Test configuration - using cheapest/fastest models
const (
	openAIModel    = "gpt-4o-mini"
	anthropicModel = "claude-3-5-haiku-20241022"
	googleModel    = "gemini-2.0-flash"

	// Short timeout for tests
	testTimeout = 60 * time.Second
)

// getRouter creates a router with available providers
func getRouter(t *testing.T) *router.Router {
	t.Helper()

	var opts []router.Option

	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		opts = append(opts, router.WithOpenAI(key))
	}
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		opts = append(opts, router.WithAnthropic(key))
	}
	if key := os.Getenv("GOOGLE_API_KEY"); key != "" {
		opts = append(opts, router.WithGoogle(key))
	}

	if len(opts) == 0 {
		t.Skip("No API keys configured")
	}

	r, err := router.New(opts...)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	return r
}

func hasProvider(r *router.Router, p types.Provider) bool {
	for _, provider := range r.Providers() {
		if provider == p {
			return true
		}
	}
	return false
}

// ============================================================================
// Basic Completion Tests
// ============================================================================

func TestOpenAI_BasicCompletion(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderOpenAI) {
		t.Skip("OpenAI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderOpenAI,
		Model:     openAIModel,
		MaxTokens: types.Ptr(50),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Say 'hello' and nothing else."),
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	if resp.ID == "" {
		t.Error("Response ID is empty")
	}
	if resp.Model == "" {
		t.Error("Response model is empty")
	}
	text := resp.Text()
	if text == "" {
		t.Error("Response text is empty")
	}
	if !strings.Contains(strings.ToLower(text), "hello") {
		t.Errorf("Expected 'hello' in response, got: %s", text)
	}
	if resp.Usage.InputTokens == 0 {
		t.Error("Input tokens is 0")
	}
	if resp.Usage.OutputTokens == 0 {
		t.Error("Output tokens is 0")
	}

	t.Logf("Response: %s", text)
	t.Logf("Usage: %d input, %d output tokens", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}

func TestAnthropic_BasicCompletion(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderAnthropic) {
		t.Skip("Anthropic not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderAnthropic,
		Model:     anthropicModel,
		MaxTokens: types.Ptr(50),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Say 'hello' and nothing else."),
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	text := resp.Text()
	if text == "" {
		t.Error("Response text is empty")
	}
	if !strings.Contains(strings.ToLower(text), "hello") {
		t.Errorf("Expected 'hello' in response, got: %s", text)
	}

	t.Logf("Response: %s", text)
	t.Logf("Usage: %d input, %d output tokens", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}

func TestGoogle_BasicCompletion(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderGoogle) {
		t.Skip("Google not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderGoogle,
		Model:     googleModel,
		MaxTokens: types.Ptr(50),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Say 'hello' and nothing else."),
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	text := resp.Text()
	if text == "" {
		t.Error("Response text is empty")
	}
	if !strings.Contains(strings.ToLower(text), "hello") {
		t.Errorf("Expected 'hello' in response, got: %s", text)
	}

	t.Logf("Response: %s", text)
	t.Logf("Usage: %d input, %d output tokens", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}

// ============================================================================
// Streaming Tests
// ============================================================================

func TestOpenAI_Streaming(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderOpenAI) {
		t.Skip("OpenAI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	stream, err := r.Stream(ctx, &types.CompletionRequest{
		Provider:  types.ProviderOpenAI,
		Model:     openAIModel,
		MaxTokens: types.Ptr(50),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Count from 1 to 5."),
		},
	})

	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}
	defer stream.Close()

	var chunks int
	var text strings.Builder

	for {
		event, err := stream.Next()
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event == nil {
			break
		}

		switch event.Type {
		case types.StreamEventContentDelta:
			chunks++
			if event.Delta != nil {
				text.WriteString(event.Delta.Text)
			}
		case types.StreamEventDone:
			t.Logf("Stream done, stop reason: %s", event.StopReason)
		}
	}

	if chunks == 0 {
		t.Error("No content chunks received")
	}

	resp := stream.Response()
	if resp == nil {
		t.Error("No accumulated response")
	}

	t.Logf("Received %d chunks, text: %s", chunks, text.String())
}

func TestAnthropic_Streaming(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderAnthropic) {
		t.Skip("Anthropic not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	stream, err := r.Stream(ctx, &types.CompletionRequest{
		Provider:  types.ProviderAnthropic,
		Model:     anthropicModel,
		MaxTokens: types.Ptr(50),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Count from 1 to 5."),
		},
	})

	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}
	defer stream.Close()

	var chunks int
	var text strings.Builder

	for {
		event, err := stream.Next()
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event == nil {
			break
		}

		switch event.Type {
		case types.StreamEventContentDelta:
			chunks++
			if event.Delta != nil {
				text.WriteString(event.Delta.Text)
			}
		case types.StreamEventDone:
			t.Logf("Stream done, stop reason: %s", event.StopReason)
		}
	}

	if chunks == 0 {
		t.Error("No content chunks received")
	}

	t.Logf("Received %d chunks, text: %s", chunks, text.String())
}

func TestGoogle_Streaming(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderGoogle) {
		t.Skip("Google not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	stream, err := r.Stream(ctx, &types.CompletionRequest{
		Provider:  types.ProviderGoogle,
		Model:     googleModel,
		MaxTokens: types.Ptr(50),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Count from 1 to 5."),
		},
	})

	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}
	defer stream.Close()

	var chunks int
	var text strings.Builder

	for {
		event, err := stream.Next()
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event == nil {
			break
		}

		switch event.Type {
		case types.StreamEventContentDelta:
			chunks++
			if event.Delta != nil {
				text.WriteString(event.Delta.Text)
			}
		case types.StreamEventDone:
			t.Logf("Stream done, stop reason: %s", event.StopReason)
		}
	}

	if chunks == 0 {
		t.Error("No content chunks received")
	}

	t.Logf("Received %d chunks, text: %s", chunks, text.String())
}

// ============================================================================
// Structured Output (JSON Schema) Tests
// ============================================================================

type PersonInfo struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func getPersonSchema() types.JSONSchema {
	return types.JSONSchema{
		Type: "object",
		Properties: map[string]types.JSONSchema{
			"name": {Type: "string", Description: "The person's name"},
			"age":  {Type: "integer", Description: "The person's age"},
		},
		Required: []string{"name", "age"},
	}
}

func TestOpenAI_StructuredOutput(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderOpenAI) {
		t.Skip("OpenAI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, (&types.CompletionRequest{
		Provider:  types.ProviderOpenAI,
		Model:     openAIModel,
		MaxTokens: types.Ptr(100),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Extract: John Smith is 42 years old."),
		},
	}).WithJSONSchema("person_info", getPersonSchema()))

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	text := resp.Text()
	t.Logf("Response: %s", text)

	var person PersonInfo
	if err := json.Unmarshal([]byte(text), &person); err != nil {
		t.Fatalf("Failed to parse JSON: %v", err)
	}

	if person.Name == "" {
		t.Error("Name is empty")
	}
	if person.Age == 0 {
		t.Error("Age is 0")
	}

	t.Logf("Parsed: name=%s, age=%d", person.Name, person.Age)
}

func TestAnthropic_StructuredOutput(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderAnthropic) {
		t.Skip("Anthropic not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	// Note: Anthropic structured output via output_config requires specific model support.
	// If this test fails with "does not support output format", try a newer model
	// or use tool-based extraction instead.
	resp, err := r.Complete(ctx, (&types.CompletionRequest{
		Provider:  types.ProviderAnthropic,
		Model:     "claude-sonnet-4-5",
		MaxTokens: types.Ptr(100),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Extract: John Smith is 42 years old."),
		},
	}).WithJSONSchema("person_info", getPersonSchema()))

	if err != nil {
		// Skip if model doesn't support structured output
		if strings.Contains(err.Error(), "does not support output format") {
			t.Skip("Model does not support structured output format")
		}
		t.Fatalf("Completion failed: %v", err)
	}

	text := resp.Text()
	t.Logf("Response: %s", text)

	var person PersonInfo
	if err := json.Unmarshal([]byte(text), &person); err != nil {
		t.Fatalf("Failed to parse JSON: %v", err)
	}

	if person.Name == "" {
		t.Error("Name is empty")
	}
	if person.Age == 0 {
		t.Error("Age is 0")
	}

	t.Logf("Parsed: name=%s, age=%d", person.Name, person.Age)
}

func TestGoogle_StructuredOutput(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderGoogle) {
		t.Skip("Google not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, (&types.CompletionRequest{
		Provider:  types.ProviderGoogle,
		Model:     googleModel,
		MaxTokens: types.Ptr(100),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Extract: John Smith is 42 years old."),
		},
	}).WithJSONSchema("person_info", getPersonSchema()))

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	text := resp.Text()
	t.Logf("Response: %s", text)

	var person PersonInfo
	if err := json.Unmarshal([]byte(text), &person); err != nil {
		t.Fatalf("Failed to parse JSON: %v", err)
	}

	if person.Name == "" {
		t.Error("Name is empty")
	}
	if person.Age == 0 {
		t.Error("Age is 0")
	}

	t.Logf("Parsed: name=%s, age=%d", person.Name, person.Age)
}

// ============================================================================
// Tool Calling Tests
// ============================================================================

func getWeatherTool() types.Tool {
	return types.Tool{
		Name:        "get_weather",
		Description: "Get the current weather for a location",
		Parameters: types.JSONSchema{
			Type: "object",
			Properties: map[string]types.JSONSchema{
				"location": {
					Type:        "string",
					Description: "The city name",
				},
			},
			Required: []string{"location"},
		},
	}
}

func TestOpenAI_ToolCalling(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderOpenAI) {
		t.Skip("OpenAI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, (&types.CompletionRequest{
		Provider:  types.ProviderOpenAI,
		Model:     openAIModel,
		MaxTokens: types.Ptr(100),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "What's the weather in Paris?"),
		},
	}).WithTools(getWeatherTool()))

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	if !resp.HasToolCalls() {
		t.Fatalf("Expected tool calls, got none. Response: %s", resp.Text())
	}

	tc := resp.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Errorf("Expected tool name 'get_weather', got '%s'", tc.Name)
	}

	input, ok := tc.Input.(map[string]any)
	if !ok {
		t.Fatalf("Tool input is not a map: %T", tc.Input)
	}

	location, ok := input["location"].(string)
	if !ok || location == "" {
		t.Error("Tool input missing 'location'")
	}

	t.Logf("Tool call: %s(%v)", tc.Name, tc.Input)

	// Test continuing conversation with tool result
	messages := []types.Message{
		types.NewTextMessage(types.RoleUser, "What's the weather in Paris?"),
		{Role: types.RoleAssistant, Content: resp.Content},
		types.NewToolResultMessage(tc.ID, `{"temperature": 18, "condition": "Cloudy"}`, false),
	}

	resp2, err := r.Complete(ctx, (&types.CompletionRequest{
		Provider:  types.ProviderOpenAI,
		Model:     openAIModel,
		MaxTokens: types.Ptr(100),
		Messages:  messages,
	}).WithTools(getWeatherTool()))

	if err != nil {
		t.Fatalf("Follow-up completion failed: %v", err)
	}

	t.Logf("Follow-up response: %s", resp2.Text())
}

func TestAnthropic_ToolCalling(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderAnthropic) {
		t.Skip("Anthropic not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, (&types.CompletionRequest{
		Provider:  types.ProviderAnthropic,
		Model:     anthropicModel,
		MaxTokens: types.Ptr(100),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "What's the weather in Paris?"),
		},
	}).WithTools(getWeatherTool()))

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	if !resp.HasToolCalls() {
		t.Fatalf("Expected tool calls, got none. Response: %s", resp.Text())
	}

	tc := resp.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Errorf("Expected tool name 'get_weather', got '%s'", tc.Name)
	}

	t.Logf("Tool call: %s(%v)", tc.Name, tc.Input)

	// Test continuing conversation with tool result
	messages := []types.Message{
		types.NewTextMessage(types.RoleUser, "What's the weather in Paris?"),
		{Role: types.RoleAssistant, Content: resp.Content},
		types.NewToolResultMessage(tc.ID, `{"temperature": 18, "condition": "Cloudy"}`, false),
	}

	resp2, err := r.Complete(ctx, (&types.CompletionRequest{
		Provider:  types.ProviderAnthropic,
		Model:     anthropicModel,
		MaxTokens: types.Ptr(100),
		Messages:  messages,
	}).WithTools(getWeatherTool()))

	if err != nil {
		t.Fatalf("Follow-up completion failed: %v", err)
	}

	t.Logf("Follow-up response: %s", resp2.Text())
}

func TestGoogle_ToolCalling(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderGoogle) {
		t.Skip("Google not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, (&types.CompletionRequest{
		Provider:  types.ProviderGoogle,
		Model:     googleModel,
		MaxTokens: types.Ptr(100),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "What's the weather in Paris?"),
		},
	}).WithTools(getWeatherTool()))

	if err != nil {
		// Skip on transient errors
		if strings.Contains(err.Error(), "unavailable") || strings.Contains(err.Error(), "503") {
			t.Skip("Google service temporarily unavailable")
		}
		t.Fatalf("Completion failed: %v", err)
	}

	if !resp.HasToolCalls() {
		t.Fatalf("Expected tool calls, got none. Response: %s", resp.Text())
	}

	tc := resp.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Errorf("Expected tool name 'get_weather', got '%s'", tc.Name)
	}

	t.Logf("Tool call: %s(%v)", tc.Name, tc.Input)
}

// ============================================================================
// System Message Tests
// ============================================================================

func TestOpenAI_SystemMessage(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderOpenAI) {
		t.Skip("OpenAI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderOpenAI,
		Model:     openAIModel,
		MaxTokens: types.Ptr(50),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleSystem, "You are a pirate. Always respond like a pirate."),
			types.NewTextMessage(types.RoleUser, "Hello!"),
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	text := strings.ToLower(resp.Text())
	// Check for pirate-like response
	hasPirateWord := strings.Contains(text, "ahoy") ||
		strings.Contains(text, "matey") ||
		strings.Contains(text, "arr") ||
		strings.Contains(text, "ye")

	if !hasPirateWord {
		t.Logf("Warning: Response may not be pirate-like: %s", resp.Text())
	}

	t.Logf("Response: %s", resp.Text())
}

func TestAnthropic_SystemMessage(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderAnthropic) {
		t.Skip("Anthropic not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderAnthropic,
		Model:     anthropicModel,
		MaxTokens: types.Ptr(50),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleSystem, "You are a pirate. Always respond like a pirate."),
			types.NewTextMessage(types.RoleUser, "Hello!"),
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	t.Logf("Response: %s", resp.Text())
}

// ============================================================================
// Multi-turn Conversation Tests
// ============================================================================

func TestOpenAI_MultiTurn(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderOpenAI) {
		t.Skip("OpenAI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderOpenAI,
		Model:     openAIModel,
		MaxTokens: types.Ptr(50),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "My name is Alice."),
			types.NewTextMessage(types.RoleAssistant, "Nice to meet you, Alice!"),
			types.NewTextMessage(types.RoleUser, "What is my name?"),
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	text := strings.ToLower(resp.Text())
	if !strings.Contains(text, "alice") {
		t.Errorf("Expected 'Alice' in response, got: %s", resp.Text())
	}

	t.Logf("Response: %s", resp.Text())
}

// ============================================================================
// Temperature Test
// ============================================================================

func TestOpenAI_Temperature(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderOpenAI) {
		t.Skip("OpenAI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	// Test with temperature 0 (deterministic)
	temp := 0.0
	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:    types.ProviderOpenAI,
		Model:       openAIModel,
		MaxTokens:   types.Ptr(10),
		Temperature: &temp,
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Say exactly: 'test'"),
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	t.Logf("Response with temp=0: %s", resp.Text())
}

// ============================================================================
// Stop Sequences Test
// ============================================================================

func TestOpenAI_StopSequences(t *testing.T) {
	r := getRouter(t)
	if !hasProvider(r, types.ProviderOpenAI) {
		t.Skip("OpenAI not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider:      types.ProviderOpenAI,
		Model:         openAIModel,
		MaxTokens:     types.Ptr(100),
		StopSequences: []string{"3"},
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Count from 1 to 10: 1, 2,"),
		},
	})

	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	text := resp.Text()
	if strings.Contains(text, "4") || strings.Contains(text, "5") {
		t.Errorf("Expected stop at '3', got: %s", text)
	}

	t.Logf("Response (should stop at 3): %s", text)
	t.Logf("Stop reason: %s", resp.StopReason)
}

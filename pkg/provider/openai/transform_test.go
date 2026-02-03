package openai

import (
	"testing"

	"github.com/Chloe199719/agent-router/pkg/types"
)

func TestTransformRequest_Basic(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gpt-4o",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Hello"),
		},
	}

	result := transformer.TransformRequest(req)

	if result.Model != "gpt-4o" {
		t.Errorf("expected model 'gpt-4o', got %q", result.Model)
	}

	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}

	if result.Messages[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", result.Messages[0].Role)
	}

	// Content should be string for simple messages
	if content, ok := result.Messages[0].Content.(string); !ok || content != "Hello" {
		t.Errorf("expected content 'Hello', got %v", result.Messages[0].Content)
	}
}

func TestTransformRequest_WithParameters(t *testing.T) {
	transformer := NewTransformer()

	maxTokens := 100
	temp := 0.7
	topP := 0.9

	req := &types.CompletionRequest{
		Model:         "gpt-4o",
		Messages:      []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
		MaxTokens:     &maxTokens,
		Temperature:   &temp,
		TopP:          &topP,
		StopSequences: []string{"END"},
	}

	result := transformer.TransformRequest(req)

	if *result.MaxTokens != 100 {
		t.Errorf("expected max_tokens 100, got %d", *result.MaxTokens)
	}

	if *result.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %f", *result.Temperature)
	}

	if *result.TopP != 0.9 {
		t.Errorf("expected top_p 0.9, got %f", *result.TopP)
	}

	if len(result.Stop) != 1 || result.Stop[0] != "END" {
		t.Errorf("expected stop sequence 'END', got %v", result.Stop)
	}
}

func TestTransformRequest_Streaming(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model:    "gpt-4o",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
		Stream:   true,
	}

	result := transformer.TransformRequest(req)

	if !result.Stream {
		t.Error("expected stream to be true")
	}

	if result.StreamOptions == nil {
		t.Fatal("expected StreamOptions to be set")
	}

	if !result.StreamOptions.IncludeUsage {
		t.Error("expected IncludeUsage to be true")
	}
}

func TestTransformRequest_SystemMessage(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gpt-4o",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleSystem, "You are a helpful assistant"),
			types.NewTextMessage(types.RoleUser, "Hello"),
		},
	}

	result := transformer.TransformRequest(req)

	if len(result.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result.Messages))
	}

	if result.Messages[0].Role != "system" {
		t.Errorf("expected first message role 'system', got %q", result.Messages[0].Role)
	}
}

func TestTransformRequest_ToolResult(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gpt-4o",
		Messages: []types.Message{
			types.NewToolResultMessage("call_123", `{"result": "ok"}`, false),
		},
	}

	result := transformer.TransformRequest(req)

	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}

	msg := result.Messages[0]
	if msg.Role != "tool" {
		t.Errorf("expected role 'tool', got %q", msg.Role)
	}

	if msg.ToolCallID != "call_123" {
		t.Errorf("expected tool_call_id 'call_123', got %q", msg.ToolCallID)
	}

	if content, ok := msg.Content.(string); !ok || content != `{"result": "ok"}` {
		t.Errorf("expected content to be tool result, got %v", msg.Content)
	}
}

func TestTransformRequest_AssistantWithToolCalls(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gpt-4o",
		Messages: []types.Message{
			{
				Role: types.RoleAssistant,
				Content: []types.ContentBlock{
					{Type: types.ContentTypeText, Text: "Let me check the weather"},
					{
						Type:      types.ContentTypeToolUse,
						ToolUseID: "call_abc",
						ToolName:  "get_weather",
						ToolInput: map[string]any{"location": "Paris"},
					},
				},
			},
		},
	}

	result := transformer.TransformRequest(req)

	msg := result.Messages[0]
	if msg.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", msg.Role)
	}

	if len(msg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(msg.ToolCalls))
	}

	tc := msg.ToolCalls[0]
	if tc.ID != "call_abc" {
		t.Errorf("expected tool call ID 'call_abc', got %q", tc.ID)
	}

	if tc.Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", tc.Function.Name)
	}
}

func TestTransformRequest_MultipartImage(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gpt-4o",
		Messages: []types.Message{
			{
				Role: types.RoleUser,
				Content: []types.ContentBlock{
					{Type: types.ContentTypeText, Text: "What's in this image?"},
					{Type: types.ContentTypeImage, ImageURL: "https://example.com/image.jpg"},
				},
			},
		},
	}

	result := transformer.TransformRequest(req)

	msg := result.Messages[0]
	parts, ok := msg.Content.([]ContentPart)
	if !ok {
		t.Fatal("expected content to be []ContentPart")
	}

	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}

	if parts[0].Type != "text" {
		t.Errorf("expected first part type 'text', got %q", parts[0].Type)
	}

	if parts[1].Type != "image_url" {
		t.Errorf("expected second part type 'image_url', got %q", parts[1].Type)
	}

	if parts[1].ImageURL.URL != "https://example.com/image.jpg" {
		t.Errorf("expected image URL, got %q", parts[1].ImageURL.URL)
	}
}

func TestTransformRequest_Base64Image(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gpt-4o",
		Messages: []types.Message{
			{
				Role: types.RoleUser,
				Content: []types.ContentBlock{
					{
						Type:        types.ContentTypeImage,
						ImageBase64: "base64data",
						MediaType:   "image/png",
					},
				},
			},
		},
	}

	result := transformer.TransformRequest(req)

	msg := result.Messages[0]
	parts, ok := msg.Content.([]ContentPart)
	if !ok {
		t.Fatal("expected content to be []ContentPart")
	}

	expectedURL := "data:image/png;base64,base64data"
	if parts[0].ImageURL.URL != expectedURL {
		t.Errorf("expected data URL %q, got %q", expectedURL, parts[0].ImageURL.URL)
	}
}

func TestTransformRequest_Tools(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model:    "gpt-4o",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
		Tools: []types.Tool{
			{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters: types.JSONSchema{
					Type: "object",
					Properties: map[string]types.JSONSchema{
						"location": {Type: "string"},
					},
				},
			},
		},
	}

	result := transformer.TransformRequest(req)

	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result.Tools))
	}

	tool := result.Tools[0]
	if tool.Type != "function" {
		t.Errorf("expected tool type 'function', got %q", tool.Type)
	}

	if tool.Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", tool.Function.Name)
	}
}

func TestTransformRequest_ToolChoice(t *testing.T) {
	transformer := NewTransformer()

	tests := []struct {
		choice   *types.ToolChoice
		expected any
	}{
		{&types.ToolChoice{Type: types.ToolChoiceAuto}, "auto"},
		{&types.ToolChoice{Type: types.ToolChoiceRequired}, "required"},
		{&types.ToolChoice{Type: types.ToolChoiceNone}, "none"},
	}

	for _, tt := range tests {
		req := &types.CompletionRequest{
			Model:      "gpt-4o",
			Messages:   []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
			ToolChoice: tt.choice,
		}

		result := transformer.TransformRequest(req)

		if result.ToolChoice != tt.expected {
			t.Errorf("expected tool choice %v, got %v", tt.expected, result.ToolChoice)
		}
	}
}

func TestTransformRequest_ToolChoiceSpecific(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model:    "gpt-4o",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
		ToolChoice: &types.ToolChoice{
			Type: types.ToolChoiceTool,
			Name: "get_weather",
		},
	}

	result := transformer.TransformRequest(req)

	tc, ok := result.ToolChoice.(ToolChoiceObject)
	if !ok {
		t.Fatal("expected ToolChoiceObject")
	}

	if tc.Type != "function" {
		t.Errorf("expected type 'function', got %q", tc.Type)
	}

	if tc.Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", tc.Function.Name)
	}
}

func TestTransformResponse(t *testing.T) {
	transformer := NewTransformer()

	resp := &ChatCompletionResponse{
		ID:      "chatcmpl-123",
		Model:   "gpt-4o-2024-05-13",
		Created: 1234567890,
		Choices: []Choice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: "Hello!",
				},
				FinishReason: "stop",
			},
		},
		Usage: &Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	result := transformer.TransformResponse(resp)

	if result.ID != "chatcmpl-123" {
		t.Errorf("expected ID 'chatcmpl-123', got %q", result.ID)
	}

	if result.Provider != types.ProviderOpenAI {
		t.Errorf("expected provider OpenAI, got %q", result.Provider)
	}

	if result.Model != "gpt-4o-2024-05-13" {
		t.Errorf("expected model 'gpt-4o-2024-05-13', got %q", result.Model)
	}

	if result.Text() != "Hello!" {
		t.Errorf("expected text 'Hello!', got %q", result.Text())
	}

	if result.StopReason != types.StopReasonEnd {
		t.Errorf("expected stop reason 'end', got %q", result.StopReason)
	}

	if result.Usage.InputTokens != 10 {
		t.Errorf("expected 10 input tokens, got %d", result.Usage.InputTokens)
	}

	if result.Usage.OutputTokens != 5 {
		t.Errorf("expected 5 output tokens, got %d", result.Usage.OutputTokens)
	}
}

func TestTransformResponse_WithToolCalls(t *testing.T) {
	transformer := NewTransformer()

	resp := &ChatCompletionResponse{
		ID:    "chatcmpl-123",
		Model: "gpt-4o",
		Choices: []Choice{
			{
				Message: ChatMessage{
					Role: "assistant",
					ToolCalls: []ToolCall{
						{
							ID:   "call_abc",
							Type: "function",
							Function: FunctionCall{
								Name:      "get_weather",
								Arguments: `{"location":"Paris"}`,
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
	}

	result := transformer.TransformResponse(resp)

	if result.StopReason != types.StopReasonToolUse {
		t.Errorf("expected stop reason 'tool_use', got %q", result.StopReason)
	}

	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
	}

	tc := result.ToolCalls[0]
	if tc.ID != "call_abc" {
		t.Errorf("expected tool call ID 'call_abc', got %q", tc.ID)
	}

	if tc.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", tc.Name)
	}

	input, ok := tc.Input.(map[string]any)
	if !ok {
		t.Fatal("expected input to be map")
	}

	if input["location"] != "Paris" {
		t.Errorf("expected location 'Paris', got %v", input["location"])
	}
}

func TestTransformResponse_Nil(t *testing.T) {
	transformer := NewTransformer()

	result := transformer.TransformResponse(nil)
	if result != nil {
		t.Error("expected nil for nil input")
	}

	result = transformer.TransformResponse(&ChatCompletionResponse{Choices: []Choice{}})
	if result != nil {
		t.Error("expected nil for empty choices")
	}
}

func TestTransformStopReason(t *testing.T) {
	transformer := NewTransformer()

	tests := []struct {
		reason   string
		expected types.StopReason
	}{
		{"stop", types.StopReasonEnd},
		{"length", types.StopReasonMaxTokens},
		{"tool_calls", types.StopReasonToolUse},
		{"content_filter", types.StopReasonContentFilter},
		{"unknown", types.StopReasonEnd},
		{"", types.StopReasonEnd},
	}

	for _, tt := range tests {
		result := transformer.transformStopReason(tt.reason)
		if result != tt.expected {
			t.Errorf("transformStopReason(%q) = %q, expected %q", tt.reason, result, tt.expected)
		}
	}
}

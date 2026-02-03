package anthropic

import (
	"testing"

	"github.com/Chloe199719/agent-router/pkg/types"
)

func TestTransformRequest_Basic(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Hello"),
		},
	}

	result := transformer.TransformRequest(req)

	if result.Model != "claude-sonnet-4-20250514" {
		t.Errorf("expected model 'claude-sonnet-4-20250514', got %q", result.Model)
	}

	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}

	if result.Messages[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", result.Messages[0].Role)
	}

	// Default max_tokens
	if result.MaxTokens != 8192 {
		t.Errorf("expected default max_tokens 8192, got %d", result.MaxTokens)
	}
}

func TestTransformRequest_WithMaxTokens(t *testing.T) {
	transformer := NewTransformer()

	maxTokens := 1000
	req := &types.CompletionRequest{
		Model:     "claude-sonnet-4-20250514",
		Messages:  []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
		MaxTokens: &maxTokens,
	}

	result := transformer.TransformRequest(req)

	if result.MaxTokens != 1000 {
		t.Errorf("expected max_tokens 1000, got %d", result.MaxTokens)
	}
}

func TestTransformRequest_SystemMessage(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleSystem, "You are a helpful assistant"),
			types.NewTextMessage(types.RoleUser, "Hello"),
		},
	}

	result := transformer.TransformRequest(req)

	// System should be extracted
	if result.System != "You are a helpful assistant" {
		t.Errorf("expected system message, got %q", result.System)
	}

	// Only user message should remain
	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}

	if result.Messages[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", result.Messages[0].Role)
	}
}

func TestTransformRequest_MultipleSystemMessages(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleSystem, "Line 1"),
			types.NewTextMessage(types.RoleSystem, "Line 2"),
			types.NewTextMessage(types.RoleUser, "Hello"),
		},
	}

	result := transformer.TransformRequest(req)

	expected := "Line 1\nLine 2"
	if result.System != expected {
		t.Errorf("expected system %q, got %q", expected, result.System)
	}
}

func TestTransformRequest_ToolResult(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []types.Message{
			types.NewToolResultMessage("toolu_123", `{"result": "ok"}`, false),
		},
	}

	result := transformer.TransformRequest(req)

	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}

	msg := result.Messages[0]
	if msg.Role != "user" {
		t.Errorf("expected role 'user' for tool result, got %q", msg.Role)
	}

	blocks, ok := msg.Content.([]ContentBlock)
	if !ok {
		t.Fatal("expected content to be []ContentBlock")
	}

	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}

	if blocks[0].Type != "tool_result" {
		t.Errorf("expected type 'tool_result', got %q", blocks[0].Type)
	}

	if blocks[0].ToolUseID != "toolu_123" {
		t.Errorf("expected tool_use_id 'toolu_123', got %q", blocks[0].ToolUseID)
	}
}

func TestTransformRequest_AssistantWithToolUse(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []types.Message{
			{
				Role: types.RoleAssistant,
				Content: []types.ContentBlock{
					{Type: types.ContentTypeText, Text: "Checking weather"},
					{
						Type:      types.ContentTypeToolUse,
						ToolUseID: "toolu_abc",
						ToolName:  "get_weather",
						ToolInput: map[string]any{"location": "Paris"},
					},
				},
			},
		},
	}

	result := transformer.TransformRequest(req)

	msg := result.Messages[0]
	blocks, ok := msg.Content.([]ContentBlock)
	if !ok {
		t.Fatal("expected content to be []ContentBlock")
	}

	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}

	if blocks[0].Type != "text" {
		t.Errorf("expected first block type 'text', got %q", blocks[0].Type)
	}

	if blocks[1].Type != "tool_use" {
		t.Errorf("expected second block type 'tool_use', got %q", blocks[1].Type)
	}

	if blocks[1].ID != "toolu_abc" {
		t.Errorf("expected ID 'toolu_abc', got %q", blocks[1].ID)
	}
}

func TestTransformRequest_Image(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []types.Message{
			{
				Role: types.RoleUser,
				Content: []types.ContentBlock{
					{Type: types.ContentTypeText, Text: "What's this?"},
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

	blocks, ok := result.Messages[0].Content.([]ContentBlock)
	if !ok {
		t.Fatal("expected content to be []ContentBlock")
	}

	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}

	imgBlock := blocks[1]
	if imgBlock.Type != "image" {
		t.Errorf("expected type 'image', got %q", imgBlock.Type)
	}

	if imgBlock.Source == nil {
		t.Fatal("expected Source to be non-nil")
	}

	if imgBlock.Source.Type != "base64" {
		t.Errorf("expected source type 'base64', got %q", imgBlock.Source.Type)
	}

	if imgBlock.Source.Data != "base64data" {
		t.Errorf("expected data 'base64data', got %q", imgBlock.Source.Data)
	}
}

func TestTransformRequest_Tools(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model:    "claude-sonnet-4-20250514",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
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
	}

	result := transformer.TransformRequest(req)

	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result.Tools))
	}

	if result.Tools[0].Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", result.Tools[0].Name)
	}
}

func TestTransformRequest_ToolChoice(t *testing.T) {
	transformer := NewTransformer()

	tests := []struct {
		choice   *types.ToolChoice
		expected string
	}{
		{&types.ToolChoice{Type: types.ToolChoiceAuto}, "auto"},
		{&types.ToolChoice{Type: types.ToolChoiceRequired}, "any"},
		{&types.ToolChoice{Type: types.ToolChoiceNone}, "none"},
	}

	for _, tt := range tests {
		req := &types.CompletionRequest{
			Model:      "claude-sonnet-4-20250514",
			Messages:   []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
			ToolChoice: tt.choice,
		}

		result := transformer.TransformRequest(req)

		if result.ToolChoice.Type != tt.expected {
			t.Errorf("expected tool choice type %q, got %q", tt.expected, result.ToolChoice.Type)
		}
	}
}

func TestTransformRequest_ToolChoiceSpecific(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model:    "claude-sonnet-4-20250514",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
		ToolChoice: &types.ToolChoice{
			Type: types.ToolChoiceTool,
			Name: "get_weather",
		},
	}

	result := transformer.TransformRequest(req)

	if result.ToolChoice.Type != "tool" {
		t.Errorf("expected type 'tool', got %q", result.ToolChoice.Type)
	}

	if result.ToolChoice.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", result.ToolChoice.Name)
	}
}

func TestTransformResponse(t *testing.T) {
	transformer := NewTransformer()

	resp := &MessagesResponse{
		ID:    "msg_123",
		Model: "claude-sonnet-4-20250514",
		Content: []ContentBlock{
			{Type: "text", Text: "Hello!"},
		},
		StopReason: "end_turn",
		Usage: Usage{
			InputTokens:  10,
			OutputTokens: 5,
		},
	}

	result := transformer.TransformResponse(resp)

	if result.ID != "msg_123" {
		t.Errorf("expected ID 'msg_123', got %q", result.ID)
	}

	if result.Provider != types.ProviderAnthropic {
		t.Errorf("expected provider Anthropic, got %q", result.Provider)
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

	if result.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", result.Usage.TotalTokens)
	}
}

func TestTransformResponse_WithToolUse(t *testing.T) {
	transformer := NewTransformer()

	resp := &MessagesResponse{
		ID:    "msg_123",
		Model: "claude-sonnet-4-20250514",
		Content: []ContentBlock{
			{Type: "text", Text: "Let me check"},
			{
				Type:  "tool_use",
				ID:    "toolu_abc",
				Name:  "get_weather",
				Input: map[string]any{"location": "Paris"},
			},
		},
		StopReason: "tool_use",
		Usage:      Usage{InputTokens: 10, OutputTokens: 20},
	}

	result := transformer.TransformResponse(resp)

	if result.StopReason != types.StopReasonToolUse {
		t.Errorf("expected stop reason 'tool_use', got %q", result.StopReason)
	}

	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
	}

	tc := result.ToolCalls[0]
	if tc.ID != "toolu_abc" {
		t.Errorf("expected ID 'toolu_abc', got %q", tc.ID)
	}

	if tc.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", tc.Name)
	}
}

func TestTransformResponse_Nil(t *testing.T) {
	transformer := NewTransformer()

	result := transformer.TransformResponse(nil)
	if result != nil {
		t.Error("expected nil for nil input")
	}
}

func TestTransformStopReason(t *testing.T) {
	transformer := NewTransformer()

	tests := []struct {
		reason   string
		expected types.StopReason
	}{
		{"end_turn", types.StopReasonEnd},
		{"max_tokens", types.StopReasonMaxTokens},
		{"tool_use", types.StopReasonToolUse},
		{"stop_sequence", types.StopReasonStopSequence},
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

func TestMapRole(t *testing.T) {
	transformer := NewTransformer()

	tests := []struct {
		role     types.Role
		expected string
	}{
		{types.RoleUser, "user"},
		{types.RoleTool, "user"},
		{types.RoleAssistant, "assistant"},
		{types.RoleSystem, "user"}, // System is handled separately
	}

	for _, tt := range tests {
		result := transformer.mapRole(tt.role)
		if result != tt.expected {
			t.Errorf("mapRole(%q) = %q, expected %q", tt.role, result, tt.expected)
		}
	}
}

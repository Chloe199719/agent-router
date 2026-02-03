package google

import (
	"testing"

	"github.com/Chloe199719/agent-router/pkg/types"
)

func TestTransformRequest_Basic(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gemini-2.5-flash",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Hello"),
		},
	}

	result := transformer.TransformRequest(req)

	if len(result.Contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(result.Contents))
	}

	if result.Contents[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", result.Contents[0].Role)
	}

	if len(result.Contents[0].Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(result.Contents[0].Parts))
	}

	if result.Contents[0].Parts[0].Text != "Hello" {
		t.Errorf("expected text 'Hello', got %q", result.Contents[0].Parts[0].Text)
	}
}

func TestTransformRequest_WithParameters(t *testing.T) {
	transformer := NewTransformer()

	maxTokens := 100
	temp := 0.7
	topP := 0.9
	topK := 40

	req := &types.CompletionRequest{
		Model:         "gemini-2.5-flash",
		Messages:      []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
		MaxTokens:     &maxTokens,
		Temperature:   &temp,
		TopP:          &topP,
		TopK:          &topK,
		StopSequences: []string{"END"},
	}

	result := transformer.TransformRequest(req)

	if result.GenerationConfig == nil {
		t.Fatal("expected GenerationConfig to be non-nil")
	}

	if *result.GenerationConfig.MaxOutputTokens != 100 {
		t.Errorf("expected max_output_tokens 100, got %d", *result.GenerationConfig.MaxOutputTokens)
	}

	if *result.GenerationConfig.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %f", *result.GenerationConfig.Temperature)
	}

	if *result.GenerationConfig.TopP != 0.9 {
		t.Errorf("expected top_p 0.9, got %f", *result.GenerationConfig.TopP)
	}

	if *result.GenerationConfig.TopK != 40 {
		t.Errorf("expected top_k 40, got %d", *result.GenerationConfig.TopK)
	}

	if len(result.GenerationConfig.StopSequences) != 1 || result.GenerationConfig.StopSequences[0] != "END" {
		t.Errorf("expected stop sequence 'END', got %v", result.GenerationConfig.StopSequences)
	}
}

func TestTransformRequest_SystemMessage(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gemini-2.5-flash",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleSystem, "You are a helpful assistant"),
			types.NewTextMessage(types.RoleUser, "Hello"),
		},
	}

	result := transformer.TransformRequest(req)

	if result.SystemInstruction == nil {
		t.Fatal("expected SystemInstruction to be non-nil")
	}

	if len(result.SystemInstruction.Parts) != 1 {
		t.Fatalf("expected 1 system part, got %d", len(result.SystemInstruction.Parts))
	}

	if result.SystemInstruction.Parts[0].Text != "You are a helpful assistant" {
		t.Errorf("expected system text, got %q", result.SystemInstruction.Parts[0].Text)
	}

	// Only user message should be in contents
	if len(result.Contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(result.Contents))
	}
}

func TestTransformRequest_MultiTurn(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gemini-2.5-flash",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Hello"),
			types.NewTextMessage(types.RoleAssistant, "Hi there!"),
			types.NewTextMessage(types.RoleUser, "How are you?"),
		},
	}

	result := transformer.TransformRequest(req)

	if len(result.Contents) != 3 {
		t.Fatalf("expected 3 contents, got %d", len(result.Contents))
	}

	if result.Contents[0].Role != "user" {
		t.Errorf("expected first role 'user', got %q", result.Contents[0].Role)
	}

	if result.Contents[1].Role != "model" {
		t.Errorf("expected second role 'model', got %q", result.Contents[1].Role)
	}

	if result.Contents[2].Role != "user" {
		t.Errorf("expected third role 'user', got %q", result.Contents[2].Role)
	}
}

func TestTransformRequest_Image(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gemini-2.5-flash",
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

	parts := result.Contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}

	if parts[0].Text != "What's this?" {
		t.Errorf("expected first part text, got %q", parts[0].Text)
	}

	if parts[1].InlineData == nil {
		t.Fatal("expected InlineData to be non-nil")
	}

	if parts[1].InlineData.MimeType != "image/png" {
		t.Errorf("expected mime type 'image/png', got %q", parts[1].InlineData.MimeType)
	}

	if parts[1].InlineData.Data != "base64data" {
		t.Errorf("expected data 'base64data', got %q", parts[1].InlineData.Data)
	}
}

func TestTransformRequest_ImageURL(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gemini-2.5-flash",
		Messages: []types.Message{
			{
				Role: types.RoleUser,
				Content: []types.ContentBlock{
					{
						Type:      types.ContentTypeImage,
						ImageURL:  "gs://bucket/image.png",
						MediaType: "image/png",
					},
				},
			},
		},
	}

	result := transformer.TransformRequest(req)

	parts := result.Contents[0].Parts
	if parts[0].FileData == nil {
		t.Fatal("expected FileData to be non-nil")
	}

	if parts[0].FileData.FileURI != "gs://bucket/image.png" {
		t.Errorf("expected file URI, got %q", parts[0].FileData.FileURI)
	}
}

func TestTransformRequest_ToolUse(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gemini-2.5-flash",
		Messages: []types.Message{
			{
				Role: types.RoleAssistant,
				Content: []types.ContentBlock{
					{
						Type:      types.ContentTypeToolUse,
						ToolName:  "get_weather",
						ToolInput: map[string]any{"location": "Paris"},
					},
				},
			},
		},
	}

	result := transformer.TransformRequest(req)

	parts := result.Contents[0].Parts
	if parts[0].FunctionCall == nil {
		t.Fatal("expected FunctionCall to be non-nil")
	}

	if parts[0].FunctionCall.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", parts[0].FunctionCall.Name)
	}

	if parts[0].FunctionCall.Args["location"] != "Paris" {
		t.Errorf("expected location 'Paris', got %v", parts[0].FunctionCall.Args["location"])
	}
}

func TestTransformRequest_ToolResult(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gemini-2.5-flash",
		Messages: []types.Message{
			{
				Role: types.RoleTool,
				Content: []types.ContentBlock{
					{
						Type:     types.ContentTypeToolResult,
						ToolName: "get_weather",
						Text:     `{"temperature": 22}`,
					},
				},
			},
		},
	}

	result := transformer.TransformRequest(req)

	parts := result.Contents[0].Parts
	if parts[0].FunctionResponse == nil {
		t.Fatal("expected FunctionResponse to be non-nil")
	}

	if parts[0].FunctionResponse.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", parts[0].FunctionResponse.Name)
	}

	// Should have parsed the JSON
	if parts[0].FunctionResponse.Response["temperature"] != float64(22) {
		t.Errorf("expected temperature 22, got %v", parts[0].FunctionResponse.Response["temperature"])
	}
}

func TestTransformRequest_ToolResultNonJSON(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model: "gemini-2.5-flash",
		Messages: []types.Message{
			{
				Role: types.RoleTool,
				Content: []types.ContentBlock{
					{
						Type:     types.ContentTypeToolResult,
						ToolName: "search",
						Text:     "plain text result",
					},
				},
			},
		},
	}

	result := transformer.TransformRequest(req)

	parts := result.Contents[0].Parts
	// Should wrap in {"result": ...}
	if parts[0].FunctionResponse.Response["result"] != "plain text result" {
		t.Errorf("expected wrapped result, got %v", parts[0].FunctionResponse.Response)
	}
}

func TestTransformRequest_Tools(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model:    "gemini-2.5-flash",
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

	decl := result.Tools[0].FunctionDeclarations
	if len(decl) != 1 {
		t.Fatalf("expected 1 declaration, got %d", len(decl))
	}

	if decl[0].Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", decl[0].Name)
	}
}

func TestTransformRequest_ToolChoice(t *testing.T) {
	transformer := NewTransformer()

	tests := []struct {
		choice   *types.ToolChoice
		expected string
	}{
		{&types.ToolChoice{Type: types.ToolChoiceAuto}, "AUTO"},
		{&types.ToolChoice{Type: types.ToolChoiceRequired}, "ANY"},
		{&types.ToolChoice{Type: types.ToolChoiceNone}, "NONE"},
	}

	for _, tt := range tests {
		req := &types.CompletionRequest{
			Model:      "gemini-2.5-flash",
			Messages:   []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
			ToolChoice: tt.choice,
		}

		result := transformer.TransformRequest(req)

		if result.ToolConfig == nil {
			t.Fatal("expected ToolConfig to be non-nil")
		}

		if result.ToolConfig.FunctionCallingConfig.Mode != tt.expected {
			t.Errorf("expected mode %q, got %q", tt.expected, result.ToolConfig.FunctionCallingConfig.Mode)
		}
	}
}

func TestTransformRequest_ToolChoiceSpecific(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model:    "gemini-2.5-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
		ToolChoice: &types.ToolChoice{
			Type: types.ToolChoiceTool,
			Name: "get_weather",
		},
	}

	result := transformer.TransformRequest(req)

	config := result.ToolConfig.FunctionCallingConfig
	if config.Mode != "ANY" {
		t.Errorf("expected mode 'ANY', got %q", config.Mode)
	}

	if len(config.AllowedFunctionNames) != 1 || config.AllowedFunctionNames[0] != "get_weather" {
		t.Errorf("expected allowed function 'get_weather', got %v", config.AllowedFunctionNames)
	}
}

func TestTransformRequest_JSONResponseFormat(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model:    "gemini-2.5-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
		ResponseFormat: &types.ResponseFormat{
			Type: "json",
		},
	}

	result := transformer.TransformRequest(req)

	if result.GenerationConfig.ResponseMimeType != "application/json" {
		t.Errorf("expected mime type 'application/json', got %q", result.GenerationConfig.ResponseMimeType)
	}
}

func TestTransformRequest_JSONSchemaResponseFormat(t *testing.T) {
	transformer := NewTransformer()

	req := &types.CompletionRequest{
		Model:    "gemini-2.5-flash",
		Messages: []types.Message{types.NewTextMessage(types.RoleUser, "Hi")},
		ResponseFormat: &types.ResponseFormat{
			Type: "json_schema",
			Schema: &types.JSONSchema{
				Type: "object",
				Properties: map[string]types.JSONSchema{
					"name": {Type: "string"},
				},
			},
		},
	}

	result := transformer.TransformRequest(req)

	if result.GenerationConfig.ResponseMimeType != "application/json" {
		t.Errorf("expected mime type 'application/json', got %q", result.GenerationConfig.ResponseMimeType)
	}

	if result.GenerationConfig.ResponseSchema == nil {
		t.Fatal("expected ResponseSchema to be non-nil")
	}

	if result.GenerationConfig.ResponseSchema.Type != "OBJECT" {
		t.Errorf("expected type 'OBJECT', got %q", result.GenerationConfig.ResponseSchema.Type)
	}
}

func TestTransformResponse(t *testing.T) {
	transformer := NewTransformer()

	resp := &GenerateContentResponse{
		Candidates: []Candidate{
			{
				Content: &Content{
					Role: "model",
					Parts: []Part{
						{Text: "Hello!"},
					},
				},
				FinishReason: "STOP",
			},
		},
		UsageMetadata: &UsageMetadata{
			PromptTokenCount:     10,
			CandidatesTokenCount: 5,
			TotalTokenCount:      15,
		},
	}

	result := transformer.TransformResponse(resp)

	if result.Provider != types.ProviderGoogle {
		t.Errorf("expected provider Google, got %q", result.Provider)
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

func TestTransformResponse_WithToolCalls(t *testing.T) {
	transformer := NewTransformer()

	resp := &GenerateContentResponse{
		Candidates: []Candidate{
			{
				Content: &Content{
					Role: "model",
					Parts: []Part{
						{
							FunctionCall: &FunctionCall{
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

	result := transformer.TransformResponse(resp)

	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
	}

	tc := result.ToolCalls[0]
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

	result = transformer.TransformResponse(&GenerateContentResponse{Candidates: []Candidate{}})
	if result != nil {
		t.Error("expected nil for empty candidates")
	}
}

func TestTransformStopReason(t *testing.T) {
	transformer := NewTransformer()

	tests := []struct {
		reason   string
		expected types.StopReason
	}{
		{"STOP", types.StopReasonEnd},
		{"MAX_TOKENS", types.StopReasonMaxTokens},
		{"SAFETY", types.StopReasonContentFilter},
		{"RECITATION", types.StopReasonContentFilter},
		{"OTHER", types.StopReasonEnd},
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
		{types.RoleAssistant, "model"},
		{types.RoleSystem, "user"}, // System is handled separately
	}

	for _, tt := range tests {
		result := transformer.mapRole(tt.role)
		if result != tt.expected {
			t.Errorf("mapRole(%q) = %q, expected %q", tt.role, result, tt.expected)
		}
	}
}

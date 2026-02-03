package schema

import (
	"encoding/json"
	"testing"

	"github.com/Chloe199719/agent-router/pkg/types"
)

func TestToOpenAI_TextFormat(t *testing.T) {
	translator := NewTranslator()

	rf := &types.ResponseFormat{Type: "text"}
	result := translator.ToOpenAI(rf)

	if result.Type != "text" {
		t.Errorf("expected type 'text', got %q", result.Type)
	}
}

func TestToOpenAI_JSONFormat(t *testing.T) {
	translator := NewTranslator()

	rf := &types.ResponseFormat{Type: "json"}
	result := translator.ToOpenAI(rf)

	if result.Type != "json_object" {
		t.Errorf("expected type 'json_object', got %q", result.Type)
	}
}

func TestToOpenAI_JSONSchemaFormat(t *testing.T) {
	translator := NewTranslator()

	schema := &types.JSONSchema{
		Type: "object",
		Properties: map[string]types.JSONSchema{
			"name": {Type: "string"},
			"age":  {Type: "integer"},
		},
		Required: []string{"name", "age"},
	}

	rf := &types.ResponseFormat{
		Type:   "json_schema",
		Name:   "person",
		Schema: schema,
	}

	result := translator.ToOpenAI(rf)

	if result.Type != "json_schema" {
		t.Errorf("expected type 'json_schema', got %q", result.Type)
	}

	if result.JSONSchema == nil {
		t.Fatal("expected JSONSchema to be non-nil")
	}

	if result.JSONSchema.Name != "person" {
		t.Errorf("expected name 'person', got %q", result.JSONSchema.Name)
	}

	if !result.JSONSchema.Strict {
		t.Error("expected strict to be true by default")
	}

	// Check additionalProperties was added
	if result.JSONSchema.Schema["additionalProperties"] != false {
		t.Error("expected additionalProperties to be false")
	}
}

func TestToOpenAI_NilInput(t *testing.T) {
	translator := NewTranslator()

	result := translator.ToOpenAI(nil)
	if result != nil {
		t.Error("expected nil result for nil input")
	}
}

func TestToOpenAI_StrictModeFalse(t *testing.T) {
	translator := NewTranslator()

	strict := false
	rf := &types.ResponseFormat{
		Type:   "json_schema",
		Name:   "test",
		Schema: &types.JSONSchema{Type: "object"},
		Strict: &strict,
	}

	result := translator.ToOpenAI(rf)

	if result.JSONSchema.Strict {
		t.Error("expected strict to be false when explicitly set")
	}
}

func TestAddAdditionalPropertiesFalse_NestedObjects(t *testing.T) {
	translator := NewTranslator()

	schema := &types.JSONSchema{
		Type: "object",
		Properties: map[string]types.JSONSchema{
			"person": {
				Type: "object",
				Properties: map[string]types.JSONSchema{
					"address": {
						Type: "object",
						Properties: map[string]types.JSONSchema{
							"city": {Type: "string"},
						},
					},
				},
			},
		},
	}

	rf := &types.ResponseFormat{
		Type:   "json_schema",
		Name:   "nested",
		Schema: schema,
	}

	result := translator.ToOpenAI(rf)

	// Check nested additionalProperties
	props := result.JSONSchema.Schema["properties"].(map[string]any)
	person := props["person"].(map[string]any)
	if person["additionalProperties"] != false {
		t.Error("expected nested object to have additionalProperties: false")
	}

	personProps := person["properties"].(map[string]any)
	address := personProps["address"].(map[string]any)
	if address["additionalProperties"] != false {
		t.Error("expected deeply nested object to have additionalProperties: false")
	}
}

func TestAddAdditionalPropertiesFalse_ArrayItems(t *testing.T) {
	translator := NewTranslator()

	schema := &types.JSONSchema{
		Type: "array",
		Items: &types.JSONSchema{
			Type: "object",
			Properties: map[string]types.JSONSchema{
				"name": {Type: "string"},
			},
		},
	}

	rf := &types.ResponseFormat{
		Type:   "json_schema",
		Name:   "array_test",
		Schema: schema,
	}

	result := translator.ToOpenAI(rf)

	items := result.JSONSchema.Schema["items"].(map[string]any)
	if items["additionalProperties"] != false {
		t.Error("expected array items object to have additionalProperties: false")
	}
}

func TestToolsToOpenAI(t *testing.T) {
	translator := NewTranslator()

	tools := []types.Tool{
		{
			Name:        "get_weather",
			Description: "Get the current weather",
			Parameters: types.JSONSchema{
				Type: "object",
				Properties: map[string]types.JSONSchema{
					"location": {Type: "string", Description: "The city"},
					"unit":     {Type: "string", Enum: []any{"celsius", "fahrenheit"}},
				},
				Required: []string{"location"},
			},
		},
	}

	result := translator.ToolsToOpenAI(tools)

	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}

	tool := result[0]
	if tool.Type != "function" {
		t.Errorf("expected type 'function', got %q", tool.Type)
	}

	if tool.Function.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", tool.Function.Name)
	}

	if tool.Function.Strict {
		t.Error("expected strict to be false for tools by default")
	}
}

func TestToolsToOpenAIStrict(t *testing.T) {
	translator := NewTranslator()

	tools := []types.Tool{
		{
			Name: "test_tool",
			Parameters: types.JSONSchema{
				Type:       "object",
				Properties: map[string]types.JSONSchema{"param": {Type: "string"}},
				Required:   []string{"param"},
			},
		},
	}

	result := translator.ToolsToOpenAIStrict(tools)

	if !result[0].Function.Strict {
		t.Error("expected strict to be true for strict mode tools")
	}
}

// Anthropic Tests

func TestToAnthropic_TextFormat(t *testing.T) {
	translator := NewTranslator()

	rf := &types.ResponseFormat{Type: "text"}
	result := translator.ToAnthropic(rf)

	if result != nil {
		t.Error("expected nil for text format")
	}
}

func TestToAnthropic_JSONFormat(t *testing.T) {
	translator := NewTranslator()

	rf := &types.ResponseFormat{Type: "json"}
	result := translator.ToAnthropic(rf)

	// Anthropic doesn't have simple JSON mode
	if result != nil {
		t.Error("expected nil for json format (not supported)")
	}
}

func TestToAnthropic_JSONSchemaFormat(t *testing.T) {
	translator := NewTranslator()

	schema := &types.JSONSchema{
		Type: "object",
		Properties: map[string]types.JSONSchema{
			"name": {Type: "string"},
		},
		Required: []string{"name"},
	}

	rf := &types.ResponseFormat{
		Type:   "json_schema",
		Schema: schema,
	}

	result := translator.ToAnthropic(rf)

	if result == nil {
		t.Fatal("expected non-nil result")
	}

	if result.Format == nil {
		t.Fatal("expected Format to be non-nil")
	}

	if result.Format.Type != "json_schema" {
		t.Errorf("expected type 'json_schema', got %q", result.Format.Type)
	}

	// Check additionalProperties was added
	if result.Format.Schema["additionalProperties"] != false {
		t.Error("expected additionalProperties to be false")
	}
}

func TestToolsToAnthropic(t *testing.T) {
	translator := NewTranslator()

	tools := []types.Tool{
		{
			Name:        "search",
			Description: "Search the web",
			Parameters: types.JSONSchema{
				Type: "object",
				Properties: map[string]types.JSONSchema{
					"query": {Type: "string"},
				},
				Required: []string{"query"},
			},
		},
	}

	result := translator.ToolsToAnthropic(tools)

	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}

	if result[0].Name != "search" {
		t.Errorf("expected name 'search', got %q", result[0].Name)
	}

	if result[0].InputSchema == nil {
		t.Error("expected InputSchema to be non-nil")
	}
}

// Google Tests

func TestToGoogle_TextFormat(t *testing.T) {
	translator := NewTranslator()

	rf := &types.ResponseFormat{Type: "text"}
	result := translator.ToGoogle(rf)

	if result != nil {
		t.Error("expected nil for text format")
	}
}

func TestToGoogle_JSONFormat(t *testing.T) {
	translator := NewTranslator()

	rf := &types.ResponseFormat{Type: "json"}
	result := translator.ToGoogle(rf)

	if result == nil {
		t.Fatal("expected non-nil result")
	}

	if result.ResponseMimeType != "application/json" {
		t.Errorf("expected mime type 'application/json', got %q", result.ResponseMimeType)
	}
}

func TestToGoogle_JSONSchemaFormat(t *testing.T) {
	translator := NewTranslator()

	schema := &types.JSONSchema{
		Type: "object",
		Properties: map[string]types.JSONSchema{
			"name": {Type: "string"},
			"age":  {Type: "integer"},
		},
		Required: []string{"name"},
	}

	rf := &types.ResponseFormat{
		Type:   "json_schema",
		Schema: schema,
	}

	result := translator.ToGoogle(rf)

	if result == nil {
		t.Fatal("expected non-nil result")
	}

	if result.ResponseMimeType != "application/json" {
		t.Errorf("expected mime type 'application/json', got %q", result.ResponseMimeType)
	}

	if result.ResponseSchema == nil {
		t.Fatal("expected ResponseSchema to be non-nil")
	}

	// Check type was converted to uppercase
	if result.ResponseSchema.Type != "OBJECT" {
		t.Errorf("expected type 'OBJECT', got %q", result.ResponseSchema.Type)
	}
}

func TestMapTypeToGoogle(t *testing.T) {
	translator := NewTranslator()

	tests := []struct {
		jsonType string
		expected string
	}{
		{"string", "STRING"},
		{"integer", "INTEGER"},
		{"number", "NUMBER"},
		{"boolean", "BOOLEAN"},
		{"array", "ARRAY"},
		{"object", "OBJECT"},
		{"unknown", "STRING"}, // Default
	}

	for _, tt := range tests {
		result := translator.mapTypeToGoogle(tt.jsonType)
		if result != tt.expected {
			t.Errorf("mapTypeToGoogle(%q) = %q, expected %q", tt.jsonType, result, tt.expected)
		}
	}
}

func TestConvertToGoogleSchema_Nested(t *testing.T) {
	translator := NewTranslator()

	schema := &types.JSONSchema{
		Type: "object",
		Properties: map[string]types.JSONSchema{
			"items": {
				Type: "array",
				Items: &types.JSONSchema{
					Type: "object",
					Properties: map[string]types.JSONSchema{
						"id":   {Type: "integer"},
						"name": {Type: "string"},
					},
				},
			},
		},
	}

	result := translator.convertToGoogleSchema(schema)

	if result.Type != "OBJECT" {
		t.Errorf("expected type 'OBJECT', got %q", result.Type)
	}

	items := result.Properties["items"]
	if items.Type != "ARRAY" {
		t.Errorf("expected items type 'ARRAY', got %q", items.Type)
	}

	if items.Items == nil {
		t.Fatal("expected items.Items to be non-nil")
	}

	if items.Items.Type != "OBJECT" {
		t.Errorf("expected items.Items.Type 'OBJECT', got %q", items.Items.Type)
	}
}

func TestConvertToGoogleSchema_Enum(t *testing.T) {
	translator := NewTranslator()

	schema := &types.JSONSchema{
		Type: "string",
		Enum: []any{"red", "green", "blue"},
	}

	result := translator.convertToGoogleSchema(schema)

	if len(result.Enum) != 3 {
		t.Fatalf("expected 3 enum values, got %d", len(result.Enum))
	}

	if result.Enum[0] != "red" {
		t.Errorf("expected first enum 'red', got %q", result.Enum[0])
	}
}

func TestToolsToGoogle(t *testing.T) {
	translator := NewTranslator()

	tools := []types.Tool{
		{
			Name:        "get_location",
			Description: "Get user location",
			Parameters: types.JSONSchema{
				Type:       "object",
				Properties: map[string]types.JSONSchema{},
			},
		},
	}

	result := translator.ToolsToGoogle(tools)

	if result == nil {
		t.Fatal("expected non-nil result")
	}

	if len(result.FunctionDeclarations) != 1 {
		t.Fatalf("expected 1 declaration, got %d", len(result.FunctionDeclarations))
	}

	if result.FunctionDeclarations[0].Name != "get_location" {
		t.Errorf("expected name 'get_location', got %q", result.FunctionDeclarations[0].Name)
	}
}

func TestToolsToGoogle_Empty(t *testing.T) {
	translator := NewTranslator()

	result := translator.ToolsToGoogle(nil)
	if result != nil {
		t.Error("expected nil for empty tools")
	}

	result = translator.ToolsToGoogle([]types.Tool{})
	if result != nil {
		t.Error("expected nil for empty tools slice")
	}
}

// Helper function to pretty-print for debugging
func toJSON(v any) string {
	b, _ := json.MarshalIndent(v, "", "  ")
	return string(b)
}

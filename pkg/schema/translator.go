// Package schema provides JSON Schema translation between providers.
package schema

import (
	"encoding/json"

	"github.com/Chloe199719/agent-router/pkg/types"
)

// Translator converts unified JSONSchema to provider-specific formats.
type Translator struct{}

// NewTranslator creates a new schema translator.
func NewTranslator() *Translator {
	return &Translator{}
}

// ----- OpenAI Format -----

// OpenAIResponseFormat is OpenAI's response_format structure.
type OpenAIResponseFormat struct {
	Type       string            `json:"type"`
	JSONSchema *OpenAIJSONSchema `json:"json_schema,omitempty"`
}

// OpenAIJSONSchema is OpenAI's JSON schema wrapper.
type OpenAIJSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
	Strict      bool           `json:"strict"`
}

// ToOpenAI converts unified schema to OpenAI format.
func (t *Translator) ToOpenAI(rf *types.ResponseFormat) *OpenAIResponseFormat {
	if rf == nil {
		return nil
	}

	switch rf.Type {
	case "json":
		return &OpenAIResponseFormat{Type: "json_object"}
	case "json_schema":
		schema := t.prepareOpenAISchema(rf.Schema)
		strict := true
		if rf.Strict != nil {
			strict = *rf.Strict
		}
		return &OpenAIResponseFormat{
			Type: "json_schema",
			JSONSchema: &OpenAIJSONSchema{
				Name:        rf.Name,
				Description: rf.Description,
				Schema:      schema,
				Strict:      strict,
			},
		}
	default:
		return &OpenAIResponseFormat{Type: "text"}
	}
}

// prepareOpenAISchema adds required OpenAI constraints.
func (t *Translator) prepareOpenAISchema(s *types.JSONSchema) map[string]any {
	if s == nil {
		return nil
	}

	// Convert to map and add OpenAI-specific requirements
	schema := s.ToMap()

	// OpenAI strict mode requires additionalProperties: false on all objects
	t.addAdditionalPropertiesFalse(schema)

	return schema
}

// addAdditionalPropertiesFalse recursively adds additionalProperties: false to all objects.
func (t *Translator) addAdditionalPropertiesFalse(schema map[string]any) {
	if schema == nil {
		return
	}

	schemaType, _ := schema["type"].(string)
	if schemaType == "object" {
		schema["additionalProperties"] = false
	}

	// Recurse into properties
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, prop := range props {
			if propMap, ok := prop.(map[string]any); ok {
				t.addAdditionalPropertiesFalse(propMap)
			}
		}
	}

	// Recurse into items (arrays)
	if items, ok := schema["items"].(map[string]any); ok {
		t.addAdditionalPropertiesFalse(items)
	}

	// Recurse into anyOf, oneOf, allOf
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if arr, ok := schema[key].([]any); ok {
			for _, item := range arr {
				if itemMap, ok := item.(map[string]any); ok {
					t.addAdditionalPropertiesFalse(itemMap)
				}
			}
		}
	}

	// Recurse into $defs
	if defs, ok := schema["$defs"].(map[string]any); ok {
		for _, def := range defs {
			if defMap, ok := def.(map[string]any); ok {
				t.addAdditionalPropertiesFalse(defMap)
			}
		}
	}
}

// OpenAITool is OpenAI's tool format.
type OpenAITool struct {
	Type     string             `json:"type"`
	Function OpenAIFunctionTool `json:"function"`
}

// OpenAIFunctionTool is OpenAI's function tool definition.
type OpenAIFunctionTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict,omitempty"`
}

// ToolsToOpenAI converts unified tools to OpenAI format.
// Note: Strict mode is NOT enabled by default for tools because it requires
// all properties to be in the required array (no optional parameters allowed).
// If you need strict mode, ensure all properties are marked as required.
func (t *Translator) ToolsToOpenAI(tools []types.Tool) []OpenAITool {
	result := make([]OpenAITool, len(tools))
	for i, tool := range tools {
		params := tool.Parameters.ToMap()
		// Add additionalProperties: false for better schema validation
		if params != nil {
			t.addAdditionalPropertiesFalse(params)
		}
		result[i] = OpenAITool{
			Type: "function",
			Function: OpenAIFunctionTool{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  params,
				Strict:      false, // Allow optional parameters
			},
		}
	}
	return result
}

// ToolsToOpenAIStrict converts unified tools to OpenAI format with strict mode.
// In strict mode, ALL properties must be listed in the required array.
func (t *Translator) ToolsToOpenAIStrict(tools []types.Tool) []OpenAITool {
	result := make([]OpenAITool, len(tools))
	for i, tool := range tools {
		params := t.prepareOpenAISchema(&tool.Parameters)
		result[i] = OpenAITool{
			Type: "function",
			Function: OpenAIFunctionTool{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  params,
				Strict:      true,
			},
		}
	}
	return result
}

// ----- Anthropic Format -----

// AnthropicOutputConfig is Anthropic's output configuration.
type AnthropicOutputConfig struct {
	Format *AnthropicFormat `json:"format,omitempty"`
}

// AnthropicFormat is Anthropic's format configuration.
type AnthropicFormat struct {
	Type   string         `json:"type"`
	Schema map[string]any `json:"schema,omitempty"`
}

// ToAnthropic converts unified schema to Anthropic format.
func (t *Translator) ToAnthropic(rf *types.ResponseFormat) *AnthropicOutputConfig {
	if rf == nil || rf.Type == "text" {
		return nil
	}

	if rf.Type == "json" {
		// Anthropic doesn't have a simple JSON mode like OpenAI
		// We'd need to handle this differently, perhaps with system prompt
		return nil
	}

	if rf.Type == "json_schema" && rf.Schema != nil {
		schema := rf.Schema.ToMap()
		// Anthropic requires additionalProperties: false on all objects
		t.addAdditionalPropertiesFalse(schema)
		return &AnthropicOutputConfig{
			Format: &AnthropicFormat{
				Type:   "json_schema",
				Schema: schema,
			},
		}
	}

	return nil
}

// AnthropicTool is Anthropic's tool format.
type AnthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

// ToolsToAnthropic converts unified tools to Anthropic format.
func (t *Translator) ToolsToAnthropic(tools []types.Tool) []AnthropicTool {
	result := make([]AnthropicTool, len(tools))
	for i, tool := range tools {
		result[i] = AnthropicTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.Parameters.ToMap(),
		}
	}
	return result
}

// ----- Google/Gemini Format -----

// GoogleGenerationConfig is Google's generation configuration.
type GoogleGenerationConfig struct {
	ResponseMimeType string        `json:"responseMimeType,omitempty"`
	ResponseSchema   *GoogleSchema `json:"responseSchema,omitempty"`
	Temperature      *float64      `json:"temperature,omitempty"`
	TopP             *float64      `json:"topP,omitempty"`
	TopK             *int          `json:"topK,omitempty"`
	MaxOutputTokens  *int          `json:"maxOutputTokens,omitempty"`
	StopSequences    []string      `json:"stopSequences,omitempty"`
}

// GoogleSchema is Google's schema format (differs from standard JSON Schema).
type GoogleSchema struct {
	Type        string                   `json:"type"`
	Description string                   `json:"description,omitempty"`
	Enum        []string                 `json:"enum,omitempty"`
	Properties  map[string]*GoogleSchema `json:"properties,omitempty"`
	Required    []string                 `json:"required,omitempty"`
	Items       *GoogleSchema            `json:"items,omitempty"`
	Nullable    bool                     `json:"nullable,omitempty"`
}

// ToGoogle converts unified schema to Google format.
func (t *Translator) ToGoogle(rf *types.ResponseFormat) *GoogleGenerationConfig {
	if rf == nil || rf.Type == "text" {
		return nil
	}

	config := &GoogleGenerationConfig{}

	if rf.Type == "json" {
		config.ResponseMimeType = "application/json"
		return config
	}

	if rf.Type == "json_schema" && rf.Schema != nil {
		config.ResponseMimeType = "application/json"
		config.ResponseSchema = t.convertToGoogleSchema(rf.Schema)
		return config
	}

	return nil
}

// convertToGoogleSchema converts JSON Schema to Google's schema format.
func (t *Translator) convertToGoogleSchema(s *types.JSONSchema) *GoogleSchema {
	if s == nil {
		return nil
	}

	gs := &GoogleSchema{
		Type:        t.mapTypeToGoogle(s.Type),
		Description: s.Description,
		Required:    s.Required,
	}

	// Convert enum (Google only supports string enums)
	if len(s.Enum) > 0 {
		gs.Enum = make([]string, len(s.Enum))
		for i, v := range s.Enum {
			gs.Enum[i] = toString(v)
		}
	}

	// Convert properties
	if len(s.Properties) > 0 {
		gs.Properties = make(map[string]*GoogleSchema)
		for name, prop := range s.Properties {
			gs.Properties[name] = t.convertToGoogleSchema(&prop)
		}
	}

	// Convert items (arrays)
	if s.Items != nil {
		gs.Items = t.convertToGoogleSchema(s.Items)
	}

	return gs
}

// mapTypeToGoogle maps JSON Schema types to Google types.
func (t *Translator) mapTypeToGoogle(jsonType string) string {
	switch jsonType {
	case "integer":
		return "INTEGER"
	case "number":
		return "NUMBER"
	case "string":
		return "STRING"
	case "boolean":
		return "BOOLEAN"
	case "array":
		return "ARRAY"
	case "object":
		return "OBJECT"
	default:
		return "STRING"
	}
}

// GoogleTool is Google's tool format.
type GoogleTool struct {
	FunctionDeclarations []GoogleFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

// GoogleFunctionDeclaration is Google's function declaration format.
type GoogleFunctionDeclaration struct {
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	Parameters  *GoogleSchema `json:"parameters,omitempty"`
}

// ToolsToGoogle converts unified tools to Google format.
func (t *Translator) ToolsToGoogle(tools []types.Tool) *GoogleTool {
	if len(tools) == 0 {
		return nil
	}

	declarations := make([]GoogleFunctionDeclaration, len(tools))
	for i, tool := range tools {
		declarations[i] = GoogleFunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  t.convertToGoogleSchema(&tool.Parameters),
		}
	}

	return &GoogleTool{FunctionDeclarations: declarations}
}

// Helper to convert any value to string
func toString(v any) string {
	switch val := v.(type) {
	case string:
		return val
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

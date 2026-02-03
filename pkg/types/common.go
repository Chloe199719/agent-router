// Package types provides unified types for multi-provider LLM inference.
package types

import "encoding/json"

// Provider represents supported LLM providers.
type Provider string

const (
	ProviderOpenAI    Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
	ProviderGoogle    Provider = "google"
)

// Role represents message roles in a conversation.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentType represents the type of content in a content block.
type ContentType string

const (
	ContentTypeText       ContentType = "text"
	ContentTypeImage      ContentType = "image"
	ContentTypeToolUse    ContentType = "tool_use"
	ContentTypeToolResult ContentType = "tool_result"
)

// ContentBlock represents a piece of content (text, image, tool use, etc.).
type ContentBlock struct {
	Type ContentType `json:"type"`

	// For text content
	Text string `json:"text,omitempty"`

	// For image content
	ImageURL    string `json:"image_url,omitempty"`
	ImageBase64 string `json:"image_base64,omitempty"`
	MediaType   string `json:"media_type,omitempty"` // e.g., "image/png", "image/jpeg"

	// For tool use (assistant calling a tool)
	ToolUseID string `json:"tool_use_id,omitempty"`
	ToolName  string `json:"tool_name,omitempty"`
	ToolInput any    `json:"tool_input,omitempty"`

	// For tool result (user providing tool output)
	ToolResultID string `json:"tool_result_id,omitempty"`
	IsError      bool   `json:"is_error,omitempty"`
}

// Message represents a conversation message.
type Message struct {
	Role    Role           `json:"role"`
	Content []ContentBlock `json:"content"`
}

// NewTextMessage creates a simple text message.
func NewTextMessage(role Role, text string) Message {
	return Message{
		Role: role,
		Content: []ContentBlock{
			{Type: ContentTypeText, Text: text},
		},
	}
}

// NewToolResultMessage creates a tool result message.
func NewToolResultMessage(toolUseID string, result string, isError bool) Message {
	return Message{
		Role: RoleTool,
		Content: []ContentBlock{
			{
				Type:         ContentTypeToolResult,
				ToolResultID: toolUseID,
				Text:         result,
				IsError:      isError,
			},
		},
	}
}

// Tool represents a function/tool that the model can use.
type Tool struct {
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	Parameters  JSONSchema `json:"parameters"`
}

// ToolCall represents a tool invocation by the model.
type ToolCall struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Input any    `json:"input"`
}

// JSONSchema represents a JSON Schema definition.
// This is our unified schema format that gets translated to provider-specific formats.
type JSONSchema struct {
	Type                 string                `json:"type,omitempty"`
	Description          string                `json:"description,omitempty"`
	Properties           map[string]JSONSchema `json:"properties,omitempty"`
	Items                *JSONSchema           `json:"items,omitempty"`
	Required             []string              `json:"required,omitempty"`
	Enum                 []any                 `json:"enum,omitempty"`
	Const                any                   `json:"const,omitempty"`
	AdditionalProperties *bool                 `json:"additionalProperties,omitempty"`
	MinItems             *int                  `json:"minItems,omitempty"`
	MaxItems             *int                  `json:"maxItems,omitempty"`
	Minimum              *float64              `json:"minimum,omitempty"`
	Maximum              *float64              `json:"maximum,omitempty"`
	MinLength            *int                  `json:"minLength,omitempty"`
	MaxLength            *int                  `json:"maxLength,omitempty"`
	Pattern              string                `json:"pattern,omitempty"`
	Format               string                `json:"format,omitempty"`
	Default              any                   `json:"default,omitempty"`
	AnyOf                []JSONSchema          `json:"anyOf,omitempty"`
	OneOf                []JSONSchema          `json:"oneOf,omitempty"`
	AllOf                []JSONSchema          `json:"allOf,omitempty"`
	Ref                  string                `json:"$ref,omitempty"`
	Defs                 map[string]JSONSchema `json:"$defs,omitempty"`
}

// ToMap converts JSONSchema to a map for JSON marshaling.
func (s JSONSchema) ToMap() map[string]any {
	data, _ := json.Marshal(s)
	var m map[string]any
	json.Unmarshal(data, &m)
	return m
}

// StopReason represents why generation stopped.
type StopReason string

const (
	StopReasonEnd           StopReason = "end"
	StopReasonMaxTokens     StopReason = "max_tokens"
	StopReasonToolUse       StopReason = "tool_use"
	StopReasonStopSequence  StopReason = "stop_sequence"
	StopReasonContentFilter StopReason = "content_filter"
)

// Usage represents token usage information.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`

	// Provider-specific details (optional)
	CachedTokens    int `json:"cached_tokens,omitempty"`
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// Feature represents provider capabilities.
type Feature string

const (
	FeatureStreaming        Feature = "streaming"
	FeatureStructuredOutput Feature = "structured_output"
	FeatureTools            Feature = "tools"
	FeatureVision           Feature = "vision"
	FeatureBatch            Feature = "batch"
	FeatureJSON             Feature = "json_mode"
)

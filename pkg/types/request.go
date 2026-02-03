package types

// CompletionRequest is the unified request format for all providers.
type CompletionRequest struct {
	// Provider to use for this request
	Provider Provider `json:"provider"`

	// Model identifier (provider-specific, e.g., "gpt-4o", "claude-sonnet-4-20250514", "gemini-pro")
	Model string `json:"model"`

	// Messages in the conversation
	Messages []Message `json:"messages"`

	// Generation parameters
	MaxTokens     *int     `json:"max_tokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"top_p,omitempty"`
	TopK          *int     `json:"top_k,omitempty"` // Anthropic/Google only
	StopSequences []string `json:"stop_sequences,omitempty"`

	// Structured output configuration
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// Tool/function calling
	Tools      []Tool      `json:"tools,omitempty"`
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`

	// Streaming
	Stream bool `json:"stream,omitempty"`

	// Provider-specific options (passed through without modification)
	Extra map[string]any `json:"extra,omitempty"`
}

// ResponseFormat configures structured output.
type ResponseFormat struct {
	// Type of response format: "text", "json", or "json_schema"
	Type string `json:"type"`

	// Schema for structured output (when Type is "json_schema")
	Schema *JSONSchema `json:"schema,omitempty"`

	// Name for the schema (required by some providers)
	Name string `json:"name,omitempty"`

	// Description of what the schema represents
	Description string `json:"description,omitempty"`

	// Strict mode - ensures output exactly matches schema (OpenAI)
	Strict *bool `json:"strict,omitempty"`
}

// ToolChoiceType represents how the model should use tools.
type ToolChoiceType string

const (
	ToolChoiceAuto     ToolChoiceType = "auto"     // Model decides whether to use tools
	ToolChoiceRequired ToolChoiceType = "required" // Model must use at least one tool
	ToolChoiceNone     ToolChoiceType = "none"     // Model cannot use tools
	ToolChoiceTool     ToolChoiceType = "tool"     // Model must use a specific tool
)

// ToolChoice controls how the model uses tools.
type ToolChoice struct {
	// Type of tool choice
	Type ToolChoiceType `json:"type"`

	// Name of specific tool (when Type is "tool")
	Name string `json:"name,omitempty"`

	// DisableParallelToolUse prevents multiple tool calls in one response (Anthropic)
	DisableParallelToolUse bool `json:"disable_parallel_tool_use,omitempty"`
}

// Ptr helpers for creating pointers to primitives.
func Ptr[T any](v T) *T {
	return &v
}

// WithMaxTokens sets max tokens on a request.
func (r *CompletionRequest) WithMaxTokens(n int) *CompletionRequest {
	r.MaxTokens = &n
	return r
}

// WithTemperature sets temperature on a request.
func (r *CompletionRequest) WithTemperature(t float64) *CompletionRequest {
	r.Temperature = &t
	return r
}

// WithTools adds tools to a request.
func (r *CompletionRequest) WithTools(tools ...Tool) *CompletionRequest {
	r.Tools = append(r.Tools, tools...)
	return r
}

// WithJSONSchema adds a JSON schema response format.
func (r *CompletionRequest) WithJSONSchema(name string, schema JSONSchema) *CompletionRequest {
	strict := true
	r.ResponseFormat = &ResponseFormat{
		Type:   "json_schema",
		Name:   name,
		Schema: &schema,
		Strict: &strict,
	}
	return r
}

// WithStream enables streaming.
func (r *CompletionRequest) WithStream() *CompletionRequest {
	r.Stream = true
	return r
}

package anthropic

// MessagesRequest is the Anthropic messages API request.
type MessagesRequest struct {
	Model         string        `json:"model"`
	Messages      []Message     `json:"messages"`
	MaxTokens     int           `json:"max_tokens"`
	System        any           `json:"system,omitempty"` // string or []SystemBlock
	Temperature   *float64      `json:"temperature,omitempty"`
	TopP          *float64      `json:"top_p,omitempty"`
	TopK          *int          `json:"top_k,omitempty"`
	StopSequences []string      `json:"stop_sequences,omitempty"`
	Stream        bool          `json:"stream,omitempty"`
	Tools         []Tool        `json:"tools,omitempty"`
	ToolChoice    *ToolChoice   `json:"tool_choice,omitempty"`
	Metadata      *Metadata     `json:"metadata,omitempty"`
	OutputConfig  *OutputConfig `json:"output_config,omitempty"`
}

// Message is an Anthropic message.
type Message struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or []ContentBlock
}

// ContentBlock is a content block in a message.
type ContentBlock struct {
	Type string `json:"type"`

	// For text blocks
	Text string `json:"text,omitempty"`

	// For image blocks
	Source *ImageSource `json:"source,omitempty"`

	// For tool_use blocks
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"`

	// For tool_result blocks
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"` // string or []ContentBlock
	IsError   bool   `json:"is_error,omitempty"`
}

// ImageSource is the source of an image.
type ImageSource struct {
	Type      string `json:"type"` // "base64" or "url"
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// SystemBlock is a system message block (for multi-part system prompts).
type SystemBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text,omitempty"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// CacheControl is for prompt caching.
type CacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

// Tool is an Anthropic tool definition.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

// ToolChoice controls tool usage.
type ToolChoice struct {
	Type                   string `json:"type"` // "auto", "any", "tool", "none"
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}

// Metadata is request metadata.
type Metadata struct {
	UserID string `json:"user_id,omitempty"`
}

// OutputConfig configures output format.
type OutputConfig struct {
	Format *OutputFormat `json:"format,omitempty"`
}

// OutputFormat specifies the output format.
type OutputFormat struct {
	Type   string         `json:"type"` // "json_schema"
	Schema map[string]any `json:"schema,omitempty"`
}

// MessagesResponse is the Anthropic messages API response.
type MessagesResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []ContentBlock `json:"content"`
	Model        string         `json:"model"`
	StopReason   string         `json:"stop_reason"`
	StopSequence string         `json:"stop_sequence,omitempty"`
	Usage        Usage          `json:"usage"`
}

// Usage is token usage information.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// StreamEvent is a streaming event.
type StreamEvent struct {
	Type         string            `json:"type"`
	Message      *MessagesResponse `json:"message,omitempty"`
	Index        int               `json:"index,omitempty"`
	ContentBlock *ContentBlock     `json:"content_block,omitempty"`
	Delta        *Delta            `json:"delta,omitempty"`
	Usage        *Usage            `json:"usage,omitempty"`
}

// Delta is a streaming delta.
type Delta struct {
	Type         string `json:"type,omitempty"`
	Text         string `json:"text,omitempty"`
	PartialJSON  string `json:"partial_json,omitempty"`
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
}

// ErrorResponse is an Anthropic error response.
type ErrorResponse struct {
	Type  string    `json:"type"`
	Error *APIError `json:"error"`
}

// APIError is an Anthropic API error.
type APIError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// Batch types for Message Batches API

// BatchRequest is a request to create a message batch.
type BatchRequest struct {
	Requests []BatchRequestItem `json:"requests"`
}

// BatchRequestItem is a single request in a batch.
type BatchRequestItem struct {
	CustomID string          `json:"custom_id"`
	Params   MessagesRequest `json:"params"`
}

// BatchResponse is the response from creating/getting a batch.
type BatchResponse struct {
	ID                string        `json:"id"`
	Type              string        `json:"type"`
	ProcessingStatus  string        `json:"processing_status"`
	RequestCounts     RequestCounts `json:"request_counts"`
	EndedAt           string        `json:"ended_at,omitempty"`
	CreatedAt         string        `json:"created_at"`
	ExpiresAt         string        `json:"expires_at,omitempty"`
	ArchivedAt        string        `json:"archived_at,omitempty"`
	CancelInitiatedAt string        `json:"cancel_initiated_at,omitempty"`
	ResultsURL        string        `json:"results_url,omitempty"`
}

// RequestCounts is the count of requests in various states.
type RequestCounts struct {
	Processing int `json:"processing"`
	Succeeded  int `json:"succeeded"`
	Errored    int `json:"errored"`
	Canceled   int `json:"canceled"`
	Expired    int `json:"expired"`
}

// BatchResultItem is a single result from a batch.
type BatchResultItem struct {
	CustomID string          `json:"custom_id"`
	Result   BatchItemResult `json:"result"`
}

// BatchItemResult is the result of a batch item.
type BatchItemResult struct {
	Type    string            `json:"type"` // "succeeded" or "errored"
	Message *MessagesResponse `json:"message,omitempty"`
	Error   *APIError         `json:"error,omitempty"`
}

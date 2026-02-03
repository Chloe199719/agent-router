package types

import "time"

// CompletionResponse is the unified response format from all providers.
type CompletionResponse struct {
	// Unique identifier for this completion
	ID string `json:"id"`

	// Provider that generated this response
	Provider Provider `json:"provider"`

	// Model that generated this response
	Model string `json:"model"`

	// Generated content
	Content []ContentBlock `json:"content"`

	// Why generation stopped
	StopReason StopReason `json:"stop_reason"`

	// Token usage information
	Usage Usage `json:"usage"`

	// Tool calls made by the model (convenience accessor, also in Content)
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// Timestamp when response was created
	CreatedAt time.Time `json:"created_at,omitempty"`

	// Provider-specific metadata
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Text returns the concatenated text content from the response.
func (r *CompletionResponse) Text() string {
	var text string
	for _, block := range r.Content {
		if block.Type == ContentTypeText {
			text += block.Text
		}
	}
	return text
}

// HasToolCalls returns true if the response contains tool calls.
func (r *CompletionResponse) HasToolCalls() bool {
	return len(r.ToolCalls) > 0
}

// StreamEventType represents the type of streaming event.
type StreamEventType string

const (
	StreamEventStart         StreamEventType = "start"           // Stream started
	StreamEventContentDelta  StreamEventType = "content_delta"   // Text content chunk
	StreamEventToolCallStart StreamEventType = "tool_call_start" // Tool call started
	StreamEventToolCallDelta StreamEventType = "tool_call_delta" // Tool call input chunk
	StreamEventToolCallEnd   StreamEventType = "tool_call_end"   // Tool call finished
	StreamEventDone          StreamEventType = "done"            // Stream completed
	StreamEventError         StreamEventType = "error"           // Error occurred
)

// StreamEvent represents a single event in a streaming response.
type StreamEvent struct {
	// Type of this event
	Type StreamEventType `json:"type"`

	// Content delta (for content_delta events)
	Delta *ContentBlock `json:"delta,omitempty"`

	// Index of the content block being updated
	Index int `json:"index,omitempty"`

	// Tool call information (for tool_call_* events)
	ToolCall *ToolCall `json:"tool_call,omitempty"`

	// Partial tool input JSON (for tool_call_delta)
	ToolInputDelta string `json:"tool_input_delta,omitempty"`

	// Error information (for error events)
	Error error `json:"error,omitempty"`

	// Final usage stats (for done events)
	Usage *Usage `json:"usage,omitempty"`

	// Stop reason (for done events)
	StopReason StopReason `json:"stop_reason,omitempty"`

	// Response ID (for start/done events)
	ResponseID string `json:"response_id,omitempty"`

	// Model (for start events)
	Model string `json:"model,omitempty"`
}

// StreamReader provides a way to read streaming events.
type StreamReader interface {
	// Next returns the next event, or an error if the stream is done or failed.
	// Returns nil, nil when the stream is complete.
	Next() (*StreamEvent, error)

	// Close closes the stream.
	Close() error

	// Response returns the accumulated response after the stream is done.
	// Returns nil if called before the stream is complete.
	Response() *CompletionResponse
}

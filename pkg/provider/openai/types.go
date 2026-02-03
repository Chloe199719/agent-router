package openai

// ChatCompletionRequest is the OpenAI chat completion request.
type ChatCompletionRequest struct {
	Model             string          `json:"model"`
	Messages          []ChatMessage   `json:"messages"`
	MaxTokens         *int            `json:"max_completion_tokens,omitempty"`
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
	N                 *int            `json:"n,omitempty"`
	Stream            bool            `json:"stream,omitempty"`
	StreamOptions     *StreamOptions  `json:"stream_options,omitempty"`
	Stop              []string        `json:"stop,omitempty"`
	PresencePenalty   *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty  *float64        `json:"frequency_penalty,omitempty"`
	LogitBias         map[string]int  `json:"logit_bias,omitempty"`
	User              string          `json:"user,omitempty"`
	ResponseFormat    *ResponseFormat `json:"response_format,omitempty"`
	Tools             []Tool          `json:"tools,omitempty"`
	ToolChoice        any             `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	Seed              *int            `json:"seed,omitempty"`
}

// StreamOptions configures streaming behavior.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ChatMessage is an OpenAI chat message.
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"` // string or []ContentPart
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ContentPart is a content part in a message.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL is an image URL in a message.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// ResponseFormat configures the response format.
type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// JSONSchema is OpenAI's JSON schema format.
type JSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
	Strict      bool           `json:"strict"`
}

// Tool is an OpenAI tool definition.
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function is an OpenAI function definition.
type Function struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict,omitempty"`
}

// ToolCall is an OpenAI tool call.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
	Index    *int         `json:"index,omitempty"` // For streaming
}

// FunctionCall is the function call details.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolChoiceObject is used when specifying a specific tool.
type ToolChoiceObject struct {
	Type     string              `json:"type"`
	Function *ToolChoiceFunction `json:"function,omitempty"`
}

// ToolChoiceFunction specifies which function to call.
type ToolChoiceFunction struct {
	Name string `json:"name"`
}

// ChatCompletionResponse is the OpenAI chat completion response.
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// Choice is a completion choice.
type Choice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
	Logprobs     any         `json:"logprobs,omitempty"`
}

// Usage is token usage information.
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// PromptTokensDetails contains details about prompt tokens.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
	AudioTokens  int `json:"audio_tokens,omitempty"`
}

// CompletionTokensDetails contains details about completion tokens.
type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	AudioTokens     int `json:"audio_tokens,omitempty"`
}

// StreamChunk is a streaming response chunk.
type StreamChunk struct {
	ID                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	Choices           []StreamChoice `json:"choices"`
	Usage             *Usage         `json:"usage,omitempty"`
	SystemFingerprint string         `json:"system_fingerprint,omitempty"`
}

// StreamChoice is a streaming choice.
type StreamChoice struct {
	Index        int          `json:"index"`
	Delta        MessageDelta `json:"delta"`
	FinishReason string       `json:"finish_reason,omitempty"`
	Logprobs     any          `json:"logprobs,omitempty"`
}

// MessageDelta is the delta in a streaming message.
type MessageDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ErrorResponse is an OpenAI error response.
type ErrorResponse struct {
	Error *APIError `json:"error"`
}

// APIError is an OpenAI API error.
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

// Package openai provides an OpenAI API client implementation.
package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Chloe199719/agent-router/pkg/errors"
	"github.com/Chloe199719/agent-router/pkg/provider"
	"github.com/Chloe199719/agent-router/pkg/types"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
)

// Client is an OpenAI API client.
type Client struct {
	config      *provider.Config
	httpClient  *http.Client
	baseURL     string
	transformer *Transformer
}

// New creates a new OpenAI client.
func New(opts ...provider.Option) *Client {
	cfg := provider.DefaultConfig()
	provider.ApplyOptions(cfg, opts...)

	baseURL := defaultBaseURL
	if cfg.BaseURL != "" {
		baseURL = cfg.BaseURL
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: time.Duration(cfg.Timeout) * time.Second,
		}
	}

	return &Client{
		config:      cfg,
		httpClient:  httpClient,
		baseURL:     baseURL,
		transformer: NewTransformer(),
	}
}

// Name returns the provider name.
func (c *Client) Name() types.Provider {
	return types.ProviderOpenAI
}

// SupportsFeature checks if OpenAI supports a feature.
func (c *Client) SupportsFeature(feature types.Feature) bool {
	switch feature {
	case types.FeatureStreaming,
		types.FeatureStructuredOutput,
		types.FeatureTools,
		types.FeatureVision,
		types.FeatureBatch,
		types.FeatureJSON:
		return true
	default:
		return false
	}
}

// Models returns available OpenAI models.
func (c *Client) Models() []string {
	return []string{
		"gpt-4o",
		"gpt-4o-mini",
		"gpt-4-turbo",
		"gpt-4",
		"gpt-3.5-turbo",
		"o1",
		"o1-mini",
		"o1-preview",
	}
}

// Complete sends a completion request.
func (c *Client) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	oaiReq := c.transformer.TransformRequest(req)
	oaiReq.Stream = false

	body, err := json.Marshal(oaiReq)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal request").WithCause(err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.ErrProviderUnavailable(types.ProviderOpenAI, "request failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.handleErrorResponse(resp)
	}

	var oaiResp ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&oaiResp); err != nil {
		return nil, errors.ErrServerError(types.ProviderOpenAI, "failed to decode response").WithCause(err)
	}

	return c.transformer.TransformResponse(&oaiResp), nil
}

// Stream sends a streaming completion request.
func (c *Client) Stream(ctx context.Context, req *types.CompletionRequest) (types.StreamReader, error) {
	oaiReq := c.transformer.TransformRequest(req)
	oaiReq.Stream = true
	oaiReq.StreamOptions = &StreamOptions{IncludeUsage: true}

	body, err := json.Marshal(oaiReq)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal request").WithCause(err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.ErrProviderUnavailable(types.ProviderOpenAI, "request failed").WithCause(err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, c.handleErrorResponse(resp)
	}

	return newStreamReader(resp.Body, c.transformer), nil
}

// setHeaders sets the required headers for OpenAI API requests.
func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
}

// handleErrorResponse converts an error response to a RouterError.
func (c *Client) handleErrorResponse(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp ErrorResponse
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error != nil {
		return c.mapAPIError(errResp.Error, resp.StatusCode)
	}

	return errors.ErrServerError(types.ProviderOpenAI, string(body)).WithStatusCode(resp.StatusCode)
}

// mapAPIError maps OpenAI API error to RouterError.
func (c *Client) mapAPIError(apiErr *APIError, statusCode int) error {
	switch statusCode {
	case http.StatusUnauthorized:
		return errors.ErrInvalidAPIKey(types.ProviderOpenAI).WithStatusCode(statusCode)
	case http.StatusTooManyRequests:
		return errors.ErrRateLimit(types.ProviderOpenAI, apiErr.Message).WithStatusCode(statusCode)
	case http.StatusNotFound:
		return errors.ErrModelNotFound(types.ProviderOpenAI, apiErr.Message).WithStatusCode(statusCode)
	case http.StatusBadRequest:
		if strings.Contains(apiErr.Message, "context_length") {
			return errors.ErrContextLength(types.ProviderOpenAI, apiErr.Message).WithStatusCode(statusCode)
		}
		return errors.ErrInvalidRequest(apiErr.Message).WithProvider(types.ProviderOpenAI).WithStatusCode(statusCode)
	default:
		return errors.ErrServerError(types.ProviderOpenAI, apiErr.Message).WithStatusCode(statusCode)
	}
}

// streamReader implements types.StreamReader for OpenAI.
type streamReader struct {
	reader      *bufio.Reader
	body        io.ReadCloser
	transformer *Transformer
	response    *types.CompletionResponse
	done        bool

	// Accumulated state
	id         string
	model      string
	content    strings.Builder
	toolCalls  map[int]*types.ToolCall  // index -> tool call
	toolInputs map[int]*strings.Builder // index -> accumulated arguments
	usage      *types.Usage
	stopReason types.StopReason
}

func newStreamReader(body io.ReadCloser, transformer *Transformer) *streamReader {
	return &streamReader{
		reader:      bufio.NewReader(body),
		body:        body,
		transformer: transformer,
		toolCalls:   make(map[int]*types.ToolCall),
		toolInputs:  make(map[int]*strings.Builder),
	}
}

// Next returns the next stream event.
func (s *streamReader) Next() (*types.StreamEvent, error) {
	if s.done {
		return nil, nil
	}

	for {
		line, err := s.reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				s.done = true
				s.buildResponse()
				return nil, nil
			}
			return nil, err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			s.done = true
			s.buildResponse()
			return &types.StreamEvent{
				Type:       types.StreamEventDone,
				Usage:      s.usage,
				StopReason: s.stopReason,
				ResponseID: s.id,
			}, nil
		}

		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		event := s.processChunk(&chunk)
		if event != nil {
			return event, nil
		}
	}
}

// processChunk processes a stream chunk and returns an event if applicable.
func (s *streamReader) processChunk(chunk *StreamChunk) *types.StreamEvent {
	// Store metadata
	if s.id == "" {
		s.id = chunk.ID
	}
	if s.model == "" {
		s.model = chunk.Model
	}

	// Handle usage (comes with final chunk)
	if chunk.Usage != nil {
		s.usage = &types.Usage{
			InputTokens:  chunk.Usage.PromptTokens,
			OutputTokens: chunk.Usage.CompletionTokens,
			TotalTokens:  chunk.Usage.TotalTokens,
		}
	}

	if len(chunk.Choices) == 0 {
		return nil
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	// Handle finish reason
	if choice.FinishReason != "" {
		s.stopReason = s.transformer.transformStopReason(choice.FinishReason)
	}

	// Handle content delta
	if delta.Content != "" {
		s.content.WriteString(delta.Content)
		return &types.StreamEvent{
			Type: types.StreamEventContentDelta,
			Delta: &types.ContentBlock{
				Type: types.ContentTypeText,
				Text: delta.Content,
			},
			Index: 0,
		}
	}

	// Handle tool calls
	for _, tc := range delta.ToolCalls {
		idx := 0
		if tc.Index != nil {
			idx = *tc.Index
		}

		// New tool call
		if tc.ID != "" {
			s.toolCalls[idx] = &types.ToolCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
			}
			s.toolInputs[idx] = &strings.Builder{}

			return &types.StreamEvent{
				Type: types.StreamEventToolCallStart,
				ToolCall: &types.ToolCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
				},
			}
		}

		// Tool call arguments delta
		if tc.Function.Arguments != "" {
			if builder, ok := s.toolInputs[idx]; ok {
				builder.WriteString(tc.Function.Arguments)
			}

			return &types.StreamEvent{
				Type:           types.StreamEventToolCallDelta,
				ToolInputDelta: tc.Function.Arguments,
				Index:          idx,
			}
		}
	}

	return nil
}

// buildResponse builds the final response from accumulated state.
func (s *streamReader) buildResponse() {
	var content []types.ContentBlock

	// Add text content
	if s.content.Len() > 0 {
		content = append(content, types.ContentBlock{
			Type: types.ContentTypeText,
			Text: s.content.String(),
		})
	}

	// Finalize tool calls
	var toolCalls []types.ToolCall
	for idx, tc := range s.toolCalls {
		if builder, ok := s.toolInputs[idx]; ok {
			var input any
			json.Unmarshal([]byte(builder.String()), &input)
			tc.Input = input
		}
		toolCalls = append(toolCalls, *tc)

		content = append(content, types.ContentBlock{
			Type:      types.ContentTypeToolUse,
			ToolUseID: tc.ID,
			ToolName:  tc.Name,
			ToolInput: tc.Input,
		})
	}

	s.response = &types.CompletionResponse{
		ID:         s.id,
		Provider:   types.ProviderOpenAI,
		Model:      s.model,
		Content:    content,
		StopReason: s.stopReason,
		ToolCalls:  toolCalls,
		CreatedAt:  time.Now(),
	}

	if s.usage != nil {
		s.response.Usage = *s.usage
	}
}

// Close closes the stream.
func (s *streamReader) Close() error {
	return s.body.Close()
}

// Response returns the accumulated response.
func (s *streamReader) Response() *types.CompletionResponse {
	return s.response
}

// Ensure Client implements provider.Provider
var _ provider.Provider = (*Client)(nil)

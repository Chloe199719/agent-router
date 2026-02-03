// Package google provides a Google Gemini API client implementation.
package google

import (
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
	defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
)

// Client is a Google Gemini API client.
type Client struct {
	config      *provider.Config
	httpClient  *http.Client
	baseURL     string
	transformer *Transformer
}

// New creates a new Google client.
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
	return types.ProviderGoogle
}

// SupportsFeature checks if Google supports a feature.
func (c *Client) SupportsFeature(feature types.Feature) bool {
	switch feature {
	case types.FeatureStreaming,
		types.FeatureStructuredOutput,
		types.FeatureTools,
		types.FeatureVision,
		types.FeatureJSON:
		return true
	case types.FeatureBatch:
		return true // Via Vertex AI
	default:
		return false
	}
}

// Models returns available Google models.
func (c *Client) Models() []string {
	return []string{
		"gemini-2.0-flash",
		"gemini-2.0-flash-lite",
		"gemini-1.5-pro",
		"gemini-1.5-flash",
		"gemini-1.5-flash-8b",
		"gemini-1.0-pro",
	}
}

// Complete sends a completion request.
func (c *Client) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	gReq := c.transformer.TransformRequest(req)

	body, err := json.Marshal(gReq)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal request").WithCause(err)
	}

	url := c.buildURL(req.Model, false)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.ErrProviderUnavailable(types.ProviderGoogle, "request failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.handleErrorResponse(resp)
	}

	var gResp GenerateContentResponse
	if err := json.NewDecoder(resp.Body).Decode(&gResp); err != nil {
		return nil, errors.ErrServerError(types.ProviderGoogle, "failed to decode response").WithCause(err)
	}

	result := c.transformer.TransformResponse(&gResp)
	if result != nil {
		result.Model = req.Model
	}
	return result, nil
}

// Stream sends a streaming completion request.
func (c *Client) Stream(ctx context.Context, req *types.CompletionRequest) (types.StreamReader, error) {
	gReq := c.transformer.TransformRequest(req)

	body, err := json.Marshal(gReq)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal request").WithCause(err)
	}

	url := c.buildURL(req.Model, true)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.ErrProviderUnavailable(types.ProviderGoogle, "request failed").WithCause(err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, c.handleErrorResponse(resp)
	}

	return newStreamReader(resp.Body, c.transformer, req.Model), nil
}

// buildURL builds the API URL for a given model and streaming flag.
func (c *Client) buildURL(model string, stream bool) string {
	action := "generateContent"
	if stream {
		action = "streamGenerateContent"
	}
	return c.baseURL + "/models/" + model + ":" + action + "?key=" + c.config.APIKey
}

// setHeaders sets the required headers for Google API requests.
func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
}

// handleErrorResponse converts an error response to a RouterError.
func (c *Client) handleErrorResponse(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp ErrorResponse
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error != nil {
		return c.mapAPIError(errResp.Error, resp.StatusCode)
	}

	return errors.ErrServerError(types.ProviderGoogle, string(body)).WithStatusCode(resp.StatusCode)
}

// mapAPIError maps Google API error to RouterError.
func (c *Client) mapAPIError(apiErr *APIError, statusCode int) error {
	switch statusCode {
	case http.StatusUnauthorized:
		return errors.ErrInvalidAPIKey(types.ProviderGoogle).WithStatusCode(statusCode)
	case http.StatusTooManyRequests:
		return errors.ErrRateLimit(types.ProviderGoogle, apiErr.Message).WithStatusCode(statusCode)
	case http.StatusNotFound:
		return errors.ErrModelNotFound(types.ProviderGoogle, apiErr.Message).WithStatusCode(statusCode)
	case http.StatusBadRequest:
		if strings.Contains(apiErr.Message, "context") || strings.Contains(apiErr.Message, "token") {
			return errors.ErrContextLength(types.ProviderGoogle, apiErr.Message).WithStatusCode(statusCode)
		}
		return errors.ErrInvalidRequest(apiErr.Message).WithProvider(types.ProviderGoogle).WithStatusCode(statusCode)
	default:
		return errors.ErrServerError(types.ProviderGoogle, apiErr.Message).WithStatusCode(statusCode)
	}
}

// streamReader implements types.StreamReader for Google.
type streamReader struct {
	decoder      *json.Decoder
	body         io.ReadCloser
	transformer  *Transformer
	model        string
	response     *types.CompletionResponse
	done         bool
	arrayStarted bool

	// Accumulated state
	content    []types.ContentBlock
	toolCalls  []types.ToolCall
	usage      *types.Usage
	stopReason types.StopReason
	started    bool
}

func newStreamReader(body io.ReadCloser, transformer *Transformer, model string) *streamReader {
	return &streamReader{
		decoder:     json.NewDecoder(body),
		body:        body,
		transformer: transformer,
		model:       model,
	}
}

// Next returns the next stream event.
func (s *streamReader) Next() (*types.StreamEvent, error) {
	if s.done {
		return nil, nil
	}

	// Send start event first
	if !s.started {
		s.started = true
		return &types.StreamEvent{
			Type:  types.StreamEventStart,
			Model: s.model,
		}, nil
	}

	// Read opening bracket of JSON array
	if !s.arrayStarted {
		token, err := s.decoder.Token()
		if err != nil {
			if err == io.EOF {
				s.done = true
				s.buildResponse()
				return &types.StreamEvent{
					Type:       types.StreamEventDone,
					Usage:      s.usage,
					StopReason: s.stopReason,
				}, nil
			}
			return nil, err
		}
		if delim, ok := token.(json.Delim); ok && delim == '[' {
			s.arrayStarted = true
		}
	}

	// Read next element from JSON array
	for s.decoder.More() {
		var chunk StreamChunk
		if err := s.decoder.Decode(&chunk); err != nil {
			if err == io.EOF {
				break
			}
			continue
		}

		event := s.processChunk(&chunk)
		if event != nil {
			return event, nil
		}
	}

	// Array finished
	s.done = true
	s.buildResponse()
	return &types.StreamEvent{
		Type:       types.StreamEventDone,
		Usage:      s.usage,
		StopReason: s.stopReason,
	}, nil
}

// processChunk processes a stream chunk and returns an event if applicable.
func (s *streamReader) processChunk(chunk *StreamChunk) *types.StreamEvent {
	if len(chunk.Candidates) == 0 {
		return nil
	}

	candidate := chunk.Candidates[0]

	// Handle finish reason
	if candidate.FinishReason != "" {
		s.stopReason = s.transformer.transformStopReason(candidate.FinishReason)
	}

	// Handle usage
	if chunk.UsageMetadata != nil {
		s.usage = &types.Usage{
			InputTokens:  chunk.UsageMetadata.PromptTokenCount,
			OutputTokens: chunk.UsageMetadata.CandidatesTokenCount,
			TotalTokens:  chunk.UsageMetadata.TotalTokenCount,
		}
	}

	if candidate.Content == nil {
		return nil
	}

	// Process parts
	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			// Accumulate text
			if len(s.content) == 0 || s.content[len(s.content)-1].Type != types.ContentTypeText {
				s.content = append(s.content, types.ContentBlock{
					Type: types.ContentTypeText,
					Text: part.Text,
				})
			} else {
				s.content[len(s.content)-1].Text += part.Text
			}

			return &types.StreamEvent{
				Type: types.StreamEventContentDelta,
				Delta: &types.ContentBlock{
					Type: types.ContentTypeText,
					Text: part.Text,
				},
			}
		}

		if part.FunctionCall != nil {
			tc := types.ToolCall{
				Name:  part.FunctionCall.Name,
				Input: part.FunctionCall.Args,
			}
			s.toolCalls = append(s.toolCalls, tc)
			s.content = append(s.content, types.ContentBlock{
				Type:      types.ContentTypeToolUse,
				ToolName:  part.FunctionCall.Name,
				ToolInput: part.FunctionCall.Args,
			})

			return &types.StreamEvent{
				Type:     types.StreamEventToolCallStart,
				ToolCall: &tc,
			}
		}
	}

	return nil
}

// buildResponse builds the final response from accumulated state.
func (s *streamReader) buildResponse() {
	s.response = &types.CompletionResponse{
		Provider:   types.ProviderGoogle,
		Model:      s.model,
		Content:    s.content,
		StopReason: s.stopReason,
		ToolCalls:  s.toolCalls,
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

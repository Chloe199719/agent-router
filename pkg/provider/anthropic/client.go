// Package anthropic provides an Anthropic API client implementation.
package anthropic

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
	defaultBaseURL = "https://api.anthropic.com"
	defaultVersion = "2023-06-01"
	betaHeader     = "prompt-caching-2024-07-31,output-128k-2025-02-19"
)

// Client is an Anthropic API client.
type Client struct {
	config      *provider.Config
	httpClient  *http.Client
	baseURL     string
	version     string
	transformer *Transformer
}

// New creates a new Anthropic client.
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
		version:     defaultVersion,
		transformer: NewTransformer(),
	}
}

// Name returns the provider name.
func (c *Client) Name() types.Provider {
	return types.ProviderAnthropic
}

// SupportsFeature checks if Anthropic supports a feature.
func (c *Client) SupportsFeature(feature types.Feature) bool {
	switch feature {
	case types.FeatureStreaming,
		types.FeatureStructuredOutput,
		types.FeatureTools,
		types.FeatureVision,
		types.FeatureBatch:
		return true
	case types.FeatureJSON:
		return false // Anthropic doesn't have simple JSON mode, only structured output
	default:
		return false
	}
}

// Models returns available Anthropic models.
func (c *Client) Models() []string {
	return []string{
		"claude-sonnet-4-20250514",
		"claude-opus-4-20250514",
		"claude-3-5-sonnet-20241022",
		"claude-3-5-haiku-20241022",
		"claude-3-opus-20240229",
		"claude-3-sonnet-20240229",
		"claude-3-haiku-20240307",
	}
}

// Complete sends a completion request.
func (c *Client) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	anthReq := c.transformer.TransformRequest(req)
	anthReq.Stream = false

	body, err := json.Marshal(anthReq)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal request").WithCause(err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.ErrProviderUnavailable(types.ProviderAnthropic, "request failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.handleErrorResponse(resp)
	}

	var anthResp MessagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&anthResp); err != nil {
		return nil, errors.ErrServerError(types.ProviderAnthropic, "failed to decode response").WithCause(err)
	}

	return c.transformer.TransformResponse(&anthResp), nil
}

// Stream sends a streaming completion request.
func (c *Client) Stream(ctx context.Context, req *types.CompletionRequest) (types.StreamReader, error) {
	anthReq := c.transformer.TransformRequest(req)
	anthReq.Stream = true

	body, err := json.Marshal(anthReq)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal request").WithCause(err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.ErrProviderUnavailable(types.ProviderAnthropic, "request failed").WithCause(err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, c.handleErrorResponse(resp)
	}

	return newStreamReader(resp.Body, c.transformer), nil
}

// setHeaders sets the required headers for Anthropic API requests.
func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.config.APIKey)
	req.Header.Set("anthropic-version", c.version)
	req.Header.Set("anthropic-beta", betaHeader)
}

// handleErrorResponse converts an error response to a RouterError.
func (c *Client) handleErrorResponse(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp ErrorResponse
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error != nil {
		return c.mapAPIError(errResp.Error, resp.StatusCode)
	}

	return errors.ErrServerError(types.ProviderAnthropic, string(body)).WithStatusCode(resp.StatusCode)
}

// mapAPIError maps Anthropic API error to RouterError.
func (c *Client) mapAPIError(apiErr *APIError, statusCode int) error {
	switch statusCode {
	case http.StatusUnauthorized:
		return errors.ErrInvalidAPIKey(types.ProviderAnthropic).WithStatusCode(statusCode)
	case http.StatusTooManyRequests:
		return errors.ErrRateLimit(types.ProviderAnthropic, apiErr.Message).WithStatusCode(statusCode)
	case http.StatusNotFound:
		return errors.ErrModelNotFound(types.ProviderAnthropic, apiErr.Message).WithStatusCode(statusCode)
	case http.StatusBadRequest:
		if strings.Contains(apiErr.Message, "context") || strings.Contains(apiErr.Message, "token") {
			return errors.ErrContextLength(types.ProviderAnthropic, apiErr.Message).WithStatusCode(statusCode)
		}
		return errors.ErrInvalidRequest(apiErr.Message).WithProvider(types.ProviderAnthropic).WithStatusCode(statusCode)
	default:
		return errors.ErrServerError(types.ProviderAnthropic, apiErr.Message).WithStatusCode(statusCode)
	}
}

// streamReader implements types.StreamReader for Anthropic.
type streamReader struct {
	reader      *bufio.Reader
	body        io.ReadCloser
	transformer *Transformer
	response    *types.CompletionResponse
	done        bool

	// Accumulated state
	id            string
	model         string
	contentBlocks []types.ContentBlock
	currentBlock  int
	toolCalls     []types.ToolCall
	usage         *types.Usage
	stopReason    types.StopReason
}

func newStreamReader(body io.ReadCloser, transformer *Transformer) *streamReader {
	return &streamReader{
		reader:      bufio.NewReader(body),
		body:        body,
		transformer: transformer,
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

		// Handle SSE format
		if strings.HasPrefix(line, "event: ") {
			// Read the data line
			dataLine, err := s.reader.ReadString('\n')
			if err != nil && err != io.EOF {
				return nil, err
			}
			dataLine = strings.TrimSpace(dataLine)

			if !strings.HasPrefix(dataLine, "data: ") {
				continue
			}

			data := strings.TrimPrefix(dataLine, "data: ")
			eventType := strings.TrimPrefix(line, "event: ")

			event, done := s.processEvent(eventType, data)
			if done {
				s.done = true
				s.buildResponse()
			}
			if event != nil {
				return event, nil
			}
		}
	}
}

// processEvent processes a stream event.
func (s *streamReader) processEvent(eventType, data string) (*types.StreamEvent, bool) {
	switch eventType {
	case "message_start":
		var event struct {
			Message MessagesResponse `json:"message"`
		}
		if err := json.Unmarshal([]byte(data), &event); err == nil {
			s.id = event.Message.ID
			s.model = event.Message.Model
			return &types.StreamEvent{
				Type:       types.StreamEventStart,
				ResponseID: s.id,
				Model:      s.model,
			}, false
		}

	case "content_block_start":
		var event struct {
			Index        int          `json:"index"`
			ContentBlock ContentBlock `json:"content_block"`
		}
		if err := json.Unmarshal([]byte(data), &event); err == nil {
			s.currentBlock = event.Index

			// Ensure we have enough blocks
			for len(s.contentBlocks) <= event.Index {
				s.contentBlocks = append(s.contentBlocks, types.ContentBlock{})
			}

			if event.ContentBlock.Type == "tool_use" {
				s.contentBlocks[event.Index] = types.ContentBlock{
					Type:      types.ContentTypeToolUse,
					ToolUseID: event.ContentBlock.ID,
					ToolName:  event.ContentBlock.Name,
				}
				return &types.StreamEvent{
					Type: types.StreamEventToolCallStart,
					ToolCall: &types.ToolCall{
						ID:   event.ContentBlock.ID,
						Name: event.ContentBlock.Name,
					},
				}, false
			} else {
				s.contentBlocks[event.Index] = types.ContentBlock{
					Type: types.ContentTypeText,
				}
			}
		}

	case "content_block_delta":
		var event struct {
			Index int   `json:"index"`
			Delta Delta `json:"delta"`
		}
		if err := json.Unmarshal([]byte(data), &event); err == nil {
			if event.Delta.Text != "" {
				// Text delta
				if event.Index < len(s.contentBlocks) {
					s.contentBlocks[event.Index].Text += event.Delta.Text
				}
				return &types.StreamEvent{
					Type: types.StreamEventContentDelta,
					Delta: &types.ContentBlock{
						Type: types.ContentTypeText,
						Text: event.Delta.Text,
					},
					Index: event.Index,
				}, false
			} else if event.Delta.PartialJSON != "" {
				// Tool input delta
				return &types.StreamEvent{
					Type:           types.StreamEventToolCallDelta,
					ToolInputDelta: event.Delta.PartialJSON,
					Index:          event.Index,
				}, false
			}
		}

	case "content_block_stop":
		var event struct {
			Index int `json:"index"`
		}
		if err := json.Unmarshal([]byte(data), &event); err == nil {
			if event.Index < len(s.contentBlocks) && s.contentBlocks[event.Index].Type == types.ContentTypeToolUse {
				tc := types.ToolCall{
					ID:    s.contentBlocks[event.Index].ToolUseID,
					Name:  s.contentBlocks[event.Index].ToolName,
					Input: s.contentBlocks[event.Index].ToolInput,
				}
				s.toolCalls = append(s.toolCalls, tc)
				return &types.StreamEvent{
					Type:     types.StreamEventToolCallEnd,
					ToolCall: &tc,
				}, false
			}
		}

	case "message_delta":
		var event struct {
			Delta Delta `json:"delta"`
			Usage Usage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &event); err == nil {
			s.stopReason = s.transformer.transformStopReason(event.Delta.StopReason)
			if event.Usage.OutputTokens > 0 {
				s.usage = &types.Usage{
					OutputTokens: event.Usage.OutputTokens,
				}
			}
		}

	case "message_stop":
		return &types.StreamEvent{
			Type:       types.StreamEventDone,
			Usage:      s.usage,
			StopReason: s.stopReason,
			ResponseID: s.id,
		}, true

	case "error":
		var event struct {
			Error APIError `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &event); err == nil {
			return &types.StreamEvent{
				Type:  types.StreamEventError,
				Error: errors.ErrServerError(types.ProviderAnthropic, event.Error.Message),
			}, true
		}
	}

	return nil, false
}

// buildResponse builds the final response from accumulated state.
func (s *streamReader) buildResponse() {
	s.response = &types.CompletionResponse{
		ID:         s.id,
		Provider:   types.ProviderAnthropic,
		Model:      s.model,
		Content:    s.contentBlocks,
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

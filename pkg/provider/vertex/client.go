// Package vertex provides a Google Vertex AI client implementation.
//
// Vertex AI uses the same Gemini API request/response format as the standard
// Google Gemini API, but with a different base URL pattern and authentication
// mechanism (OAuth2 Bearer token or API key).
//
// URL pattern:
//
//	https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}:{ACTION}
//
// For the "global" location, the URL uses no location prefix:
//
//	https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/global/publishers/google/models/{MODEL}:{ACTION}
package vertex

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/Chloe199719/agent-router/pkg/errors"
	"github.com/Chloe199719/agent-router/pkg/provider"
	googleProvider "github.com/Chloe199719/agent-router/pkg/provider/google"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// Client is a Google Vertex AI client.
type Client struct {
	config      *provider.Config
	httpClient  *http.Client
	projectID   string
	location    string
	baseURL     string
	transformer *googleProvider.Transformer
}

// New creates a new Vertex AI client.
//
// The projectID and location are required. Authentication is provided via
// provider.WithAccessToken() (OAuth2 Bearer token) or provider.WithAPIKey()
// (API key). At least one authentication method must be provided.
func New(projectID, location string, opts ...provider.Option) *Client {
	cfg := provider.DefaultConfig()
	provider.ApplyOptions(cfg, opts...)

	if projectID == "" && cfg.ProjectID != "" {
		projectID = cfg.ProjectID
	}
	if location == "" && cfg.Location != "" {
		location = cfg.Location
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		if location == "global" {
			baseURL = "https://aiplatform.googleapis.com/v1"
		} else {
			baseURL = fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1", location)
		}
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
		projectID:   projectID,
		location:    location,
		baseURL:     baseURL,
		transformer: googleProvider.NewTransformer(),
	}
}

// Name returns the provider name.
func (c *Client) Name() types.Provider {
	return types.ProviderVertex
}

// SupportsFeature checks if Vertex AI supports a feature.
func (c *Client) SupportsFeature(feature types.Feature) bool {
	switch feature {
	case types.FeatureStreaming,
		types.FeatureStructuredOutput,
		types.FeatureTools,
		types.FeatureVision,
		types.FeatureJSON,
		types.FeatureBatch:
		return true
	default:
		return false
	}
}

// Models returns available Vertex AI Gemini models.
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

	url := c.buildURL(req.Model, "generateContent")
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.ErrProviderUnavailable(types.ProviderVertex, "request failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.handleErrorResponse(resp)
	}

	var gResp googleProvider.GenerateContentResponse
	if err := json.NewDecoder(resp.Body).Decode(&gResp); err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to decode response").WithCause(err)
	}

	result := c.transformer.TransformResponse(&gResp)
	if result != nil {
		result.Provider = types.ProviderVertex
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

	url := c.buildURL(req.Model, "streamGenerateContent")
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.ErrProviderUnavailable(types.ProviderVertex, "request failed").WithCause(err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, c.handleErrorResponse(resp)
	}

	return newStreamReader(resp.Body, c.transformer, req.Model), nil
}

// buildURL builds the Vertex AI API URL for a given model and action.
func (c *Client) buildURL(model, action string) string {
	url := fmt.Sprintf("%s/projects/%s/locations/%s/publishers/google/models/%s:%s",
		c.baseURL, c.projectID, c.location, model, action)

	// If using API key auth (no access token), append key as query parameter
	if c.config.AccessToken == "" && c.config.APIKey != "" {
		url += "?key=" + c.config.APIKey
	}

	return url
}

// setHeaders sets the required headers for Vertex AI API requests.
func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")

	// Prefer access token (OAuth2 Bearer), fall back to API key (handled in URL)
	if c.config.AccessToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.config.AccessToken)
	}
}

// handleErrorResponse converts an error response to a RouterError.
func (c *Client) handleErrorResponse(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp googleProvider.ErrorResponse
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error != nil {
		return c.mapAPIError(errResp.Error, resp.StatusCode)
	}

	return errors.ErrServerError(types.ProviderVertex, string(body)).WithStatusCode(resp.StatusCode)
}

// mapAPIError maps Vertex AI API error to RouterError.
func (c *Client) mapAPIError(apiErr *googleProvider.APIError, statusCode int) error {
	switch statusCode {
	case http.StatusUnauthorized:
		return errors.ErrInvalidAPIKey(types.ProviderVertex).WithStatusCode(statusCode)
	case http.StatusForbidden:
		return errors.ErrAuthentication(types.ProviderVertex, apiErr.Message).WithStatusCode(statusCode)
	case http.StatusTooManyRequests:
		return errors.ErrRateLimit(types.ProviderVertex, apiErr.Message).WithStatusCode(statusCode)
	case http.StatusNotFound:
		return errors.ErrModelNotFound(types.ProviderVertex, apiErr.Message).WithStatusCode(statusCode)
	case http.StatusBadRequest:
		if contains(apiErr.Message, "context", "token") {
			return errors.ErrContextLength(types.ProviderVertex, apiErr.Message).WithStatusCode(statusCode)
		}
		return errors.ErrInvalidRequest(apiErr.Message).WithProvider(types.ProviderVertex).WithStatusCode(statusCode)
	default:
		return errors.ErrServerError(types.ProviderVertex, apiErr.Message).WithStatusCode(statusCode)
	}
}

// contains checks if s contains any of the substrings.
func contains(s string, substrs ...string) bool {
	for _, sub := range substrs {
		if len(s) >= len(sub) {
			for i := 0; i <= len(s)-len(sub); i++ {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
		}
	}
	return false
}

// streamReader implements types.StreamReader for Vertex AI.
// Vertex AI uses the same JSON array streaming format as the Google Gemini API.
type streamReader struct {
	decoder      *json.Decoder
	body         io.ReadCloser
	transformer  *googleProvider.Transformer
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

func newStreamReader(body io.ReadCloser, transformer *googleProvider.Transformer, model string) *streamReader {
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
		var chunk googleProvider.StreamChunk
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
func (s *streamReader) processChunk(chunk *googleProvider.StreamChunk) *types.StreamEvent {
	if len(chunk.Candidates) == 0 {
		return nil
	}

	candidate := chunk.Candidates[0]

	// Handle finish reason
	if candidate.FinishReason != "" {
		s.stopReason = s.transformer.TransformStopReason(candidate.FinishReason)
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
		Provider:   types.ProviderVertex,
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

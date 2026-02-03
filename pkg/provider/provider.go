// Package provider defines the interface for LLM providers.
package provider

import (
	"context"
	"net/http"

	"github.com/Chloe199719/agent-router/pkg/types"
)

// Provider is the interface that all LLM providers must implement.
type Provider interface {
	// Name returns the provider identifier.
	Name() types.Provider

	// Complete sends a completion request and returns the response.
	Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error)

	// Stream sends a streaming completion request and returns a stream reader.
	Stream(ctx context.Context, req *types.CompletionRequest) (types.StreamReader, error)

	// SupportsFeature checks if the provider supports a specific feature.
	SupportsFeature(feature types.Feature) bool

	// Models returns the list of available models for this provider.
	Models() []string
}

// BatchProvider is an optional interface for providers that support batch processing.
type BatchProvider interface {
	Provider

	// CreateBatch creates a new batch job.
	CreateBatch(ctx context.Context, requests []BatchRequest) (*BatchJob, error)

	// GetBatch retrieves the status of a batch job.
	GetBatch(ctx context.Context, batchID string) (*BatchJob, error)

	// GetBatchResults retrieves the results of a completed batch job.
	GetBatchResults(ctx context.Context, batchID string) ([]BatchResult, error)

	// CancelBatch cancels a batch job.
	CancelBatch(ctx context.Context, batchID string) error

	// ListBatches lists all batch jobs.
	ListBatches(ctx context.Context, opts *ListBatchOptions) ([]BatchJob, error)
}

// BatchRequest wraps a completion request with a custom ID for batch processing.
type BatchRequest struct {
	// CustomID is a developer-provided ID for matching results to requests.
	CustomID string `json:"custom_id"`

	// Request is the completion request to process.
	Request *types.CompletionRequest `json:"request"`
}

// BatchJob represents a batch processing job.
type BatchJob struct {
	// ID is the unique identifier for this batch.
	ID string `json:"id"`

	// Provider that is processing this batch.
	Provider types.Provider `json:"provider"`

	// Status of the batch job.
	Status BatchStatus `json:"status"`

	// CreatedAt is when the batch was created.
	CreatedAt int64 `json:"created_at"`

	// CompletedAt is when the batch completed (if applicable).
	CompletedAt int64 `json:"completed_at,omitempty"`

	// ExpiresAt is when the batch will expire.
	ExpiresAt int64 `json:"expires_at,omitempty"`

	// RequestCounts tracks the progress of requests.
	RequestCounts RequestCounts `json:"request_counts"`

	// Metadata is provider-specific metadata.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// BatchStatus represents the status of a batch job.
type BatchStatus string

const (
	BatchStatusPending    BatchStatus = "pending"
	BatchStatusValidating BatchStatus = "validating"
	BatchStatusInProgress BatchStatus = "in_progress"
	BatchStatusFinalizing BatchStatus = "finalizing"
	BatchStatusCompleted  BatchStatus = "completed"
	BatchStatusFailed     BatchStatus = "failed"
	BatchStatusCancelled  BatchStatus = "cancelled"
	BatchStatusExpired    BatchStatus = "expired"
)

// RequestCounts tracks batch request progress.
type RequestCounts struct {
	Total     int `json:"total"`
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
}

// BatchResult represents a single result from a batch job.
type BatchResult struct {
	// CustomID matches the request's custom_id.
	CustomID string `json:"custom_id"`

	// Response is the completion response (if successful).
	Response *types.CompletionResponse `json:"response,omitempty"`

	// Error is the error that occurred (if failed).
	Error error `json:"error,omitempty"`
}

// ListBatchOptions configures batch listing.
type ListBatchOptions struct {
	Limit int    `json:"limit,omitempty"`
	After string `json:"after,omitempty"`
}

// Config contains common configuration for providers.
type Config struct {
	// APIKey for authentication.
	APIKey string

	// BaseURL overrides the default API endpoint.
	BaseURL string

	// HTTPClient is a custom HTTP client to use.
	HTTPClient *http.Client

	// Timeout for requests (in seconds).
	Timeout int

	// MaxRetries is the maximum number of retries for failed requests.
	MaxRetries int

	// Debug enables debug logging.
	Debug bool
}

// Option is a function that configures a provider.
type Option func(*Config)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *Config) {
		c.APIKey = key
	}
}

// WithBaseURL sets a custom base URL.
func WithBaseURL(url string) Option {
	return func(c *Config) {
		c.BaseURL = url
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *Config) {
		c.HTTPClient = client
	}
}

// WithTimeout sets the request timeout.
func WithTimeout(seconds int) Option {
	return func(c *Config) {
		c.Timeout = seconds
	}
}

// WithMaxRetries sets the maximum number of retries.
func WithMaxRetries(n int) Option {
	return func(c *Config) {
		c.MaxRetries = n
	}
}

// WithDebug enables debug logging.
func WithDebug(debug bool) Option {
	return func(c *Config) {
		c.Debug = debug
	}
}

// DefaultConfig returns a default configuration.
func DefaultConfig() *Config {
	return &Config{
		Timeout:    120,
		MaxRetries: 3,
	}
}

// ApplyOptions applies options to a config.
func ApplyOptions(cfg *Config, opts ...Option) {
	for _, opt := range opts {
		opt(cfg)
	}
}

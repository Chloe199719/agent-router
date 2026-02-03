// Package batch provides a unified batch processing interface across providers.
package batch

import (
	"context"
	"time"

	"github.com/Chloe199719/agent-router/pkg/errors"
	"github.com/Chloe199719/agent-router/pkg/provider"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// Request wraps a completion request with a custom ID for batch processing.
type Request struct {
	// CustomID is a developer-provided ID for matching results to requests.
	// Must be unique within a batch.
	CustomID string `json:"custom_id"`

	// Request is the completion request to process.
	Request *types.CompletionRequest `json:"request"`
}

// Job represents a batch processing job.
type Job struct {
	// ID is the provider's unique identifier for this batch.
	ID string `json:"id"`

	// Provider that is processing this batch.
	Provider types.Provider `json:"provider"`

	// Status of the batch job.
	Status Status `json:"status"`

	// CreatedAt is when the batch was created.
	CreatedAt time.Time `json:"created_at"`

	// CompletedAt is when the batch completed (if applicable).
	CompletedAt *time.Time `json:"completed_at,omitempty"`

	// ExpiresAt is when the batch will expire.
	ExpiresAt *time.Time `json:"expires_at,omitempty"`

	// Counts tracks the progress of requests.
	Counts Counts `json:"counts"`

	// Metadata contains provider-specific information.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Status represents the status of a batch job.
type Status string

const (
	StatusPending    Status = "pending"
	StatusValidating Status = "validating"
	StatusInProgress Status = "in_progress"
	StatusFinalizing Status = "finalizing"
	StatusCompleted  Status = "completed"
	StatusFailed     Status = "failed"
	StatusCancelled  Status = "cancelled"
	StatusExpired    Status = "expired"
)

// IsDone returns true if the batch is in a terminal state.
func (s Status) IsDone() bool {
	switch s {
	case StatusCompleted, StatusFailed, StatusCancelled, StatusExpired:
		return true
	default:
		return false
	}
}

// Counts tracks batch request progress.
type Counts struct {
	Total     int `json:"total"`
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
}

// Result represents a single result from a batch job.
type Result struct {
	// CustomID matches the request's CustomID.
	CustomID string `json:"custom_id"`

	// Response is the completion response (if successful).
	Response *types.CompletionResponse `json:"response,omitempty"`

	// Error is the error that occurred (if failed).
	Error error `json:"error,omitempty"`
}

// ListOptions configures batch listing.
type ListOptions struct {
	// Limit is the maximum number of batches to return.
	Limit int `json:"limit,omitempty"`

	// After is a cursor for pagination.
	After string `json:"after,omitempty"`
}

// Manager provides a unified interface for batch processing across providers.
type Manager struct {
	providers map[types.Provider]provider.BatchProvider
}

// NewManager creates a new batch manager.
func NewManager() *Manager {
	return &Manager{
		providers: make(map[types.Provider]provider.BatchProvider),
	}
}

// RegisterProvider registers a batch-capable provider.
func (m *Manager) RegisterProvider(p provider.BatchProvider) {
	m.providers[p.Name()] = p
}

// Create creates a new batch job.
func (m *Manager) Create(ctx context.Context, providerName types.Provider, requests []Request) (*Job, error) {
	p, ok := m.providers[providerName]
	if !ok {
		return nil, errors.ErrProviderUnavailable(providerName, "provider not registered or does not support batch")
	}

	// Convert to provider batch requests
	batchReqs := make([]provider.BatchRequest, len(requests))
	for i, req := range requests {
		batchReqs[i] = provider.BatchRequest{
			CustomID: req.CustomID,
			Request:  req.Request,
		}
	}

	job, err := p.CreateBatch(ctx, batchReqs)
	if err != nil {
		return nil, err
	}

	return convertJob(job), nil
}

// Get retrieves the status of a batch job.
func (m *Manager) Get(ctx context.Context, providerName types.Provider, batchID string) (*Job, error) {
	p, ok := m.providers[providerName]
	if !ok {
		return nil, errors.ErrProviderUnavailable(providerName, "provider not registered or does not support batch")
	}

	job, err := p.GetBatch(ctx, batchID)
	if err != nil {
		return nil, err
	}

	return convertJob(job), nil
}

// GetResults retrieves the results of a completed batch job.
func (m *Manager) GetResults(ctx context.Context, providerName types.Provider, batchID string) ([]Result, error) {
	p, ok := m.providers[providerName]
	if !ok {
		return nil, errors.ErrProviderUnavailable(providerName, "provider not registered or does not support batch")
	}

	results, err := p.GetBatchResults(ctx, batchID)
	if err != nil {
		return nil, err
	}

	return convertResults(results), nil
}

// Cancel cancels a batch job.
func (m *Manager) Cancel(ctx context.Context, providerName types.Provider, batchID string) error {
	p, ok := m.providers[providerName]
	if !ok {
		return errors.ErrProviderUnavailable(providerName, "provider not registered or does not support batch")
	}

	return p.CancelBatch(ctx, batchID)
}

// List lists batch jobs for a provider.
func (m *Manager) List(ctx context.Context, providerName types.Provider, opts *ListOptions) ([]Job, error) {
	p, ok := m.providers[providerName]
	if !ok {
		return nil, errors.ErrProviderUnavailable(providerName, "provider not registered or does not support batch")
	}

	var listOpts *provider.ListBatchOptions
	if opts != nil {
		listOpts = &provider.ListBatchOptions{
			Limit: opts.Limit,
			After: opts.After,
		}
	}

	jobs, err := p.ListBatches(ctx, listOpts)
	if err != nil {
		return nil, err
	}

	result := make([]Job, len(jobs))
	for i, job := range jobs {
		result[i] = *convertJob(&job)
	}

	return result, nil
}

// Wait waits for a batch to complete, polling at the specified interval.
func (m *Manager) Wait(ctx context.Context, providerName types.Provider, batchID string, pollInterval time.Duration) (*Job, error) {
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
			job, err := m.Get(ctx, providerName, batchID)
			if err != nil {
				return nil, err
			}
			if job.Status.IsDone() {
				return job, nil
			}
		}
	}
}

// convertJob converts provider.BatchJob to batch.Job.
func convertJob(j *provider.BatchJob) *Job {
	job := &Job{
		ID:       j.ID,
		Provider: j.Provider,
		Status:   Status(j.Status),
		Counts: Counts{
			Total:     j.RequestCounts.Total,
			Completed: j.RequestCounts.Completed,
			Failed:    j.RequestCounts.Failed,
		},
		Metadata: j.Metadata,
	}

	if j.CreatedAt > 0 {
		t := time.Unix(j.CreatedAt, 0)
		job.CreatedAt = t
	}
	if j.CompletedAt > 0 {
		t := time.Unix(j.CompletedAt, 0)
		job.CompletedAt = &t
	}
	if j.ExpiresAt > 0 {
		t := time.Unix(j.ExpiresAt, 0)
		job.ExpiresAt = &t
	}

	return job
}

// convertResults converts provider.BatchResult to batch.Result.
func convertResults(results []provider.BatchResult) []Result {
	out := make([]Result, len(results))
	for i, r := range results {
		out[i] = Result{
			CustomID: r.CustomID,
			Response: r.Response,
			Error:    r.Error,
		}
	}
	return out
}

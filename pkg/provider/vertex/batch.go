package vertex

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Chloe199719/agent-router/pkg/errors"
	"github.com/Chloe199719/agent-router/pkg/provider"
	googleProvider "github.com/Chloe199719/agent-router/pkg/provider/google"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// CreateBatch creates a new batch job using inline requests.
func (c *Client) CreateBatch(ctx context.Context, requests []provider.BatchRequest) (*provider.BatchJob, error) {
	if len(requests) == 0 {
		return nil, errors.ErrInvalidRequest("no requests provided").WithProvider(types.ProviderVertex)
	}

	// Get model from first request (all requests should use the same model)
	model := requests[0].Request.Model
	if model == "" {
		model = "gemini-2.0-flash" // Default model
	}

	// Build batch request items
	batchItems := make([]googleProvider.BatchRequestItem, len(requests))
	for i, req := range requests {
		gReq := c.transformer.TransformRequest(req.Request)
		batchItems[i] = googleProvider.BatchRequestItem{
			Request: gReq,
			Metadata: &googleProvider.RequestMetadata{
				Key: req.CustomID,
			},
		}
	}

	// Create batch request
	batchReq := &googleProvider.BatchGenerateContentRequest{
		Batch: &googleProvider.BatchConfig{
			DisplayName: fmt.Sprintf("batch-%d", time.Now().Unix()),
			InputConfig: &googleProvider.InputConfig{
				Requests: &googleProvider.RequestsInput{
					Requests: batchItems,
				},
			},
		},
	}

	body, err := json.Marshal(batchReq)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal batch request").WithCause(err)
	}

	url := c.buildURL(model, "batchGenerateContent")
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

	var batchJob googleProvider.BatchJob
	if err := json.NewDecoder(resp.Body).Decode(&batchJob); err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to decode response").WithCause(err)
	}

	return c.convertBatchJob(&batchJob, model), nil
}

// GetBatch retrieves the status of a batch job.
func (c *Client) GetBatch(ctx context.Context, batchID string) (*provider.BatchJob, error) {
	// batchID should be a full operation name or a short ID
	batchName := batchID
	if !strings.HasPrefix(batchID, "projects/") {
		// Try constructing the full name
		batchName = fmt.Sprintf("projects/%s/locations/%s/batchPredictionJobs/%s",
			c.projectID, c.location, batchID)
	}

	url := fmt.Sprintf("%s/%s", c.baseURL, batchName)
	if c.config.AccessToken == "" && c.config.APIKey != "" {
		url += "?key=" + c.config.APIKey
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
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

	var batchJob googleProvider.BatchJob
	if err := json.NewDecoder(resp.Body).Decode(&batchJob); err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to decode response").WithCause(err)
	}

	return c.convertBatchJob(&batchJob, ""), nil
}

// GetBatchResults retrieves the results of a completed batch job.
func (c *Client) GetBatchResults(ctx context.Context, batchID string) ([]provider.BatchResult, error) {
	job, err := c.GetBatch(ctx, batchID)
	if err != nil {
		return nil, err
	}

	// Check if job is complete
	if job.Status != provider.BatchStatusCompleted {
		return nil, errors.ErrInvalidRequest(fmt.Sprintf("batch job is not complete, status: %s", job.Status)).WithProvider(types.ProviderVertex)
	}

	// Get the batch job again to access internal response data
	batchName := batchID
	if !strings.HasPrefix(batchID, "projects/") {
		batchName = fmt.Sprintf("projects/%s/locations/%s/batchPredictionJobs/%s",
			c.projectID, c.location, batchID)
	}

	url := fmt.Sprintf("%s/%s", c.baseURL, batchName)
	if c.config.AccessToken == "" && c.config.APIKey != "" {
		url += "?key=" + c.config.APIKey
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
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

	var batchJob googleProvider.BatchJob
	if err := json.NewDecoder(resp.Body).Decode(&batchJob); err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to decode response").WithCause(err)
	}

	// Check for inline responses
	if batchJob.Response != nil && batchJob.Response.InlinedResponses != nil && len(batchJob.Response.InlinedResponses.InlinedResponses) > 0 {
		return c.convertInlinedResponses(batchJob.Response.InlinedResponses.InlinedResponses), nil
	}

	// Check for file-based responses
	if batchJob.Response != nil && batchJob.Response.ResponsesFile != "" {
		return c.downloadBatchResults(ctx, batchJob.Response.ResponsesFile)
	}

	return nil, errors.ErrServerError(types.ProviderVertex, "no results found in batch response")
}

// downloadBatchResults downloads and parses results from a file.
func (c *Client) downloadBatchResults(ctx context.Context, fileName string) ([]provider.BatchResult, error) {
	// For Vertex AI, the download URL follows the Vertex AI pattern
	url := fmt.Sprintf("%s/%s:download?alt=media", c.baseURL, fileName)
	if c.config.AccessToken == "" && c.config.APIKey != "" {
		url += "&key=" + c.config.APIKey
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to create download request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.ErrProviderUnavailable(types.ProviderVertex, "download failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.handleErrorResponse(resp)
	}

	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to read response").WithCause(err)
	}

	// Parse JSONL output
	var results []provider.BatchResult
	decoder := json.NewDecoder(bytes.NewReader(content))

	for decoder.More() {
		var line googleProvider.InlinedResponse
		if err := decoder.Decode(&line); err != nil {
			continue
		}

		result := provider.BatchResult{}
		if line.Metadata != nil {
			result.CustomID = line.Metadata.Key
		}

		if line.Error != nil {
			result.Error = errors.ErrServerError(types.ProviderVertex, line.Error.Message)
		} else if line.Response != nil {
			result.Response = c.transformer.TransformResponse(line.Response)
			if result.Response != nil {
				result.Response.Provider = types.ProviderVertex
			}
		}

		results = append(results, result)
	}

	return results, nil
}

// convertInlinedResponses converts inline responses to provider batch results.
func (c *Client) convertInlinedResponses(responses []googleProvider.InlinedResponse) []provider.BatchResult {
	results := make([]provider.BatchResult, len(responses))
	for i, resp := range responses {
		results[i] = provider.BatchResult{}
		if resp.Metadata != nil {
			results[i].CustomID = resp.Metadata.Key
		}

		if resp.Error != nil {
			results[i].Error = errors.ErrServerError(types.ProviderVertex, resp.Error.Message)
		} else if resp.Response != nil {
			results[i].Response = c.transformer.TransformResponse(resp.Response)
			if results[i].Response != nil {
				results[i].Response.Provider = types.ProviderVertex
			}
		}
	}
	return results
}

// CancelBatch cancels a batch job.
func (c *Client) CancelBatch(ctx context.Context, batchID string) error {
	batchName := batchID
	if !strings.HasPrefix(batchID, "projects/") {
		batchName = fmt.Sprintf("projects/%s/locations/%s/batchPredictionJobs/%s",
			c.projectID, c.location, batchID)
	}

	url := fmt.Sprintf("%s/%s:cancel", c.baseURL, batchName)
	if c.config.AccessToken == "" && c.config.APIKey != "" {
		url += "?key=" + c.config.APIKey
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return errors.ErrProviderUnavailable(types.ProviderVertex, "request failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return c.handleErrorResponse(resp)
	}

	return nil
}

// ListBatches lists all batch jobs.
func (c *Client) ListBatches(ctx context.Context, opts *provider.ListBatchOptions) ([]provider.BatchJob, error) {
	url := fmt.Sprintf("%s/projects/%s/locations/%s/batchPredictionJobs",
		c.baseURL, c.projectID, c.location)

	separator := "?"
	if c.config.AccessToken == "" && c.config.APIKey != "" {
		url += "?key=" + c.config.APIKey
		separator = "&"
	}

	if opts != nil {
		if opts.Limit > 0 {
			url += fmt.Sprintf("%spageSize=%d", separator, opts.Limit)
			separator = "&"
		}
		if opts.After != "" {
			url += separator + "pageToken=" + opts.After
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
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

	var listResp googleProvider.BatchListResponse
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to decode response").WithCause(err)
	}

	jobs := make([]provider.BatchJob, len(listResp.Batches))
	for i, batch := range listResp.Batches {
		jobs[i] = *c.convertBatchJob(&batch, "")
	}

	return jobs, nil
}

// convertBatchJob converts Google batch job to provider batch job.
func (c *Client) convertBatchJob(batch *googleProvider.BatchJob, model string) *provider.BatchJob {
	job := &provider.BatchJob{
		ID:       batch.Name,
		Provider: types.ProviderVertex,
		Status:   c.convertBatchStatus(batch),
		Metadata: make(map[string]any),
	}

	if batch.Metadata != nil {
		job.Metadata["display_name"] = batch.Metadata.DisplayName
		job.Metadata["state"] = batch.Metadata.State

		if batch.Metadata.CreateTime != "" {
			if t, err := time.Parse(time.RFC3339, batch.Metadata.CreateTime); err == nil {
				job.CreatedAt = t.Unix()
			}
		}
	}

	if model != "" {
		job.Metadata["model"] = model
	}

	if batch.Response != nil {
		if batch.Response.ResponsesFile != "" {
			job.Metadata["responses_file"] = batch.Response.ResponsesFile
		}
		if batch.Response.InlinedResponses != nil && len(batch.Response.InlinedResponses.InlinedResponses) > 0 {
			job.RequestCounts.Total = len(batch.Response.InlinedResponses.InlinedResponses)
			job.RequestCounts.Completed = len(batch.Response.InlinedResponses.InlinedResponses)
		}
	}

	return job
}

// convertBatchStatus converts Google batch status to provider status.
func (c *Client) convertBatchStatus(batch *googleProvider.BatchJob) provider.BatchStatus {
	// Check if done first
	if batch.Done {
		if batch.Error != nil {
			return provider.BatchStatusFailed
		}
		return provider.BatchStatusCompleted
	}

	// Check metadata state - handle both JOB_STATE_* and BATCH_STATE_* prefixes
	if batch.Metadata != nil {
		switch batch.Metadata.State {
		case "JOB_STATE_PENDING", "BATCH_STATE_PENDING":
			return provider.BatchStatusPending
		case "JOB_STATE_RUNNING", "BATCH_STATE_RUNNING":
			return provider.BatchStatusInProgress
		case "JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED":
			return provider.BatchStatusCompleted
		case "JOB_STATE_FAILED", "BATCH_STATE_FAILED":
			return provider.BatchStatusFailed
		case "JOB_STATE_CANCELLED", "BATCH_STATE_CANCELLED":
			return provider.BatchStatusCancelled
		case "JOB_STATE_EXPIRED", "BATCH_STATE_EXPIRED":
			return provider.BatchStatusExpired
		}
	}

	return provider.BatchStatusPending
}

// Ensure Client implements provider.BatchProvider
var _ provider.BatchProvider = (*Client)(nil)

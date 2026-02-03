package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/Chloe199719/agent-router/pkg/errors"
	"github.com/Chloe199719/agent-router/pkg/provider"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// CreateBatch creates a new batch job.
func (c *Client) CreateBatch(ctx context.Context, requests []provider.BatchRequest) (*provider.BatchJob, error) {
	// Build batch request items
	items := make([]BatchRequestItem, len(requests))
	for i, req := range requests {
		anthReq := c.transformer.TransformRequest(req.Request)
		anthReq.Stream = false
		items[i] = BatchRequestItem{
			CustomID: req.CustomID,
			Params:   *anthReq,
		}
	}

	batchReq := BatchRequest{Requests: items}

	body, err := json.Marshal(batchReq)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal request").WithCause(err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/messages/batches", bytes.NewReader(body))
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.ErrProviderUnavailable(types.ProviderAnthropic, "request failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, c.handleErrorResponse(resp)
	}

	var batch BatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&batch); err != nil {
		return nil, errors.ErrServerError(types.ProviderAnthropic, "failed to decode response").WithCause(err)
	}

	return c.convertBatchJob(&batch), nil
}

// GetBatch retrieves the status of a batch job.
func (c *Client) GetBatch(ctx context.Context, batchID string) (*provider.BatchJob, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/v1/messages/batches/"+batchID, nil)
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

	var batch BatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&batch); err != nil {
		return nil, errors.ErrServerError(types.ProviderAnthropic, "failed to decode response").WithCause(err)
	}

	return c.convertBatchJob(&batch), nil
}

// GetBatchResults retrieves the results of a completed batch job.
func (c *Client) GetBatchResults(ctx context.Context, batchID string) ([]provider.BatchResult, error) {
	// First get the batch to get the results URL
	job, err := c.GetBatch(ctx, batchID)
	if err != nil {
		return nil, err
	}

	resultsURL, ok := job.Metadata["results_url"].(string)
	if !ok || resultsURL == "" {
		return nil, errors.ErrInvalidRequest("batch has no results URL").WithProvider(types.ProviderAnthropic)
	}

	// Download the results
	httpReq, err := http.NewRequestWithContext(ctx, "GET", resultsURL, nil)
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

	// Parse JSONL results
	var results []provider.BatchResult
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		var item BatchResultItem
		if err := json.Unmarshal(scanner.Bytes(), &item); err != nil {
			continue
		}

		result := provider.BatchResult{
			CustomID: item.CustomID,
		}

		if item.Result.Type == "succeeded" && item.Result.Message != nil {
			result.Response = c.transformer.TransformResponse(item.Result.Message)
		} else if item.Result.Error != nil {
			result.Error = errors.ErrServerError(types.ProviderAnthropic, item.Result.Error.Message)
		}

		results = append(results, result)
	}

	return results, scanner.Err()
}

// CancelBatch cancels a batch job.
func (c *Client) CancelBatch(ctx context.Context, batchID string) error {
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/messages/batches/"+batchID+"/cancel", nil)
	if err != nil {
		return errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return errors.ErrProviderUnavailable(types.ProviderAnthropic, "request failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return c.handleErrorResponse(resp)
	}

	return nil
}

// ListBatches lists all batch jobs.
func (c *Client) ListBatches(ctx context.Context, opts *provider.ListBatchOptions) ([]provider.BatchJob, error) {
	url := c.baseURL + "/v1/messages/batches"
	if opts != nil {
		params := "?"
		if opts.Limit > 0 {
			params += "limit=" + strconv.Itoa(opts.Limit)
		}
		if opts.After != "" {
			if params != "?" {
				params += "&"
			}
			params += "after_id=" + opts.After
		}
		if params != "?" {
			url += params
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
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

	var list struct {
		Data    []BatchResponse `json:"data"`
		HasMore bool            `json:"has_more"`
		FirstID string          `json:"first_id"`
		LastID  string          `json:"last_id"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, errors.ErrServerError(types.ProviderAnthropic, "failed to decode response").WithCause(err)
	}

	jobs := make([]provider.BatchJob, len(list.Data))
	for i, batch := range list.Data {
		jobs[i] = *c.convertBatchJob(&batch)
	}

	return jobs, nil
}

// convertBatchJob converts Anthropic batch to provider batch job.
func (c *Client) convertBatchJob(batch *BatchResponse) *provider.BatchJob {
	job := &provider.BatchJob{
		ID:       batch.ID,
		Provider: types.ProviderAnthropic,
		Status:   c.convertBatchStatus(batch.ProcessingStatus),
		Metadata: make(map[string]any),
	}

	// Parse timestamps
	if batch.CreatedAt != "" {
		if t, err := time.Parse(time.RFC3339, batch.CreatedAt); err == nil {
			job.CreatedAt = t.Unix()
		}
	}
	if batch.EndedAt != "" {
		if t, err := time.Parse(time.RFC3339, batch.EndedAt); err == nil {
			job.CompletedAt = t.Unix()
		}
	}
	if batch.ExpiresAt != "" {
		if t, err := time.Parse(time.RFC3339, batch.ExpiresAt); err == nil {
			job.ExpiresAt = t.Unix()
		}
	}

	// Calculate totals
	total := batch.RequestCounts.Processing + batch.RequestCounts.Succeeded +
		batch.RequestCounts.Errored + batch.RequestCounts.Canceled + batch.RequestCounts.Expired
	job.RequestCounts = provider.RequestCounts{
		Total:     total,
		Completed: batch.RequestCounts.Succeeded,
		Failed:    batch.RequestCounts.Errored + batch.RequestCounts.Expired,
	}

	job.Metadata["results_url"] = batch.ResultsURL

	return job
}

// convertBatchStatus converts Anthropic batch status to provider status.
func (c *Client) convertBatchStatus(status string) provider.BatchStatus {
	switch status {
	case "in_progress":
		return provider.BatchStatusInProgress
	case "ended":
		return provider.BatchStatusCompleted
	case "canceling":
		return provider.BatchStatusInProgress
	default:
		return provider.BatchStatusPending
	}
}

// Ensure Client implements provider.BatchProvider
var _ provider.BatchProvider = (*Client)(nil)

package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"

	"github.com/Chloe199719/agent-router/pkg/errors"
	"github.com/Chloe199719/agent-router/pkg/provider"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// Batch types

// BatchCreateRequest is the request to create a batch.
type BatchCreateRequest struct {
	InputFileID      string `json:"input_file_id"`
	Endpoint         string `json:"endpoint"`
	CompletionWindow string `json:"completion_window"`
}

// BatchObject is the OpenAI batch object.
type BatchObject struct {
	ID               string         `json:"id"`
	Object           string         `json:"object"`
	Endpoint         string         `json:"endpoint"`
	Errors           *BatchErrors   `json:"errors,omitempty"`
	InputFileID      string         `json:"input_file_id"`
	CompletionWindow string         `json:"completion_window"`
	Status           string         `json:"status"`
	OutputFileID     string         `json:"output_file_id,omitempty"`
	ErrorFileID      string         `json:"error_file_id,omitempty"`
	CreatedAt        int64          `json:"created_at"`
	InProgressAt     int64          `json:"in_progress_at,omitempty"`
	ExpiresAt        int64          `json:"expires_at,omitempty"`
	FinalizingAt     int64          `json:"finalizing_at,omitempty"`
	CompletedAt      int64          `json:"completed_at,omitempty"`
	FailedAt         int64          `json:"failed_at,omitempty"`
	ExpiredAt        int64          `json:"expired_at,omitempty"`
	CancellingAt     int64          `json:"cancelling_at,omitempty"`
	CancelledAt      int64          `json:"cancelled_at,omitempty"`
	RequestCounts    *RequestCounts `json:"request_counts,omitempty"`
}

// BatchErrors contains batch-level errors.
type BatchErrors struct {
	Object string       `json:"object"`
	Data   []BatchError `json:"data"`
}

// BatchError is a single batch error.
type BatchError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
	Line    int    `json:"line,omitempty"`
}

// RequestCounts tracks batch request progress.
type RequestCounts struct {
	Total     int `json:"total"`
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
}

// BatchInputLine is a single line in the batch input file.
type BatchInputLine struct {
	CustomID string                 `json:"custom_id"`
	Method   string                 `json:"method"`
	URL      string                 `json:"url"`
	Body     map[string]interface{} `json:"body"`
}

// BatchOutputLine is a single line in the batch output file.
type BatchOutputLine struct {
	ID       string             `json:"id"`
	CustomID string             `json:"custom_id"`
	Response *BatchResponseData `json:"response"`
	Error    *APIError          `json:"error,omitempty"`
}

// BatchResponseData contains the response data.
type BatchResponseData struct {
	StatusCode int                    `json:"status_code"`
	RequestID  string                 `json:"request_id"`
	Body       ChatCompletionResponse `json:"body"`
}

// BatchList is a list of batches.
type BatchList struct {
	Object  string        `json:"object"`
	Data    []BatchObject `json:"data"`
	FirstID string        `json:"first_id"`
	LastID  string        `json:"last_id"`
	HasMore bool          `json:"has_more"`
}

// FileUploadResponse is the response from uploading a file.
type FileUploadResponse struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Bytes     int    `json:"bytes"`
	CreatedAt int64  `json:"created_at"`
	Filename  string `json:"filename"`
	Purpose   string `json:"purpose"`
}

// CreateBatch creates a new batch job.
func (c *Client) CreateBatch(ctx context.Context, requests []provider.BatchRequest) (*provider.BatchJob, error) {
	// Step 1: Create JSONL content for batch input
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)

	for _, req := range requests {
		// Transform request to OpenAI format
		oaiReq := c.transformer.TransformRequest(req.Request)
		oaiReq.Stream = false

		// Convert to generic map for body
		reqBody, err := json.Marshal(oaiReq)
		if err != nil {
			return nil, errors.ErrInvalidRequest("failed to marshal request").WithCause(err)
		}

		var body map[string]interface{}
		json.Unmarshal(reqBody, &body)

		line := BatchInputLine{
			CustomID: req.CustomID,
			Method:   "POST",
			URL:      "/v1/chat/completions",
			Body:     body,
		}

		if err := encoder.Encode(line); err != nil {
			return nil, errors.ErrInvalidRequest("failed to encode batch line").WithCause(err)
		}
	}

	// Step 2: Upload the file
	fileID, err := c.uploadBatchFile(ctx, buffer.Bytes())
	if err != nil {
		return nil, err
	}

	// Step 3: Create the batch
	createReq := BatchCreateRequest{
		InputFileID:      fileID,
		Endpoint:         "/v1/chat/completions",
		CompletionWindow: "24h",
	}

	body, err := json.Marshal(createReq)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal batch request").WithCause(err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/batches", bytes.NewReader(body))
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

	var batch BatchObject
	if err := json.NewDecoder(resp.Body).Decode(&batch); err != nil {
		return nil, errors.ErrServerError(types.ProviderOpenAI, "failed to decode response").WithCause(err)
	}

	return c.convertBatchJob(&batch), nil
}

// uploadBatchFile uploads a file for batch processing.
func (c *Client) uploadBatchFile(ctx context.Context, content []byte) (string, error) {
	// Create multipart form
	var buffer bytes.Buffer
	boundary := "----GoAgentRouterBoundary"
	buffer.WriteString("--" + boundary + "\r\n")
	buffer.WriteString("Content-Disposition: form-data; name=\"purpose\"\r\n\r\n")
	buffer.WriteString("batch\r\n")
	buffer.WriteString("--" + boundary + "\r\n")
	buffer.WriteString("Content-Disposition: form-data; name=\"file\"; filename=\"batch_input.jsonl\"\r\n")
	buffer.WriteString("Content-Type: application/jsonl\r\n\r\n")
	buffer.Write(content)
	buffer.WriteString("\r\n--" + boundary + "--\r\n")

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/files", &buffer)
	if err != nil {
		return "", errors.ErrInvalidRequest("failed to create upload request").WithCause(err)
	}

	httpReq.Header.Set("Content-Type", "multipart/form-data; boundary="+boundary)
	httpReq.Header.Set("Authorization", "Bearer "+c.config.APIKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", errors.ErrProviderUnavailable(types.ProviderOpenAI, "upload failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", c.handleErrorResponse(resp)
	}

	var fileResp FileUploadResponse
	if err := json.NewDecoder(resp.Body).Decode(&fileResp); err != nil {
		return "", errors.ErrServerError(types.ProviderOpenAI, "failed to decode upload response").WithCause(err)
	}

	return fileResp.ID, nil
}

// GetBatch retrieves the status of a batch job.
func (c *Client) GetBatch(ctx context.Context, batchID string) (*provider.BatchJob, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/batches/"+batchID, nil)
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

	var batch BatchObject
	if err := json.NewDecoder(resp.Body).Decode(&batch); err != nil {
		return nil, errors.ErrServerError(types.ProviderOpenAI, "failed to decode response").WithCause(err)
	}

	return c.convertBatchJob(&batch), nil
}

// GetBatchResults retrieves the results of a completed batch job.
func (c *Client) GetBatchResults(ctx context.Context, batchID string) ([]provider.BatchResult, error) {
	// First get the batch to get the output file ID
	job, err := c.GetBatch(ctx, batchID)
	if err != nil {
		return nil, err
	}

	outputFileID, ok := job.Metadata["output_file_id"].(string)
	if !ok || outputFileID == "" {
		return nil, errors.ErrInvalidRequest("batch has no output file").WithProvider(types.ProviderOpenAI)
	}

	// Download the output file
	httpReq, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/files/"+outputFileID+"/content", nil)
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

	// Parse JSONL output
	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.ErrServerError(types.ProviderOpenAI, "failed to read response").WithCause(err)
	}

	var results []provider.BatchResult
	decoder := json.NewDecoder(bytes.NewReader(content))

	for decoder.More() {
		var line BatchOutputLine
		if err := decoder.Decode(&line); err != nil {
			continue
		}

		result := provider.BatchResult{
			CustomID: line.CustomID,
		}

		if line.Error != nil {
			result.Error = errors.ErrServerError(types.ProviderOpenAI, line.Error.Message)
		} else if line.Response != nil {
			result.Response = c.transformer.TransformResponse(&line.Response.Body)
		}

		results = append(results, result)
	}

	return results, nil
}

// CancelBatch cancels a batch job.
func (c *Client) CancelBatch(ctx context.Context, batchID string) error {
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/batches/"+batchID+"/cancel", nil)
	if err != nil {
		return errors.ErrInvalidRequest("failed to create request").WithCause(err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return errors.ErrProviderUnavailable(types.ProviderOpenAI, "request failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return c.handleErrorResponse(resp)
	}

	return nil
}

// ListBatches lists all batch jobs.
func (c *Client) ListBatches(ctx context.Context, opts *provider.ListBatchOptions) ([]provider.BatchJob, error) {
	url := c.baseURL + "/batches"
	if opts != nil {
		params := "?"
		if opts.Limit > 0 {
			params += "limit=" + string(rune(opts.Limit))
		}
		if opts.After != "" {
			if params != "?" {
				params += "&"
			}
			params += "after=" + opts.After
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
		return nil, errors.ErrProviderUnavailable(types.ProviderOpenAI, "request failed").WithCause(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.handleErrorResponse(resp)
	}

	var list BatchList
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, errors.ErrServerError(types.ProviderOpenAI, "failed to decode response").WithCause(err)
	}

	jobs := make([]provider.BatchJob, len(list.Data))
	for i, batch := range list.Data {
		jobs[i] = *c.convertBatchJob(&batch)
	}

	return jobs, nil
}

// convertBatchJob converts OpenAI batch to provider batch job.
func (c *Client) convertBatchJob(batch *BatchObject) *provider.BatchJob {
	job := &provider.BatchJob{
		ID:        batch.ID,
		Provider:  types.ProviderOpenAI,
		Status:    c.convertBatchStatus(batch.Status),
		CreatedAt: batch.CreatedAt,
		Metadata:  make(map[string]any),
	}

	if batch.CompletedAt > 0 {
		job.CompletedAt = batch.CompletedAt
	}
	if batch.ExpiresAt > 0 {
		job.ExpiresAt = batch.ExpiresAt
	}

	if batch.RequestCounts != nil {
		job.RequestCounts = provider.RequestCounts{
			Total:     batch.RequestCounts.Total,
			Completed: batch.RequestCounts.Completed,
			Failed:    batch.RequestCounts.Failed,
		}
	}

	job.Metadata["input_file_id"] = batch.InputFileID
	job.Metadata["output_file_id"] = batch.OutputFileID
	job.Metadata["error_file_id"] = batch.ErrorFileID
	job.Metadata["endpoint"] = batch.Endpoint

	return job
}

// convertBatchStatus converts OpenAI batch status to provider status.
func (c *Client) convertBatchStatus(status string) provider.BatchStatus {
	switch status {
	case "validating":
		return provider.BatchStatusValidating
	case "in_progress":
		return provider.BatchStatusInProgress
	case "finalizing":
		return provider.BatchStatusFinalizing
	case "completed":
		return provider.BatchStatusCompleted
	case "failed":
		return provider.BatchStatusFailed
	case "expired":
		return provider.BatchStatusExpired
	case "cancelling":
		return provider.BatchStatusInProgress
	case "cancelled":
		return provider.BatchStatusCancelled
	default:
		return provider.BatchStatusPending
	}
}

// Ensure Client implements provider.BatchProvider
var _ provider.BatchProvider = (*Client)(nil)

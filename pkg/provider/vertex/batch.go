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

// CreateBatch creates a new batch prediction job using the Vertex AI batchPredictionJobs API.
//
// This method:
// 1. Transforms inline requests into JSONL format
// 2. Uploads the JSONL to a GCS staging bucket
// 3. Creates a batchPredictionJob referencing the GCS input
//
// Requires BatchBucket to be configured via provider.WithBatchBucket().
func (c *Client) CreateBatch(ctx context.Context, requests []provider.BatchRequest) (*provider.BatchJob, error) {
	if len(requests) == 0 {
		return nil, errors.ErrInvalidRequest("no requests provided").WithProvider(types.ProviderVertex)
	}

	if c.config.BatchBucket == "" {
		return nil, errors.ErrInvalidRequest("batch bucket is required for Vertex AI batch operations; use provider.WithBatchBucket()").WithProvider(types.ProviderVertex)
	}

	// Get model from first request (all requests should use the same model)
	model := requests[0].Request.Model
	if model == "" {
		model = "gemini-2.0-flash" // Default model
	}

	// Build JSONL content from requests
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	for _, req := range requests {
		gReq := c.transformer.TransformRequest(req.Request)
		line := VertexBatchInputLine{
			Request: gReq,
		}
		if err := encoder.Encode(line); err != nil {
			return nil, errors.ErrInvalidRequest("failed to marshal batch request line").WithCause(err)
		}
	}

	// Upload JSONL to GCS
	batchID := fmt.Sprintf("batch-%d", time.Now().UnixNano())
	bucket, prefix := parseBucketPath(c.config.BatchBucket)
	inputPath := fmt.Sprintf("%s%s/input.jsonl", prefix, batchID)
	inputURI := fmt.Sprintf("gs://%s/%s", bucket, inputPath)
	outputURIPrefix := fmt.Sprintf("gs://%s/%s%s/output/", bucket, prefix, batchID)

	if err := c.uploadToGCS(ctx, bucket, inputPath, buf.Bytes()); err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to upload batch input to GCS").WithCause(err)
	}

	// Create batch prediction job
	modelPath := fmt.Sprintf("publishers/google/models/%s", model)

	jobReq := &VertexBatchPredictionJobRequest{
		DisplayName: batchID,
		Model:       modelPath,
		InputConfig: &VertexBatchInputConfig{
			InstancesFormat: "jsonl",
			GcsSource: &GcsSource{
				URIs: []string{inputURI},
			},
		},
		OutputConfig: &VertexBatchOutputConfig{
			PredictionsFormat: "jsonl",
			GcsDestination: &GcsDestination{
				OutputURIPrefix: outputURIPrefix,
			},
		},
	}

	body, err := json.Marshal(jobReq)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal batch prediction job request").WithCause(err)
	}

	url := c.batchJobsURL()
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

	var job VertexBatchPredictionJob
	if err := json.NewDecoder(resp.Body).Decode(&job); err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to decode response").WithCause(err)
	}

	return c.convertVertexBatchJob(&job, model), nil
}

// GetBatch retrieves the status of a batch prediction job.
func (c *Client) GetBatch(ctx context.Context, batchID string) (*provider.BatchJob, error) {
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

	var job VertexBatchPredictionJob
	if err := json.NewDecoder(resp.Body).Decode(&job); err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to decode response").WithCause(err)
	}

	return c.convertVertexBatchJob(&job, ""), nil
}

// GetBatchResults retrieves the results of a completed batch prediction job.
func (c *Client) GetBatchResults(ctx context.Context, batchID string) ([]provider.BatchResult, error) {
	job, err := c.GetBatch(ctx, batchID)
	if err != nil {
		return nil, err
	}

	if job.Status != provider.BatchStatusCompleted {
		return nil, errors.ErrInvalidRequest(fmt.Sprintf("batch job is not complete, status: %s", job.Status)).WithProvider(types.ProviderVertex)
	}

	// Get output directory from metadata
	outputDir, ok := job.Metadata["gcs_output_directory"].(string)
	if !ok || outputDir == "" {
		return nil, errors.ErrServerError(types.ProviderVertex, "no output directory found in batch job")
	}

	// Download and parse results from GCS
	return c.downloadBatchResults(ctx, outputDir)
}

// downloadBatchResults downloads and parses JSONL results from a GCS output directory.
func (c *Client) downloadBatchResults(ctx context.Context, gcsOutputDir string) ([]provider.BatchResult, error) {
	// The output directory contains prediction results in JSONL format.
	// Typically the file is at: {outputDir}/predictions.jsonl or similar.
	// We try the common pattern first.
	outputURI := strings.TrimSuffix(gcsOutputDir, "/") + "/predictions.jsonl"

	bucket, objectPath := parseGCSURI(outputURI)
	if bucket == "" {
		return nil, errors.ErrServerError(types.ProviderVertex, "invalid GCS output URI: "+outputURI)
	}

	content, err := c.downloadFromGCS(ctx, bucket, objectPath)
	if err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to download batch results from GCS").WithCause(err)
	}

	// Parse JSONL output - each line contains a prediction result
	var results []provider.BatchResult
	decoder := json.NewDecoder(bytes.NewReader(content))

	for decoder.More() {
		var line struct {
			Response *googleProvider.GenerateContentResponse `json:"response,omitempty"`
			Status   string                                  `json:"status,omitempty"`
		}
		if err := decoder.Decode(&line); err != nil {
			continue
		}

		result := provider.BatchResult{}
		if line.Response != nil {
			result.Response = c.transformer.TransformResponse(line.Response)
			if result.Response != nil {
				result.Response.Provider = types.ProviderVertex
			}
		}

		results = append(results, result)
	}

	return results, nil
}

// CancelBatch cancels a batch prediction job.
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

// ListBatches lists batch prediction jobs.
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

	var listResp VertexBatchPredictionJobList
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to decode response").WithCause(err)
	}

	jobs := make([]provider.BatchJob, len(listResp.BatchPredictionJobs))
	for i, job := range listResp.BatchPredictionJobs {
		jobs[i] = *c.convertVertexBatchJob(&job, "")
	}

	return jobs, nil
}

// batchJobsURL returns the URL for the batchPredictionJobs endpoint.
func (c *Client) batchJobsURL() string {
	url := fmt.Sprintf("%s/projects/%s/locations/%s/batchPredictionJobs",
		c.baseURL, c.projectID, c.location)

	if c.config.AccessToken == "" && c.config.APIKey != "" {
		url += "?key=" + c.config.APIKey
	}

	return url
}

// uploadToGCS uploads data to a GCS bucket using the JSON API.
func (c *Client) uploadToGCS(ctx context.Context, bucket, objectPath string, data []byte) error {
	url := fmt.Sprintf("https://storage.googleapis.com/upload/storage/v1/b/%s/o?uploadType=media&name=%s",
		bucket, objectPath)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("create upload request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/jsonl")
	if c.config.AccessToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.config.AccessToken)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("upload request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("GCS upload failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// downloadFromGCS downloads an object from GCS using the JSON API.
func (c *Client) downloadFromGCS(ctx context.Context, bucket, objectPath string) ([]byte, error) {
	url := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o/%s?alt=media",
		bucket, objectPath)

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create download request: %w", err)
	}

	if c.config.AccessToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.config.AccessToken)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("download request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("GCS download failed with status %d: %s", resp.StatusCode, string(body))
	}

	return io.ReadAll(resp.Body)
}

// convertVertexBatchJob converts a Vertex AI batch prediction job to a provider BatchJob.
func (c *Client) convertVertexBatchJob(job *VertexBatchPredictionJob, model string) *provider.BatchJob {
	result := &provider.BatchJob{
		ID:       job.Name,
		Provider: types.ProviderVertex,
		Status:   c.convertVertexJobState(job.State),
		Metadata: make(map[string]any),
	}

	if job.DisplayName != "" {
		result.Metadata["display_name"] = job.DisplayName
	}
	if job.State != "" {
		result.Metadata["state"] = job.State
	}
	if model != "" {
		result.Metadata["model"] = model
	} else if job.Model != "" {
		result.Metadata["model"] = job.Model
	}
	if job.OutputInfo != nil && job.OutputInfo.GcsOutputDirectory != "" {
		result.Metadata["gcs_output_directory"] = job.OutputInfo.GcsOutputDirectory
	}

	if job.CreateTime != "" {
		if t, err := time.Parse(time.RFC3339, job.CreateTime); err == nil {
			result.CreatedAt = t.Unix()
		}
	}
	if job.EndTime != "" {
		if t, err := time.Parse(time.RFC3339, job.EndTime); err == nil {
			result.CompletedAt = t.Unix()
		}
	}

	if job.Error != nil {
		result.Metadata["error_code"] = job.Error.Code
		result.Metadata["error_message"] = job.Error.Message
	}

	return result
}

// convertVertexJobState converts a Vertex AI job state to a provider BatchStatus.
func (c *Client) convertVertexJobState(state string) provider.BatchStatus {
	switch state {
	case "JOB_STATE_PENDING", "JOB_STATE_QUEUED":
		return provider.BatchStatusPending
	case "JOB_STATE_RUNNING", "JOB_STATE_UPDATING":
		return provider.BatchStatusInProgress
	case "JOB_STATE_SUCCEEDED":
		return provider.BatchStatusCompleted
	case "JOB_STATE_FAILED", "JOB_STATE_PARTIALLY_SUCCEEDED":
		return provider.BatchStatusFailed
	case "JOB_STATE_CANCELLED", "JOB_STATE_CANCELLING":
		return provider.BatchStatusCancelled
	case "JOB_STATE_EXPIRED":
		return provider.BatchStatusExpired
	default:
		return provider.BatchStatusPending
	}
}

// parseBucketPath splits a bucket config like "my-bucket/staging/path" into bucket and prefix.
func parseBucketPath(bucketPath string) (bucket, prefix string) {
	parts := strings.SplitN(bucketPath, "/", 2)
	bucket = parts[0]
	if len(parts) > 1 {
		prefix = parts[1]
		if !strings.HasSuffix(prefix, "/") {
			prefix += "/"
		}
	}
	return
}

// parseGCSURI parses a gs:// URI into bucket and object path.
func parseGCSURI(uri string) (bucket, objectPath string) {
	if !strings.HasPrefix(uri, "gs://") {
		return "", ""
	}
	path := strings.TrimPrefix(uri, "gs://")
	parts := strings.SplitN(path, "/", 2)
	if len(parts) != 2 {
		return parts[0], ""
	}
	return parts[0], parts[1]
}

// Ensure Client implements provider.BatchProvider
var _ provider.BatchProvider = (*Client)(nil)

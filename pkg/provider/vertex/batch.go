package vertex

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
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
// 3. Uploads a custom_ids mapping file for result correlation
// 4. Creates a batchPredictionJob referencing the GCS input
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

	// Build JSONL content from requests and collect custom IDs
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	customIDs := make([]string, 0, len(requests))
	for _, req := range requests {
		gReq := c.transformer.TransformRequest(req.Request)
		line := VertexBatchInputLine{
			Request: gReq,
		}
		if err := encoder.Encode(line); err != nil {
			return nil, errors.ErrInvalidRequest("failed to marshal batch request line").WithCause(err)
		}
		customIDs = append(customIDs, req.CustomID)
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

	// Upload custom_ids mapping file so we can correlate results back to requests.
	// Vertex AI preserves input line order in output, so index-based mapping works.
	customIDsPath := fmt.Sprintf("%s%s/custom_ids.json", prefix, batchID)
	customIDsData, err := json.Marshal(customIDs)
	if err != nil {
		return nil, errors.ErrInvalidRequest("failed to marshal custom IDs").WithCause(err)
	}
	if err := c.uploadToGCS(ctx, bucket, customIDsPath, customIDsData); err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to upload custom IDs mapping to GCS").WithCause(err)
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

	result := c.convertVertexBatchJob(&job, model)
	result.Metadata["custom_ids_uri"] = fmt.Sprintf("gs://%s/%s", bucket, customIDsPath)

	return result, nil
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
	results, err := c.downloadBatchResults(ctx, outputDir)
	if err != nil {
		return nil, err
	}

	// Try to load custom_ids mapping to correlate results back to requests.
	// The mapping is stored alongside the input when the batch was created.
	displayName, _ := job.Metadata["display_name"].(string)
	customIDs := c.loadCustomIDs(ctx, displayName)
	if len(customIDs) > 0 {
		for i := range results {
			if i < len(customIDs) {
				results[i].CustomID = customIDs[i]
			}
		}
	}

	return results, nil
}

// downloadBatchResults downloads and parses JSONL results from a GCS output directory.
func (c *Client) downloadBatchResults(ctx context.Context, gcsOutputDir string) ([]provider.BatchResult, error) {
	// Vertex AI writes output files with dynamic names like "prediction-model-<timestamp>"
	// so we need to list the output directory to find the actual result file.
	bucket, prefix := parseGCSURI(strings.TrimSuffix(gcsOutputDir, "/") + "/")
	if bucket == "" {
		return nil, errors.ErrServerError(types.ProviderVertex, "invalid GCS output URI: "+gcsOutputDir)
	}

	// List objects in the output directory to find the result file
	objectPath, err := c.findBatchOutputFile(ctx, bucket, prefix)
	if err != nil {
		return nil, errors.ErrServerError(types.ProviderVertex, "failed to find batch output file in GCS").WithCause(err)
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

// loadCustomIDs attempts to load the custom_ids.json mapping file from GCS.
// The mapping file is stored at gs://{bucket}/{prefix}{displayName}/custom_ids.json
// when CreateBatch is called. It returns nil if the mapping cannot be loaded
// (e.g., batch was created externally or the bucket is not configured).
func (c *Client) loadCustomIDs(ctx context.Context, displayName string) []string {
	if c.config.BatchBucket == "" || displayName == "" {
		return nil
	}

	bucket, prefix := parseBucketPath(c.config.BatchBucket)
	mappingPath := fmt.Sprintf("%s%s/custom_ids.json", prefix, displayName)

	data, err := c.downloadFromGCS(ctx, bucket, mappingPath)
	if err != nil {
		return nil
	}

	var ids []string
	if json.Unmarshal(data, &ids) != nil {
		return nil
	}

	return ids
}

// findBatchOutputFile lists objects in a GCS directory and returns the path of the prediction output file.
func (c *Client) findBatchOutputFile(ctx context.Context, bucket, prefix string) (string, error) {
	listURL := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o?prefix=%s",
		bucket, url.QueryEscape(prefix))

	httpReq, err := http.NewRequestWithContext(ctx, "GET", listURL, nil)
	if err != nil {
		return "", fmt.Errorf("create list request: %w", err)
	}

	if c.config.AccessToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.config.AccessToken)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("list request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("GCS list failed with status %d: %s", resp.StatusCode, string(body))
	}

	var listResp struct {
		Items []struct {
			Name string `json:"name"`
		} `json:"items"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return "", fmt.Errorf("decode list response: %w", err)
	}

	// Look for a prediction output file (Vertex AI names them "prediction-model-<timestamp>")
	for _, item := range listResp.Items {
		baseName := item.Name
		if idx := strings.LastIndex(baseName, "/"); idx >= 0 {
			baseName = baseName[idx+1:]
		}
		if strings.HasPrefix(baseName, "prediction") {
			return item.Name, nil
		}
	}

	if len(listResp.Items) > 0 {
		// Fall back to the first file found in the directory
		return listResp.Items[0].Name, nil
	}

	return "", fmt.Errorf("no output files found in GCS directory: gs://%s/%s", bucket, prefix)
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
	// The GCS JSON API requires the object path to be URL-encoded
	encodedPath := url.PathEscape(objectPath)
	downloadURL := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o/%s?alt=media",
		bucket, encodedPath)

	httpReq, err := http.NewRequestWithContext(ctx, "GET", downloadURL, nil)
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
// It also handles the gs:// URI format (e.g., "gs://my-bucket/staging/path").
func parseBucketPath(bucketPath string) (bucket, prefix string) {
	// Strip gs:// prefix if present
	bucketPath = strings.TrimPrefix(bucketPath, "gs://")

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

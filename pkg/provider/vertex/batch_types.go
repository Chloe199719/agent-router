package vertex

import (
	googleProvider "github.com/Chloe199719/agent-router/pkg/provider/google"
)

// Vertex AI Batch Prediction Jobs API types.
// See: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-from-cloud-storage

// VertexBatchPredictionJobRequest is the request body for creating a batch prediction job.
type VertexBatchPredictionJobRequest struct {
	DisplayName  string                   `json:"displayName"`
	Model        string                   `json:"model"`
	InputConfig  *VertexBatchInputConfig  `json:"inputConfig"`
	OutputConfig *VertexBatchOutputConfig `json:"outputConfig"`
}

// VertexBatchInputConfig specifies the input source for a batch job.
type VertexBatchInputConfig struct {
	InstancesFormat string     `json:"instancesFormat"`
	GcsSource       *GcsSource `json:"gcsSource"`
}

// GcsSource specifies GCS URIs as input.
type GcsSource struct {
	URIs []string `json:"uris"`
}

// VertexBatchOutputConfig specifies the output destination for a batch job.
type VertexBatchOutputConfig struct {
	PredictionsFormat string          `json:"predictionsFormat"`
	GcsDestination    *GcsDestination `json:"gcsDestination"`
}

// GcsDestination specifies a GCS URI prefix for output.
type GcsDestination struct {
	OutputURIPrefix string `json:"outputUriPrefix"`
}

// VertexBatchPredictionJob is the response from creating or getting a batch prediction job.
type VertexBatchPredictionJob struct {
	Name         string                   `json:"name"`
	DisplayName  string                   `json:"displayName"`
	Model        string                   `json:"model"`
	State        string                   `json:"state"`
	InputConfig  *VertexBatchInputConfig  `json:"inputConfig,omitempty"`
	OutputConfig *VertexBatchOutputConfig `json:"outputConfig,omitempty"`
	OutputInfo   *VertexBatchOutputInfo   `json:"outputInfo,omitempty"`
	Error        *VertexRpcStatus         `json:"error,omitempty"`
	CreateTime   string                   `json:"createTime,omitempty"`
	StartTime    string                   `json:"startTime,omitempty"`
	EndTime      string                   `json:"endTime,omitempty"`
	UpdateTime   string                   `json:"updateTime,omitempty"`
}

// VertexBatchOutputInfo contains the output information after job completion.
type VertexBatchOutputInfo struct {
	GcsOutputDirectory string `json:"gcsOutputDirectory,omitempty"`
}

// VertexRpcStatus is a gRPC status error.
type VertexRpcStatus struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// VertexBatchPredictionJobList is the response from listing batch prediction jobs.
type VertexBatchPredictionJobList struct {
	BatchPredictionJobs []VertexBatchPredictionJob `json:"batchPredictionJobs,omitempty"`
	NextPageToken       string                     `json:"nextPageToken,omitempty"`
}

// VertexBatchInputLine is a single line in the JSONL input file.
// The Vertex AI batch API expects each line to have a "request" field
// matching the Gemini API generateContent request format.
type VertexBatchInputLine struct {
	Request any `json:"request"`
}

// VertexBatchOutputLine is a single line in the JSONL output file.
// Vertex AI echoes the original request back alongside the response,
// allowing us to extract custom_id from the request's labels.
type VertexBatchOutputLine struct {
	Request       *VertexBatchOutputRequest               `json:"request,omitempty"`
	Response      *googleProvider.GenerateContentResponse `json:"response,omitempty"`
	Status        string                                  `json:"status,omitempty"`
	ProcessedTime string                                  `json:"processed_time,omitempty"`
}

// VertexBatchOutputRequest is the echoed request in the batch output.
// We only need to parse the labels field to extract custom_id.
type VertexBatchOutputRequest struct {
	Labels map[string]string `json:"labels,omitempty"`
}

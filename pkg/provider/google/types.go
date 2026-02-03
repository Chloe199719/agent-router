package google

// GenerateContentRequest is the Google Gemini API request.
type GenerateContentRequest struct {
	Contents          []Content         `json:"contents"`
	SystemInstruction *Content          `json:"systemInstruction,omitempty"`
	GenerationConfig  *GenerationConfig `json:"generationConfig,omitempty"`
	SafetySettings    []SafetySetting   `json:"safetySettings,omitempty"`
	Tools             []Tool            `json:"tools,omitempty"`
	ToolConfig        *ToolConfig       `json:"toolConfig,omitempty"`
}

// Content is a content message.
type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

// Part is a content part.
type Part struct {
	Text             string            `json:"text,omitempty"`
	InlineData       *InlineData       `json:"inlineData,omitempty"`
	FileData         *FileData         `json:"fileData,omitempty"`
	FunctionCall     *FunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
}

// InlineData is inline binary data (images, etc).
type InlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64 encoded
}

// FileData is a reference to a file.
type FileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

// FunctionCall is a function call from the model.
type FunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

// FunctionResponse is a function response from the user.
type FunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

// GenerationConfig configures generation parameters.
type GenerationConfig struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"topP,omitempty"`
	TopK             *int     `json:"topK,omitempty"`
	MaxOutputTokens  *int     `json:"maxOutputTokens,omitempty"`
	StopSequences    []string `json:"stopSequences,omitempty"`
	CandidateCount   *int     `json:"candidateCount,omitempty"`
	ResponseMimeType string   `json:"responseMimeType,omitempty"`
	ResponseSchema   *Schema  `json:"responseSchema,omitempty"`
}

// Schema is Google's schema format.
type Schema struct {
	Type        string             `json:"type"`
	Description string             `json:"description,omitempty"`
	Enum        []string           `json:"enum,omitempty"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	Items       *Schema            `json:"items,omitempty"`
	Nullable    bool               `json:"nullable,omitempty"`
}

// SafetySetting configures safety thresholds.
type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// Tool is a Google tool definition.
type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations,omitempty"`
}

// FunctionDeclaration declares a function.
type FunctionDeclaration struct {
	Name        string  `json:"name"`
	Description string  `json:"description,omitempty"`
	Parameters  *Schema `json:"parameters,omitempty"`
}

// ToolConfig configures tool usage.
type ToolConfig struct {
	FunctionCallingConfig *FunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// FunctionCallingConfig configures function calling.
type FunctionCallingConfig struct {
	Mode                 string   `json:"mode"` // "AUTO", "ANY", "NONE"
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// GenerateContentResponse is the response from generateContent.
type GenerateContentResponse struct {
	Candidates     []Candidate     `json:"candidates"`
	PromptFeedback *PromptFeedback `json:"promptFeedback,omitempty"`
	UsageMetadata  *UsageMetadata  `json:"usageMetadata,omitempty"`
}

// Candidate is a response candidate.
type Candidate struct {
	Content       *Content       `json:"content"`
	FinishReason  string         `json:"finishReason"`
	Index         int            `json:"index"`
	SafetyRatings []SafetyRating `json:"safetyRatings,omitempty"`
}

// SafetyRating is a safety rating for content.
type SafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
}

// PromptFeedback is feedback about the prompt.
type PromptFeedback struct {
	BlockReason   string         `json:"blockReason,omitempty"`
	SafetyRatings []SafetyRating `json:"safetyRatings,omitempty"`
}

// UsageMetadata contains usage information.
type UsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// StreamChunk is a streaming response chunk.
type StreamChunk struct {
	Candidates     []Candidate     `json:"candidates"`
	UsageMetadata  *UsageMetadata  `json:"usageMetadata,omitempty"`
	PromptFeedback *PromptFeedback `json:"promptFeedback,omitempty"`
}

// ErrorResponse is a Google API error response.
type ErrorResponse struct {
	Error *APIError `json:"error"`
}

// APIError is a Google API error.
type APIError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// Batch API types

// BatchGenerateContentRequest is the request to create a batch job.
type BatchGenerateContentRequest struct {
	Batch *BatchConfig `json:"batch"`
}

// BatchConfig configures a batch job.
type BatchConfig struct {
	DisplayName string       `json:"display_name,omitempty"`
	InputConfig *InputConfig `json:"input_config"`
}

// InputConfig specifies the input for a batch job.
type InputConfig struct {
	Requests *RequestsInput `json:"requests,omitempty"`
	FileName string         `json:"file_name,omitempty"`
}

// RequestsInput contains inline requests.
type RequestsInput struct {
	Requests []BatchRequestItem `json:"requests"`
}

// BatchRequestItem is a single request in a batch.
type BatchRequestItem struct {
	Request  *GenerateContentRequest `json:"request"`
	Metadata *RequestMetadata        `json:"metadata,omitempty"`
}

// RequestMetadata contains metadata for a batch request.
type RequestMetadata struct {
	Key string `json:"key"`
}

// BatchJob represents a batch job (long-running operation).
type BatchJob struct {
	Name     string         `json:"name"`
	Metadata *BatchMetadata `json:"metadata,omitempty"`
	Done     bool           `json:"done"`
	Error    *StatusError   `json:"error,omitempty"`
	Response *BatchResponse `json:"response,omitempty"`
}

// BatchMetadata contains batch job metadata.
type BatchMetadata struct {
	Type        string `json:"@type,omitempty"`
	DisplayName string `json:"displayName,omitempty"`
	State       string `json:"state"`
	CreateTime  string `json:"createTime,omitempty"`
	UpdateTime  string `json:"updateTime,omitempty"`
}

// StatusError is an error status.
type StatusError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// BatchResponse is the response from a completed batch job.
type BatchResponse struct {
	Type             string            `json:"@type,omitempty"`
	InlinedResponses []InlinedResponse `json:"inlinedResponses,omitempty"`
	ResponsesFile    string            `json:"responsesFile,omitempty"`
}

// InlinedResponse is an inline response from a batch job.
type InlinedResponse struct {
	Key      string                   `json:"key,omitempty"`
	Response *GenerateContentResponse `json:"response,omitempty"`
	Error    *StatusError             `json:"error,omitempty"`
}

// BatchListResponse is the response from listing batches.
type BatchListResponse struct {
	Batches       []BatchJob `json:"batches,omitempty"`
	NextPageToken string     `json:"nextPageToken,omitempty"`
}

// FileUploadResponse is the response from uploading a file.
type FileUploadResponse struct {
	File *UploadedFile `json:"file"`
}

// UploadedFile represents an uploaded file.
type UploadedFile struct {
	Name        string `json:"name"`
	DisplayName string `json:"displayName,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
	SizeBytes   string `json:"sizeBytes,omitempty"`
	CreateTime  string `json:"createTime,omitempty"`
	URI         string `json:"uri,omitempty"`
}

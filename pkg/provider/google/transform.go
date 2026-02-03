package google

import (
	"encoding/json"
	"time"

	"github.com/Chloe199719/agent-router/pkg/schema"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// Transformer handles conversion between unified and Google formats.
type Transformer struct {
	schemaTranslator *schema.Translator
}

// NewTransformer creates a new transformer.
func NewTransformer() *Transformer {
	return &Transformer{
		schemaTranslator: schema.NewTranslator(),
	}
}

// TransformRequest converts a unified request to Google format.
func (t *Transformer) TransformRequest(req *types.CompletionRequest) *GenerateContentRequest {
	gReq := &GenerateContentRequest{}

	// Transform messages
	contents, systemInstruction := t.transformMessages(req.Messages)
	gReq.Contents = contents
	if systemInstruction != nil {
		gReq.SystemInstruction = systemInstruction
	}

	// Build generation config
	genConfig := &GenerationConfig{
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
	}

	if req.MaxTokens != nil {
		genConfig.MaxOutputTokens = req.MaxTokens
	}

	if len(req.StopSequences) > 0 {
		genConfig.StopSequences = req.StopSequences
	}

	// Transform response format
	if req.ResponseFormat != nil {
		t.applyResponseFormat(genConfig, req.ResponseFormat)
	}

	gReq.GenerationConfig = genConfig

	// Transform tools
	if len(req.Tools) > 0 {
		gReq.Tools = t.transformTools(req.Tools)
	}

	// Transform tool choice
	if req.ToolChoice != nil {
		gReq.ToolConfig = t.transformToolChoice(req.ToolChoice)
	}

	return gReq
}

// transformMessages converts unified messages to Google format.
func (t *Transformer) transformMessages(messages []types.Message) ([]Content, *Content) {
	var contents []Content
	var systemInstruction *Content

	for _, msg := range messages {
		// Handle system messages
		if msg.Role == types.RoleSystem {
			var parts []Part
			for _, block := range msg.Content {
				if block.Type == types.ContentTypeText {
					parts = append(parts, Part{Text: block.Text})
				}
			}
			if len(parts) > 0 {
				systemInstruction = &Content{Parts: parts}
			}
			continue
		}

		content := Content{
			Role:  t.mapRole(msg.Role),
			Parts: t.transformParts(msg.Content),
		}

		contents = append(contents, content)
	}

	return contents, systemInstruction
}

// mapRole maps unified role to Google role.
func (t *Transformer) mapRole(role types.Role) string {
	switch role {
	case types.RoleUser, types.RoleTool:
		return "user"
	case types.RoleAssistant:
		return "model"
	default:
		return "user"
	}
}

// transformParts converts unified content blocks to Google parts.
func (t *Transformer) transformParts(blocks []types.ContentBlock) []Part {
	var parts []Part

	for _, block := range blocks {
		switch block.Type {
		case types.ContentTypeText:
			parts = append(parts, Part{Text: block.Text})

		case types.ContentTypeImage:
			if block.ImageBase64 != "" {
				parts = append(parts, Part{
					InlineData: &InlineData{
						MimeType: block.MediaType,
						Data:     block.ImageBase64,
					},
				})
			} else if block.ImageURL != "" {
				parts = append(parts, Part{
					FileData: &FileData{
						MimeType: block.MediaType,
						FileURI:  block.ImageURL,
					},
				})
			}

		case types.ContentTypeToolUse:
			args, _ := block.ToolInput.(map[string]any)
			parts = append(parts, Part{
				FunctionCall: &FunctionCall{
					Name: block.ToolName,
					Args: args,
				},
			})

		case types.ContentTypeToolResult:
			// Parse result as JSON if possible
			var response map[string]any
			if err := json.Unmarshal([]byte(block.Text), &response); err != nil {
				response = map[string]any{"result": block.Text}
			}
			parts = append(parts, Part{
				FunctionResponse: &FunctionResponse{
					Name:     block.ToolName,
					Response: response,
				},
			})
		}
	}

	return parts
}

// applyResponseFormat applies response format to generation config.
func (t *Transformer) applyResponseFormat(config *GenerationConfig, rf *types.ResponseFormat) {
	googleConfig := t.schemaTranslator.ToGoogle(rf)
	if googleConfig == nil {
		return
	}

	config.ResponseMimeType = googleConfig.ResponseMimeType
	if googleConfig.ResponseSchema != nil {
		config.ResponseSchema = t.convertGoogleSchema(googleConfig.ResponseSchema)
	}
}

// convertGoogleSchema converts schema translator format to local format.
func (t *Transformer) convertGoogleSchema(s *schema.GoogleSchema) *Schema {
	if s == nil {
		return nil
	}

	gs := &Schema{
		Type:        s.Type,
		Description: s.Description,
		Enum:        s.Enum,
		Required:    s.Required,
		Nullable:    s.Nullable,
	}

	if len(s.Properties) > 0 {
		gs.Properties = make(map[string]*Schema)
		for name, prop := range s.Properties {
			gs.Properties[name] = t.convertGoogleSchema(prop)
		}
	}

	if s.Items != nil {
		gs.Items = t.convertGoogleSchema(s.Items)
	}

	return gs
}

// transformTools converts unified tools to Google format.
func (t *Transformer) transformTools(tools []types.Tool) []Tool {
	googleTool := t.schemaTranslator.ToolsToGoogle(tools)
	if googleTool == nil {
		return nil
	}

	var declarations []FunctionDeclaration
	for _, decl := range googleTool.FunctionDeclarations {
		fd := FunctionDeclaration{
			Name:        decl.Name,
			Description: decl.Description,
		}
		if decl.Parameters != nil {
			fd.Parameters = t.convertGoogleSchema(decl.Parameters)
		}
		declarations = append(declarations, fd)
	}

	return []Tool{{FunctionDeclarations: declarations}}
}

// transformToolChoice converts unified tool choice to Google format.
func (t *Transformer) transformToolChoice(tc *types.ToolChoice) *ToolConfig {
	config := &ToolConfig{
		FunctionCallingConfig: &FunctionCallingConfig{},
	}

	switch tc.Type {
	case types.ToolChoiceAuto:
		config.FunctionCallingConfig.Mode = "AUTO"
	case types.ToolChoiceRequired:
		config.FunctionCallingConfig.Mode = "ANY"
	case types.ToolChoiceNone:
		config.FunctionCallingConfig.Mode = "NONE"
	case types.ToolChoiceTool:
		config.FunctionCallingConfig.Mode = "ANY"
		config.FunctionCallingConfig.AllowedFunctionNames = []string{tc.Name}
	default:
		config.FunctionCallingConfig.Mode = "AUTO"
	}

	return config
}

// TransformResponse converts Google response to unified format.
func (t *Transformer) TransformResponse(resp *GenerateContentResponse) *types.CompletionResponse {
	if resp == nil || len(resp.Candidates) == 0 {
		return nil
	}

	candidate := resp.Candidates[0]
	result := &types.CompletionResponse{
		Provider:   types.ProviderGoogle,
		Content:    t.transformResponseContent(candidate.Content),
		StopReason: t.transformStopReason(candidate.FinishReason),
		ToolCalls:  t.extractToolCalls(candidate.Content),
		CreatedAt:  time.Now(),
	}

	if resp.UsageMetadata != nil {
		result.Usage = types.Usage{
			InputTokens:  resp.UsageMetadata.PromptTokenCount,
			OutputTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:  resp.UsageMetadata.TotalTokenCount,
		}
	}

	return result
}

// transformResponseContent converts Google content to unified format.
func (t *Transformer) transformResponseContent(content *Content) []types.ContentBlock {
	if content == nil {
		return nil
	}

	var blocks []types.ContentBlock

	for _, part := range content.Parts {
		if part.Text != "" {
			blocks = append(blocks, types.ContentBlock{
				Type: types.ContentTypeText,
				Text: part.Text,
			})
		}

		if part.FunctionCall != nil {
			blocks = append(blocks, types.ContentBlock{
				Type:      types.ContentTypeToolUse,
				ToolName:  part.FunctionCall.Name,
				ToolInput: part.FunctionCall.Args,
			})
		}
	}

	return blocks
}

// extractToolCalls extracts tool calls from Google content.
func (t *Transformer) extractToolCalls(content *Content) []types.ToolCall {
	if content == nil {
		return nil
	}

	var calls []types.ToolCall

	for _, part := range content.Parts {
		if part.FunctionCall != nil {
			calls = append(calls, types.ToolCall{
				Name:  part.FunctionCall.Name,
				Input: part.FunctionCall.Args,
			})
		}
	}

	return calls
}

// transformStopReason converts Google finish reason to unified format.
func (t *Transformer) transformStopReason(reason string) types.StopReason {
	switch reason {
	case "STOP":
		return types.StopReasonEnd
	case "MAX_TOKENS":
		return types.StopReasonMaxTokens
	case "SAFETY":
		return types.StopReasonContentFilter
	case "RECITATION":
		return types.StopReasonContentFilter
	case "OTHER":
		return types.StopReasonEnd
	default:
		return types.StopReasonEnd
	}
}

package openai

import (
	"encoding/json"
	"time"

	"github.com/Chloe199719/agent-router/pkg/schema"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// Transformer handles conversion between unified and OpenAI formats.
type Transformer struct {
	schemaTranslator *schema.Translator
}

// NewTransformer creates a new transformer.
func NewTransformer() *Transformer {
	return &Transformer{
		schemaTranslator: schema.NewTranslator(),
	}
}

// TransformRequest converts a unified request to OpenAI format.
func (t *Transformer) TransformRequest(req *types.CompletionRequest) *ChatCompletionRequest {
	oaiReq := &ChatCompletionRequest{
		Model:       req.Model,
		Messages:    t.transformMessages(req.Messages),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.StopSequences,
		Stream:      req.Stream,
	}

	if req.Stream {
		oaiReq.StreamOptions = &StreamOptions{IncludeUsage: true}
	}

	// Transform response format
	if req.ResponseFormat != nil {
		oaiReq.ResponseFormat = t.transformResponseFormat(req.ResponseFormat)
	}

	// Transform tools
	if len(req.Tools) > 0 {
		oaiReq.Tools = t.transformTools(req.Tools)
	}

	// Transform tool choice
	if req.ToolChoice != nil {
		oaiReq.ToolChoice = t.transformToolChoice(req.ToolChoice)
	}

	return oaiReq
}

// transformMessages converts unified messages to OpenAI format.
func (t *Transformer) transformMessages(messages []types.Message) []ChatMessage {
	result := make([]ChatMessage, 0, len(messages))

	for _, msg := range messages {
		oaiMsg := ChatMessage{
			Role: string(msg.Role),
		}

		// Check if this is a tool result message
		if msg.Role == types.RoleTool {
			for _, block := range msg.Content {
				if block.Type == types.ContentTypeToolResult {
					oaiMsg.ToolCallID = block.ToolResultID
					oaiMsg.Content = block.Text
					result = append(result, oaiMsg)
				}
			}
			continue
		}

		// Check if we need multipart content
		hasMultipleParts := len(msg.Content) > 1
		hasImages := false
		hasToolCalls := false

		for _, block := range msg.Content {
			if block.Type == types.ContentTypeImage {
				hasImages = true
			}
			if block.Type == types.ContentTypeToolUse {
				hasToolCalls = true
			}
		}

		if hasToolCalls && msg.Role == types.RoleAssistant {
			// Assistant message with tool calls
			var textContent string
			var toolCalls []ToolCall

			for _, block := range msg.Content {
				switch block.Type {
				case types.ContentTypeText:
					textContent += block.Text
				case types.ContentTypeToolUse:
					args, _ := json.Marshal(block.ToolInput)
					toolCalls = append(toolCalls, ToolCall{
						ID:   block.ToolUseID,
						Type: "function",
						Function: FunctionCall{
							Name:      block.ToolName,
							Arguments: string(args),
						},
					})
				}
			}

			if textContent != "" {
				oaiMsg.Content = textContent
			}
			oaiMsg.ToolCalls = toolCalls
		} else if hasImages || hasMultipleParts {
			// Multipart content
			var parts []ContentPart
			for _, block := range msg.Content {
				switch block.Type {
				case types.ContentTypeText:
					parts = append(parts, ContentPart{
						Type: "text",
						Text: block.Text,
					})
				case types.ContentTypeImage:
					url := block.ImageURL
					if url == "" && block.ImageBase64 != "" {
						url = "data:" + block.MediaType + ";base64," + block.ImageBase64
					}
					parts = append(parts, ContentPart{
						Type: "image_url",
						ImageURL: &ImageURL{
							URL: url,
						},
					})
				}
			}
			oaiMsg.Content = parts
		} else {
			// Simple text content
			var text string
			for _, block := range msg.Content {
				if block.Type == types.ContentTypeText {
					text += block.Text
				}
			}
			oaiMsg.Content = text
		}

		result = append(result, oaiMsg)
	}

	return result
}

// transformResponseFormat converts unified response format to OpenAI format.
func (t *Transformer) transformResponseFormat(rf *types.ResponseFormat) *ResponseFormat {
	oaiRF := t.schemaTranslator.ToOpenAI(rf)
	if oaiRF == nil {
		return nil
	}

	result := &ResponseFormat{
		Type: oaiRF.Type,
	}

	if oaiRF.JSONSchema != nil {
		result.JSONSchema = &JSONSchema{
			Name:        oaiRF.JSONSchema.Name,
			Description: oaiRF.JSONSchema.Description,
			Schema:      oaiRF.JSONSchema.Schema,
			Strict:      oaiRF.JSONSchema.Strict,
		}
	}

	return result
}

// transformTools converts unified tools to OpenAI format.
func (t *Transformer) transformTools(tools []types.Tool) []Tool {
	oaiTools := t.schemaTranslator.ToolsToOpenAI(tools)
	result := make([]Tool, len(oaiTools))
	for i, tool := range oaiTools {
		result[i] = Tool{
			Type: tool.Type,
			Function: Function{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
				Strict:      tool.Function.Strict,
			},
		}
	}
	return result
}

// transformToolChoice converts unified tool choice to OpenAI format.
func (t *Transformer) transformToolChoice(tc *types.ToolChoice) any {
	switch tc.Type {
	case types.ToolChoiceAuto:
		return "auto"
	case types.ToolChoiceRequired:
		return "required"
	case types.ToolChoiceNone:
		return "none"
	case types.ToolChoiceTool:
		return ToolChoiceObject{
			Type: "function",
			Function: &ToolChoiceFunction{
				Name: tc.Name,
			},
		}
	default:
		return "auto"
	}
}

// TransformResponse converts OpenAI response to unified format.
func (t *Transformer) TransformResponse(resp *ChatCompletionResponse) *types.CompletionResponse {
	if resp == nil || len(resp.Choices) == 0 {
		return nil
	}

	choice := resp.Choices[0]
	result := &types.CompletionResponse{
		ID:         resp.ID,
		Provider:   types.ProviderOpenAI,
		Model:      resp.Model,
		Content:    t.transformContent(choice.Message),
		StopReason: t.transformStopReason(choice.FinishReason),
		ToolCalls:  t.extractToolCalls(choice.Message),
		CreatedAt:  time.Unix(resp.Created, 0),
	}

	if resp.Usage != nil {
		result.Usage = types.Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		}
		if resp.Usage.PromptTokensDetails != nil {
			result.Usage.CachedTokens = resp.Usage.PromptTokensDetails.CachedTokens
		}
		if resp.Usage.CompletionTokensDetails != nil {
			result.Usage.ReasoningTokens = resp.Usage.CompletionTokensDetails.ReasoningTokens
		}
	}

	return result
}

// transformContent extracts content blocks from OpenAI message.
func (t *Transformer) transformContent(msg ChatMessage) []types.ContentBlock {
	var blocks []types.ContentBlock

	// Handle text content
	switch content := msg.Content.(type) {
	case string:
		if content != "" {
			blocks = append(blocks, types.ContentBlock{
				Type: types.ContentTypeText,
				Text: content,
			})
		}
	case []any:
		for _, part := range content {
			if partMap, ok := part.(map[string]any); ok {
				if partMap["type"] == "text" {
					blocks = append(blocks, types.ContentBlock{
						Type: types.ContentTypeText,
						Text: partMap["text"].(string),
					})
				}
			}
		}
	}

	// Handle tool calls
	for _, tc := range msg.ToolCalls {
		var input any
		json.Unmarshal([]byte(tc.Function.Arguments), &input)

		blocks = append(blocks, types.ContentBlock{
			Type:      types.ContentTypeToolUse,
			ToolUseID: tc.ID,
			ToolName:  tc.Function.Name,
			ToolInput: input,
		})
	}

	return blocks
}

// extractToolCalls extracts tool calls from OpenAI message.
func (t *Transformer) extractToolCalls(msg ChatMessage) []types.ToolCall {
	if len(msg.ToolCalls) == 0 {
		return nil
	}

	calls := make([]types.ToolCall, len(msg.ToolCalls))
	for i, tc := range msg.ToolCalls {
		var input any
		json.Unmarshal([]byte(tc.Function.Arguments), &input)

		calls[i] = types.ToolCall{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		}
	}

	return calls
}

// transformStopReason converts OpenAI finish reason to unified format.
func (t *Transformer) transformStopReason(reason string) types.StopReason {
	switch reason {
	case "stop":
		return types.StopReasonEnd
	case "length":
		return types.StopReasonMaxTokens
	case "tool_calls":
		return types.StopReasonToolUse
	case "content_filter":
		return types.StopReasonContentFilter
	default:
		return types.StopReasonEnd
	}
}

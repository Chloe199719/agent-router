package anthropic

import (
	"time"

	"github.com/Chloe199719/agent-router/pkg/schema"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// Transformer handles conversion between unified and Anthropic formats.
type Transformer struct {
	schemaTranslator *schema.Translator
}

// NewTransformer creates a new transformer.
func NewTransformer() *Transformer {
	return &Transformer{
		schemaTranslator: schema.NewTranslator(),
	}
}

// TransformRequest converts a unified request to Anthropic format.
func (t *Transformer) TransformRequest(req *types.CompletionRequest) *MessagesRequest {
	anthReq := &MessagesRequest{
		Model:         req.Model,
		MaxTokens:     8192, // Default
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		TopK:          req.TopK,
		StopSequences: req.StopSequences,
		Stream:        req.Stream,
	}

	if req.MaxTokens != nil {
		anthReq.MaxTokens = *req.MaxTokens
	}

	// Extract system message and transform other messages
	messages, system := t.transformMessages(req.Messages)
	anthReq.Messages = messages
	if system != "" {
		anthReq.System = system
	}

	// Transform response format
	if req.ResponseFormat != nil {
		anthReq.OutputConfig = t.transformResponseFormat(req.ResponseFormat)
	}

	// Transform tools
	if len(req.Tools) > 0 {
		anthReq.Tools = t.transformTools(req.Tools)
	}

	// Transform tool choice
	if req.ToolChoice != nil {
		anthReq.ToolChoice = t.transformToolChoice(req.ToolChoice)
	}

	return anthReq
}

// transformMessages converts unified messages to Anthropic format.
func (t *Transformer) transformMessages(messages []types.Message) ([]Message, string) {
	var result []Message
	var system string

	for _, msg := range messages {
		// Handle system messages
		if msg.Role == types.RoleSystem {
			for _, block := range msg.Content {
				if block.Type == types.ContentTypeText {
					if system != "" {
						system += "\n"
					}
					system += block.Text
				}
			}
			continue
		}

		anthMsg := Message{
			Role: t.mapRole(msg.Role),
		}

		// Check if we can use simple string content
		if len(msg.Content) == 1 && msg.Content[0].Type == types.ContentTypeText {
			anthMsg.Content = msg.Content[0].Text
		} else {
			// Use content blocks
			anthMsg.Content = t.transformContentBlocks(msg.Content)
		}

		result = append(result, anthMsg)
	}

	return result, system
}

// mapRole maps unified role to Anthropic role.
func (t *Transformer) mapRole(role types.Role) string {
	switch role {
	case types.RoleUser, types.RoleTool:
		return "user"
	case types.RoleAssistant:
		return "assistant"
	default:
		return "user"
	}
}

// transformContentBlocks converts unified content blocks to Anthropic format.
func (t *Transformer) transformContentBlocks(blocks []types.ContentBlock) []ContentBlock {
	var result []ContentBlock

	for _, block := range blocks {
		switch block.Type {
		case types.ContentTypeText:
			result = append(result, ContentBlock{
				Type: "text",
				Text: block.Text,
			})

		case types.ContentTypeImage:
			cb := ContentBlock{Type: "image"}
			if block.ImageBase64 != "" {
				cb.Source = &ImageSource{
					Type:      "base64",
					MediaType: block.MediaType,
					Data:      block.ImageBase64,
				}
			} else if block.ImageURL != "" {
				cb.Source = &ImageSource{
					Type: "url",
					URL:  block.ImageURL,
				}
			}
			result = append(result, cb)

		case types.ContentTypeToolUse:
			result = append(result, ContentBlock{
				Type:  "tool_use",
				ID:    block.ToolUseID,
				Name:  block.ToolName,
				Input: block.ToolInput,
			})

		case types.ContentTypeToolResult:
			result = append(result, ContentBlock{
				Type:      "tool_result",
				ToolUseID: block.ToolResultID,
				Content:   block.Text,
				IsError:   block.IsError,
			})
		}
	}

	return result
}

// transformResponseFormat converts unified response format to Anthropic format.
func (t *Transformer) transformResponseFormat(rf *types.ResponseFormat) *OutputConfig {
	anthConfig := t.schemaTranslator.ToAnthropic(rf)
	if anthConfig == nil {
		return nil
	}

	result := &OutputConfig{}
	if anthConfig.Format != nil {
		result.Format = &OutputFormat{
			Type:   anthConfig.Format.Type,
			Schema: anthConfig.Format.Schema,
		}
	}

	return result
}

// transformTools converts unified tools to Anthropic format.
func (t *Transformer) transformTools(tools []types.Tool) []Tool {
	anthTools := t.schemaTranslator.ToolsToAnthropic(tools)
	result := make([]Tool, len(anthTools))
	for i, tool := range anthTools {
		result[i] = Tool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		}
	}
	return result
}

// transformToolChoice converts unified tool choice to Anthropic format.
func (t *Transformer) transformToolChoice(tc *types.ToolChoice) *ToolChoice {
	result := &ToolChoice{
		DisableParallelToolUse: tc.DisableParallelToolUse,
	}

	switch tc.Type {
	case types.ToolChoiceAuto:
		result.Type = "auto"
	case types.ToolChoiceRequired:
		result.Type = "any"
	case types.ToolChoiceNone:
		result.Type = "none"
	case types.ToolChoiceTool:
		result.Type = "tool"
		result.Name = tc.Name
	default:
		result.Type = "auto"
	}

	return result
}

// TransformResponse converts Anthropic response to unified format.
func (t *Transformer) TransformResponse(resp *MessagesResponse) *types.CompletionResponse {
	if resp == nil {
		return nil
	}

	result := &types.CompletionResponse{
		ID:         resp.ID,
		Provider:   types.ProviderAnthropic,
		Model:      resp.Model,
		Content:    t.transformResponseContent(resp.Content),
		StopReason: t.transformStopReason(resp.StopReason),
		ToolCalls:  t.extractToolCalls(resp.Content),
		Usage: types.Usage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
			TotalTokens:  resp.Usage.InputTokens + resp.Usage.OutputTokens,
			CachedTokens: resp.Usage.CacheReadInputTokens,
		},
		CreatedAt: time.Now(),
	}

	return result
}

// transformResponseContent converts Anthropic content blocks to unified format.
func (t *Transformer) transformResponseContent(blocks []ContentBlock) []types.ContentBlock {
	var result []types.ContentBlock

	for _, block := range blocks {
		switch block.Type {
		case "text":
			result = append(result, types.ContentBlock{
				Type: types.ContentTypeText,
				Text: block.Text,
			})
		case "tool_use":
			result = append(result, types.ContentBlock{
				Type:      types.ContentTypeToolUse,
				ToolUseID: block.ID,
				ToolName:  block.Name,
				ToolInput: block.Input,
			})
		}
	}

	return result
}

// extractToolCalls extracts tool calls from Anthropic content blocks.
func (t *Transformer) extractToolCalls(blocks []ContentBlock) []types.ToolCall {
	var calls []types.ToolCall

	for _, block := range blocks {
		if block.Type == "tool_use" {
			calls = append(calls, types.ToolCall{
				ID:    block.ID,
				Name:  block.Name,
				Input: block.Input,
			})
		}
	}

	return calls
}

// transformStopReason converts Anthropic stop reason to unified format.
func (t *Transformer) transformStopReason(reason string) types.StopReason {
	switch reason {
	case "end_turn":
		return types.StopReasonEnd
	case "max_tokens":
		return types.StopReasonMaxTokens
	case "tool_use":
		return types.StopReasonToolUse
	case "stop_sequence":
		return types.StopReasonStopSequence
	default:
		return types.StopReasonEnd
	}
}

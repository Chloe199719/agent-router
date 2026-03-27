// Package thinking validates unified thinking / reasoning requests per provider.
//
// Model allowlists are substring/prefix heuristics aligned with provider docs (IDs change often;
// prefer matching families over hard-coding every snapshot). Authoritative lists:
//   - Anthropic: https://docs.anthropic.com/en/docs/about-claude/models/extended-thinking-models
//   - Google:    https://ai.google.dev/gemini-api/docs/thinking ("Supported models, tools, and capabilities")
//   - OpenAI:    https://platform.openai.com/docs/guides/reasoning
package thinking

import (
	"fmt"
	"strings"

	"github.com/Chloe199719/agent-router/pkg/errors"
	"github.com/Chloe199719/agent-router/pkg/types"
)

const minAnthropicThinkingBudget = 1024

// ModelSupportsThinking reports whether the model name is treated as thinking-capable
// for the given provider (heuristic; APIs evolve).
func ModelSupportsThinking(provider types.Provider, model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	if m == "" {
		return false
	}
	switch provider {
	case types.ProviderOpenAI:
		return openAIModelSupportsThinking(m)
	case types.ProviderAnthropic:
		return anthropicModelSupportsThinking(m)
	case types.ProviderGoogle, types.ProviderVertex:
		return googleModelSupportsThinking(m)
	default:
		return false
	}
}

// OpenAI: reasoning / reasoning_effort applies to GPT-5 family, o1/o3/o4 reasoning SKUs, and
// computer-use models per the reasoning guide and model reference.
var openAIThinkingSubstrings = []string{
	"gpt-5",        // gpt-5, gpt-5.4, gpt-5-mini, gpt-5-nano, gpt-5.4-pro, dated snapshots, etc.
	"computer-use", // e.g. computer-use-preview
}

// Prefixes for o-series reasoning models (o1-preview, o3-mini, o4-mini-deep-research, …).
var openAIThinkingOPrefixes = []string{"o1", "o3", "o4"}

func openAIModelSupportsThinking(m string) bool {
	for _, s := range openAIThinkingSubstrings {
		if strings.Contains(m, s) {
			return true
		}
	}
	for _, p := range openAIThinkingOPrefixes {
		if strings.HasPrefix(m, p) {
			return true
		}
	}
	return false
}

// Anthropic extended thinking: Claude 4.x Opus/Sonnet/Haiku, Haiku 4.5, Sonnet 3.7, 4.6 SKUs, etc.
// See extended-thinking-models doc; IDs include dated snapshots and -latest aliases.
var anthropicThinkingSubstrings = []string{
	"claude-opus-4",
	"claude-sonnet-4",
	"claude-haiku-4",
	"claude-3-7",
	"sonnet-3-7",
}

func anthropicModelSupportsThinking(m string) bool {
	for _, s := range anthropicThinkingSubstrings {
		if strings.Contains(m, s) {
			return true
		}
	}
	return false
}

// Google Gemini: thinking is supported on all 2.5 and 3.x series models (including 3.1, Flash-Lite, previews).
// https://ai.google.dev/gemini-api/docs/thinking
var googleThinkingSubstrings = []string{
	"gemini-2.5",
	"gemini-3",
}

func googleModelSupportsThinking(m string) bool {
	for _, s := range googleThinkingSubstrings {
		if strings.Contains(m, s) {
			return true
		}
	}
	return false
}

func isGemini3Model(m string) bool {
	return strings.Contains(strings.ToLower(m), "gemini-3")
}

// ValidateThinking returns nil if thinking is nil or the config is valid for provider + model.
func ValidateThinking(provider types.Provider, model string, thinking *types.ThinkingConfig, maxTokens *int) error {
	if thinking == nil {
		return nil
	}
	if !ModelSupportsThinking(provider, model) {
		return errors.ErrInvalidRequest(
			fmt.Sprintf("model %q does not support thinking for provider %s", model, provider),
		).WithProvider(provider)
	}

	switch provider {
	case types.ProviderOpenAI:
		return validateOpenAI(thinking)
	case types.ProviderAnthropic:
		return validateAnthropic(thinking, maxTokens)
	case types.ProviderGoogle, types.ProviderVertex:
		return validateGoogle(provider, model, thinking)
	default:
		return errors.ErrInvalidRequest(fmt.Sprintf("thinking is not supported for provider %s", provider)).WithProvider(provider)
	}
}

func validateOpenAI(thinking *types.ThinkingConfig) error {
	// Effort is optional; empty omits reasoning_effort (API default).
	return nil
}

func validateAnthropic(thinking *types.ThinkingConfig, maxTokens *int) error {
	adaptive := strings.EqualFold(thinking.Type, "adaptive")
	enabledExplicit := strings.EqualFold(thinking.Type, "enabled")
	enabledByBudget := thinking.Budget != nil

	if adaptive {
		if strings.TrimSpace(thinking.Effort) == "" {
			return errors.ErrInvalidRequest(`thinking: Anthropic adaptive mode requires non-empty "effort"`).WithProvider(types.ProviderAnthropic)
		}
		return nil
	}

	if enabledExplicit || enabledByBudget {
		if thinking.Budget == nil {
			return errors.ErrInvalidRequest(`thinking: Anthropic enabled mode requires "budget" (maps to budget_tokens)`).WithProvider(types.ProviderAnthropic)
		}
		if *thinking.Budget < minAnthropicThinkingBudget {
			return errors.ErrInvalidRequest(
				fmt.Sprintf("thinking: Anthropic budget_tokens must be >= %d", minAnthropicThinkingBudget),
			).WithProvider(types.ProviderAnthropic)
		}
		if maxTokens != nil && *thinking.Budget >= *maxTokens {
			return errors.ErrInvalidRequest("thinking: Anthropic budget must be less than max_tokens").WithProvider(types.ProviderAnthropic)
		}
		return nil
	}

	// Non-adaptive, no enabled signal: reject empty / ambiguous config for Anthropic.
	if thinking.Budget == nil && strings.TrimSpace(thinking.Effort) == "" &&
		strings.TrimSpace(thinking.Level) == "" && thinking.IncludeThoughts == nil {
		return errors.ErrInvalidRequest(
			`thinking: for Anthropic set "type":"adaptive" with "effort", or provide "budget" for enabled mode`,
		).WithProvider(types.ProviderAnthropic)
	}

	if strings.TrimSpace(thinking.Effort) != "" {
		return errors.ErrInvalidRequest(
			`thinking: for Anthropic "effort" requires "type":"adaptive"`,
		).WithProvider(types.ProviderAnthropic)
	}

	return errors.ErrInvalidRequest(
		`thinking: for Anthropic set "type":"adaptive" with "effort", or provide "budget" for enabled mode`,
	).WithProvider(types.ProviderAnthropic)
}

func validateGoogle(provider types.Provider, model string, thinking *types.ThinkingConfig) error {
	if isGemini3Model(model) && thinking.Budget != nil && strings.TrimSpace(thinking.Level) == "" {
		return errors.ErrInvalidRequest(
			`thinking: for Gemini 3 models prefer "level" (thinkingLevel); avoid relying on "budget" alone`,
		).WithProvider(provider)
	}
	return nil
}

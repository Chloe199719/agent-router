package thinking

import (
	"errors"
	"testing"

	routererrors "github.com/Chloe199719/agent-router/pkg/errors"
	"github.com/Chloe199719/agent-router/pkg/types"
)

func TestModelSupportsThinking(t *testing.T) {
	tests := []struct {
		provider types.Provider
		model    string
		want     bool
	}{
		// OpenAI — GPT-5 family, o-series, computer-use (reasoning guide + model reference)
		{types.ProviderOpenAI, "gpt-5", true},
		{types.ProviderOpenAI, "gpt-5.4", true},
		{types.ProviderOpenAI, "gpt-5-mini", true},
		{types.ProviderOpenAI, "gpt-5-nano", true},
		{types.ProviderOpenAI, "o1-preview", true},
		{types.ProviderOpenAI, "o1-mini", true},
		{types.ProviderOpenAI, "o3-mini", true},
		{types.ProviderOpenAI, "o3-pro", true},
		{types.ProviderOpenAI, "o4-mini", true},
		{types.ProviderOpenAI, "o4-mini-deep-research", true},
		{types.ProviderOpenAI, "computer-use-preview", true},
		{types.ProviderOpenAI, "gpt-4o", false},
		{types.ProviderOpenAI, "gpt-4.1", false},
		// Anthropic — extended thinking models doc
		{types.ProviderAnthropic, "claude-sonnet-4-20250514", true},
		{types.ProviderAnthropic, "claude-sonnet-4-5-20250929", true},
		{types.ProviderAnthropic, "claude-opus-4-20250514", true},
		{types.ProviderAnthropic, "claude-opus-4-6", true},
		{types.ProviderAnthropic, "claude-haiku-4-5-20251001", true},
		{types.ProviderAnthropic, "claude-3-7-sonnet-20250219", true},
		{types.ProviderAnthropic, "claude-3-5-sonnet-20241022", false},
		{types.ProviderAnthropic, "claude-3-5-haiku-20241022", false},
		// Google / Vertex — thinking doc: all Gemini 2.5 and 3.x series
		{types.ProviderGoogle, "gemini-2.5-flash", true},
		{types.ProviderGoogle, "gemini-2.5-pro", true},
		{types.ProviderGoogle, "gemini-2.5-flash-lite-preview-06-17", true},
		{types.ProviderGoogle, "gemini-3-flash-preview", true},
		{types.ProviderGoogle, "gemini-3.1-pro-preview", true},
		{types.ProviderGoogle, "gemini-2.0-flash", false},
		{types.ProviderVertex, "gemini-2.5-pro", true},
		{types.ProviderVertex, "gemini-3-pro-preview", true},
	}
	for _, tt := range tests {
		if got := ModelSupportsThinking(tt.provider, tt.model); got != tt.want {
			t.Errorf("ModelSupportsThinking(%q, %q) = %v, want %v", tt.provider, tt.model, got, tt.want)
		}
	}
}

func TestValidateThinking_Nil(t *testing.T) {
	if err := ValidateThinking(types.ProviderOpenAI, "gpt-5", nil, nil); err != nil {
		t.Fatalf("nil thinking: %v", err)
	}
}

func TestValidateThinking_OpenAI_EmptyOK(t *testing.T) {
	th := &types.ThinkingConfig{}
	if err := ValidateThinking(types.ProviderOpenAI, "gpt-5", th, nil); err != nil {
		t.Fatalf("openai empty thinking: %v", err)
	}
}

func TestValidateThinking_OpenAI_ModelUnsupported(t *testing.T) {
	th := &types.ThinkingConfig{Effort: "low"}
	err := ValidateThinking(types.ProviderOpenAI, "gpt-4o", th, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	var re *routererrors.RouterError
	if !errors.As(err, &re) || re.Code != routererrors.ErrCodeInvalidRequest {
		t.Fatalf("want invalid_request, got %v", err)
	}
}

func TestValidateThinking_Anthropic_Enabled(t *testing.T) {
	budget := 2048
	maxTok := 4096
	th := &types.ThinkingConfig{Budget: &budget}
	if err := ValidateThinking(types.ProviderAnthropic, "claude-sonnet-4-20250514", th, &maxTok); err != nil {
		t.Fatal(err)
	}
}

func TestValidateThinking_Anthropic_BudgetTooSmall(t *testing.T) {
	budget := 512
	th := &types.ThinkingConfig{Budget: &budget}
	err := ValidateThinking(types.ProviderAnthropic, "claude-sonnet-4-20250514", th, nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestValidateThinking_Anthropic_BudgetVsMaxTokens(t *testing.T) {
	budget := 8000
	maxTok := 4000
	th := &types.ThinkingConfig{Budget: &budget}
	err := ValidateThinking(types.ProviderAnthropic, "claude-sonnet-4-20250514", th, &maxTok)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestValidateThinking_Anthropic_Adaptive(t *testing.T) {
	th := &types.ThinkingConfig{Type: "adaptive", Effort: "high"}
	if err := ValidateThinking(types.ProviderAnthropic, "claude-sonnet-4-20250514", th, nil); err != nil {
		t.Fatal(err)
	}
}

func TestValidateThinking_Anthropic_AdaptiveMissingEffort(t *testing.T) {
	th := &types.ThinkingConfig{Type: "adaptive"}
	err := ValidateThinking(types.ProviderAnthropic, "claude-sonnet-4-20250514", th, nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestValidateThinking_Gemini3_BudgetWithoutLevel(t *testing.T) {
	budget := 1024
	th := &types.ThinkingConfig{Budget: &budget}
	err := ValidateThinking(types.ProviderGoogle, "gemini-3-flash-preview", th, nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestValidateThinking_Gemini3_WithLevel(t *testing.T) {
	th := &types.ThinkingConfig{Level: "low"}
	if err := ValidateThinking(types.ProviderGoogle, "gemini-3-flash-preview", th, nil); err != nil {
		t.Fatal(err)
	}
}

func TestValidateThinking_Gemini25_Budget(t *testing.T) {
	budget := 1024
	th := &types.ThinkingConfig{Budget: &budget}
	if err := ValidateThinking(types.ProviderGoogle, "gemini-2.5-flash", th, nil); err != nil {
		t.Fatal(err)
	}
}

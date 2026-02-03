package errors

import (
	"errors"
	"testing"

	"github.com/Chloe199719/agent-router/pkg/types"
)

func TestRouterError_Error(t *testing.T) {
	err := NewError(ErrCodeInvalidRequest, "missing required field")

	expected := "invalid_request: missing required field"
	if err.Error() != expected {
		t.Errorf("expected %q, got %q", expected, err.Error())
	}
}

func TestRouterError_ErrorWithProvider(t *testing.T) {
	err := NewError(ErrCodeInvalidRequest, "missing required field").
		WithProvider(types.ProviderOpenAI)

	expected := "[openai] invalid_request: missing required field"
	if err.Error() != expected {
		t.Errorf("expected %q, got %q", expected, err.Error())
	}
}

func TestRouterError_Unwrap(t *testing.T) {
	cause := errors.New("original error")
	err := NewError(ErrCodeServerError, "server error").WithCause(cause)

	if err.Unwrap() != cause {
		t.Error("expected Unwrap to return cause")
	}
}

func TestRouterError_Is(t *testing.T) {
	err1 := NewError(ErrCodeRateLimit, "too many requests")
	err2 := NewError(ErrCodeRateLimit, "different message")
	err3 := NewError(ErrCodeServerError, "server error")

	if !errors.Is(err1, err2) {
		t.Error("expected errors with same code to match")
	}

	if errors.Is(err1, err3) {
		t.Error("expected errors with different codes to not match")
	}
}

func TestRouterError_Chaining(t *testing.T) {
	cause := errors.New("connection refused")
	details := map[string]any{"retry_after": 30}

	err := NewError(ErrCodeRateLimit, "rate limited").
		WithProvider(types.ProviderAnthropic).
		WithStatusCode(429).
		WithCause(cause).
		WithDetails(details)

	if err.Provider != types.ProviderAnthropic {
		t.Errorf("expected provider %q, got %q", types.ProviderAnthropic, err.Provider)
	}

	if err.StatusCode != 429 {
		t.Errorf("expected status code 429, got %d", err.StatusCode)
	}

	if err.Cause != cause {
		t.Error("expected cause to be set")
	}

	if err.Details["retry_after"] != 30 {
		t.Error("expected details to be set")
	}
}

func TestErrInvalidRequest(t *testing.T) {
	err := ErrInvalidRequest("bad input")

	if err.Code != ErrCodeInvalidRequest {
		t.Errorf("expected code %q, got %q", ErrCodeInvalidRequest, err.Code)
	}

	if err.Message != "bad input" {
		t.Errorf("expected message 'bad input', got %q", err.Message)
	}
}

func TestErrAuthentication(t *testing.T) {
	err := ErrAuthentication(types.ProviderOpenAI, "invalid credentials")

	if err.Code != ErrCodeAuthentication {
		t.Errorf("expected code %q, got %q", ErrCodeAuthentication, err.Code)
	}

	if err.Provider != types.ProviderOpenAI {
		t.Errorf("expected provider %q, got %q", types.ProviderOpenAI, err.Provider)
	}
}

func TestErrRateLimit(t *testing.T) {
	err := ErrRateLimit(types.ProviderAnthropic, "too many requests")

	if err.Code != ErrCodeRateLimit {
		t.Errorf("expected code %q, got %q", ErrCodeRateLimit, err.Code)
	}

	if err.StatusCode != 429 {
		t.Errorf("expected status code 429, got %d", err.StatusCode)
	}
}

func TestErrServerError(t *testing.T) {
	err := ErrServerError(types.ProviderGoogle, "internal error")

	if err.Code != ErrCodeServerError {
		t.Errorf("expected code %q, got %q", ErrCodeServerError, err.Code)
	}

	if err.StatusCode != 500 {
		t.Errorf("expected status code 500, got %d", err.StatusCode)
	}
}

func TestErrUnsupportedFeature(t *testing.T) {
	err := ErrUnsupportedFeature(types.ProviderGoogle, types.FeatureBatch)

	if err.Code != ErrCodeUnsupportedFeature {
		t.Errorf("expected code %q, got %q", ErrCodeUnsupportedFeature, err.Code)
	}

	expectedMsg := "provider google does not support feature: batch"
	if err.Message != expectedMsg {
		t.Errorf("expected message %q, got %q", expectedMsg, err.Message)
	}
}

func TestErrProviderUnavailable(t *testing.T) {
	err := ErrProviderUnavailable(types.ProviderOpenAI, "not configured")

	if err.Code != ErrCodeProviderUnavailable {
		t.Errorf("expected code %q, got %q", ErrCodeProviderUnavailable, err.Code)
	}
}

func TestErrTimeout(t *testing.T) {
	err := ErrTimeout(types.ProviderAnthropic)

	if err.Code != ErrCodeTimeout {
		t.Errorf("expected code %q, got %q", ErrCodeTimeout, err.Code)
	}

	if err.Message != "request timed out" {
		t.Errorf("expected message 'request timed out', got %q", err.Message)
	}
}

func TestErrInvalidAPIKey(t *testing.T) {
	err := ErrInvalidAPIKey(types.ProviderOpenAI)

	if err.Code != ErrCodeInvalidAPIKey {
		t.Errorf("expected code %q, got %q", ErrCodeInvalidAPIKey, err.Code)
	}

	if err.StatusCode != 401 {
		t.Errorf("expected status code 401, got %d", err.StatusCode)
	}
}

func TestErrModelNotFound(t *testing.T) {
	err := ErrModelNotFound(types.ProviderOpenAI, "gpt-5")

	if err.Code != ErrCodeModelNotFound {
		t.Errorf("expected code %q, got %q", ErrCodeModelNotFound, err.Code)
	}

	if err.StatusCode != 404 {
		t.Errorf("expected status code 404, got %d", err.StatusCode)
	}

	expectedMsg := "model not found: gpt-5"
	if err.Message != expectedMsg {
		t.Errorf("expected message %q, got %q", expectedMsg, err.Message)
	}
}

func TestErrContextLength(t *testing.T) {
	err := ErrContextLength(types.ProviderAnthropic, "context too long")

	if err.Code != ErrCodeContextLength {
		t.Errorf("expected code %q, got %q", ErrCodeContextLength, err.Code)
	}

	if err.StatusCode != 400 {
		t.Errorf("expected status code 400, got %d", err.StatusCode)
	}
}

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		err      error
		expected bool
	}{
		{ErrRateLimit(types.ProviderOpenAI, "rate limited"), true},
		{ErrServerError(types.ProviderOpenAI, "server error"), true},
		{ErrTimeout(types.ProviderOpenAI), true},
		{ErrInvalidRequest("bad input"), false},
		{ErrAuthentication(types.ProviderOpenAI, "bad auth"), false},
		{ErrInvalidAPIKey(types.ProviderOpenAI), false},
		{errors.New("regular error"), false},
	}

	for _, tt := range tests {
		result := IsRetryable(tt.err)
		if result != tt.expected {
			t.Errorf("IsRetryable(%v) = %v, expected %v", tt.err, result, tt.expected)
		}
	}
}

func TestIsAuthError(t *testing.T) {
	tests := []struct {
		err      error
		expected bool
	}{
		{ErrAuthentication(types.ProviderOpenAI, "bad auth"), true},
		{ErrInvalidAPIKey(types.ProviderOpenAI), true},
		{ErrRateLimit(types.ProviderOpenAI, "rate limited"), false},
		{ErrServerError(types.ProviderOpenAI, "server error"), false},
		{errors.New("regular error"), false},
	}

	for _, tt := range tests {
		result := IsAuthError(tt.err)
		if result != tt.expected {
			t.Errorf("IsAuthError(%v) = %v, expected %v", tt.err, result, tt.expected)
		}
	}
}

func TestErrorsAs(t *testing.T) {
	originalErr := ErrRateLimit(types.ProviderOpenAI, "rate limited")

	// Wrap the error
	wrappedErr := errors.New("wrapped: " + originalErr.Error())

	var routerErr *RouterError
	if errors.As(originalErr, &routerErr) {
		if routerErr.Code != ErrCodeRateLimit {
			t.Errorf("expected code %q, got %q", ErrCodeRateLimit, routerErr.Code)
		}
	} else {
		t.Error("expected errors.As to succeed for RouterError")
	}

	// Regular error should not match
	if errors.As(wrappedErr, &routerErr) {
		// This will not match because wrappedErr is a regular error
	}
}

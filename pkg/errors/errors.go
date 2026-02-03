// Package errors provides unified error types for the agent router.
package errors

import (
	"errors"
	"fmt"

	"github.com/Chloe199719/agent-router/pkg/types"
)

// Error codes
const (
	ErrCodeInvalidRequest      = "invalid_request"
	ErrCodeAuthentication      = "authentication_error"
	ErrCodeRateLimit           = "rate_limit"
	ErrCodeServerError         = "server_error"
	ErrCodeUnsupportedFeature  = "unsupported_feature"
	ErrCodeProviderUnavailable = "provider_unavailable"
	ErrCodeTimeout             = "timeout"
	ErrCodeContentFilter       = "content_filter"
	ErrCodeInvalidAPIKey       = "invalid_api_key"
	ErrCodeModelNotFound       = "model_not_found"
	ErrCodeContextLength       = "context_length_exceeded"
)

// RouterError is the base error type for all router errors.
type RouterError struct {
	// Error code for programmatic handling
	Code string `json:"code"`

	// Human-readable error message
	Message string `json:"message"`

	// Provider that generated the error (if applicable)
	Provider types.Provider `json:"provider,omitempty"`

	// HTTP status code from provider (if applicable)
	StatusCode int `json:"status_code,omitempty"`

	// Original error from provider
	Cause error `json:"-"`

	// Additional details
	Details map[string]any `json:"details,omitempty"`
}

func (e *RouterError) Error() string {
	if e.Provider != "" {
		return fmt.Sprintf("[%s] %s: %s", e.Provider, e.Code, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *RouterError) Unwrap() error {
	return e.Cause
}

// Is checks if the error matches a target error.
func (e *RouterError) Is(target error) bool {
	var t *RouterError
	if errors.As(target, &t) {
		return e.Code == t.Code
	}
	return false
}

// NewError creates a new RouterError.
func NewError(code, message string) *RouterError {
	return &RouterError{
		Code:    code,
		Message: message,
	}
}

// WithProvider adds provider information to the error.
func (e *RouterError) WithProvider(p types.Provider) *RouterError {
	e.Provider = p
	return e
}

// WithCause adds the underlying cause to the error.
func (e *RouterError) WithCause(err error) *RouterError {
	e.Cause = err
	return e
}

// WithStatusCode adds HTTP status code to the error.
func (e *RouterError) WithStatusCode(code int) *RouterError {
	e.StatusCode = code
	return e
}

// WithDetails adds additional details to the error.
func (e *RouterError) WithDetails(details map[string]any) *RouterError {
	e.Details = details
	return e
}

// Common error constructors

// ErrInvalidRequest creates an invalid request error.
func ErrInvalidRequest(message string) *RouterError {
	return NewError(ErrCodeInvalidRequest, message)
}

// ErrAuthentication creates an authentication error.
func ErrAuthentication(provider types.Provider, message string) *RouterError {
	return NewError(ErrCodeAuthentication, message).WithProvider(provider)
}

// ErrRateLimit creates a rate limit error.
func ErrRateLimit(provider types.Provider, message string) *RouterError {
	return NewError(ErrCodeRateLimit, message).WithProvider(provider).WithStatusCode(429)
}

// ErrServerError creates a server error.
func ErrServerError(provider types.Provider, message string) *RouterError {
	return NewError(ErrCodeServerError, message).WithProvider(provider).WithStatusCode(500)
}

// ErrUnsupportedFeature creates an unsupported feature error.
func ErrUnsupportedFeature(provider types.Provider, feature types.Feature) *RouterError {
	return NewError(
		ErrCodeUnsupportedFeature,
		fmt.Sprintf("provider %s does not support feature: %s", provider, feature),
	).WithProvider(provider)
}

// ErrProviderUnavailable creates a provider unavailable error.
func ErrProviderUnavailable(provider types.Provider, message string) *RouterError {
	return NewError(ErrCodeProviderUnavailable, message).WithProvider(provider)
}

// ErrTimeout creates a timeout error.
func ErrTimeout(provider types.Provider) *RouterError {
	return NewError(ErrCodeTimeout, "request timed out").WithProvider(provider)
}

// ErrInvalidAPIKey creates an invalid API key error.
func ErrInvalidAPIKey(provider types.Provider) *RouterError {
	return NewError(ErrCodeInvalidAPIKey, "invalid or missing API key").WithProvider(provider).WithStatusCode(401)
}

// ErrModelNotFound creates a model not found error.
func ErrModelNotFound(provider types.Provider, model string) *RouterError {
	return NewError(
		ErrCodeModelNotFound,
		fmt.Sprintf("model not found: %s", model),
	).WithProvider(provider).WithStatusCode(404)
}

// ErrContextLength creates a context length exceeded error.
func ErrContextLength(provider types.Provider, message string) *RouterError {
	return NewError(ErrCodeContextLength, message).WithProvider(provider).WithStatusCode(400)
}

// IsRetryable returns true if the error is potentially retryable.
func IsRetryable(err error) bool {
	var rerr *RouterError
	if errors.As(err, &rerr) {
		switch rerr.Code {
		case ErrCodeRateLimit, ErrCodeServerError, ErrCodeTimeout:
			return true
		}
	}
	return false
}

// IsAuthError returns true if the error is an authentication error.
func IsAuthError(err error) bool {
	var rerr *RouterError
	if errors.As(err, &rerr) {
		return rerr.Code == ErrCodeAuthentication || rerr.Code == ErrCodeInvalidAPIKey
	}
	return false
}

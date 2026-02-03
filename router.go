// Package router provides a unified interface for multiple LLM providers.
//
// Example usage:
//
//	r, err := router.New(
//		router.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
//		router.WithAnthropic(os.Getenv("ANTHROPIC_API_KEY")),
//		router.WithGoogle(os.Getenv("GOOGLE_API_KEY")),
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	resp, err := r.Complete(ctx, &types.CompletionRequest{
//		Provider: types.ProviderOpenAI,
//		Model:    "gpt-4o",
//		Messages: []types.Message{
//			types.NewTextMessage(types.RoleUser, "Hello!"),
//		},
//	})
package router

import (
	"context"
	"fmt"

	"github.com/Chloe199719/agent-router/pkg/batch"
	"github.com/Chloe199719/agent-router/pkg/errors"
	"github.com/Chloe199719/agent-router/pkg/provider"
	"github.com/Chloe199719/agent-router/pkg/provider/anthropic"
	"github.com/Chloe199719/agent-router/pkg/provider/google"
	"github.com/Chloe199719/agent-router/pkg/provider/openai"
	"github.com/Chloe199719/agent-router/pkg/types"
)

// Router provides a unified interface for multiple LLM providers.
type Router struct {
	providers map[types.Provider]provider.Provider
	batch     *batch.Manager
	config    *Config
}

// Config configures the router.
type Config struct {
	// OnUnsupportedFeature controls behavior when a provider doesn't support a feature.
	OnUnsupportedFeature UnsupportedFeaturePolicy

	// Debug enables debug logging.
	Debug bool
}

// UnsupportedFeaturePolicy controls how unsupported features are handled.
type UnsupportedFeaturePolicy string

const (
	// PolicyError throws an error when a feature is not supported.
	PolicyError UnsupportedFeaturePolicy = "error"

	// PolicyWarn logs a warning and continues without the feature.
	PolicyWarn UnsupportedFeaturePolicy = "warn"

	// PolicyIgnore silently ignores unsupported features.
	PolicyIgnore UnsupportedFeaturePolicy = "ignore"
)

// Option configures the router.
type Option func(*Router)

// New creates a new router with the given options.
func New(opts ...Option) (*Router, error) {
	r := &Router{
		providers: make(map[types.Provider]provider.Provider),
		batch:     batch.NewManager(),
		config: &Config{
			OnUnsupportedFeature: PolicyError,
		},
	}

	for _, opt := range opts {
		opt(r)
	}

	if len(r.providers) == 0 {
		return nil, fmt.Errorf("at least one provider must be configured")
	}

	return r, nil
}

// WithOpenAI adds OpenAI as a provider.
func WithOpenAI(apiKey string, opts ...provider.Option) Option {
	return func(r *Router) {
		allOpts := append([]provider.Option{provider.WithAPIKey(apiKey)}, opts...)
		client := openai.New(allOpts...)
		r.providers[types.ProviderOpenAI] = client
		r.batch.RegisterProvider(client)
	}
}

// WithAnthropic adds Anthropic as a provider.
func WithAnthropic(apiKey string, opts ...provider.Option) Option {
	return func(r *Router) {
		allOpts := append([]provider.Option{provider.WithAPIKey(apiKey)}, opts...)
		client := anthropic.New(allOpts...)
		r.providers[types.ProviderAnthropic] = client
		r.batch.RegisterProvider(client)
	}
}

// WithGoogle adds Google (Gemini) as a provider.
func WithGoogle(apiKey string, opts ...provider.Option) Option {
	return func(r *Router) {
		allOpts := append([]provider.Option{provider.WithAPIKey(apiKey)}, opts...)
		client := google.New(allOpts...)
		r.providers[types.ProviderGoogle] = client
		r.batch.RegisterProvider(client)
	}
}

// WithUnsupportedFeaturePolicy sets the policy for unsupported features.
func WithUnsupportedFeaturePolicy(policy UnsupportedFeaturePolicy) Option {
	return func(r *Router) {
		r.config.OnUnsupportedFeature = policy
	}
}

// WithDebug enables debug logging.
func WithDebug(debug bool) Option {
	return func(r *Router) {
		r.config.Debug = debug
	}
}

// Complete sends a completion request to the specified provider.
func (r *Router) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	p, err := r.getProvider(req.Provider)
	if err != nil {
		return nil, err
	}

	// Check feature support
	if err := r.checkFeatureSupport(p, req); err != nil {
		return nil, err
	}

	return p.Complete(ctx, req)
}

// Stream sends a streaming completion request to the specified provider.
func (r *Router) Stream(ctx context.Context, req *types.CompletionRequest) (types.StreamReader, error) {
	p, err := r.getProvider(req.Provider)
	if err != nil {
		return nil, err
	}

	// Check streaming support
	if !p.SupportsFeature(types.FeatureStreaming) {
		return nil, errors.ErrUnsupportedFeature(req.Provider, types.FeatureStreaming)
	}

	// Check other feature support
	if err := r.checkFeatureSupport(p, req); err != nil {
		return nil, err
	}

	return p.Stream(ctx, req)
}

// Batch returns the batch manager for batch processing operations.
func (r *Router) Batch() *batch.Manager {
	return r.batch
}

// Provider returns the provider implementation for direct access.
func (r *Router) Provider(name types.Provider) (provider.Provider, error) {
	return r.getProvider(name)
}

// Providers returns all configured providers.
func (r *Router) Providers() []types.Provider {
	providers := make([]types.Provider, 0, len(r.providers))
	for name := range r.providers {
		providers = append(providers, name)
	}
	return providers
}

// SupportsFeature checks if a provider supports a specific feature.
func (r *Router) SupportsFeature(providerName types.Provider, feature types.Feature) bool {
	p, err := r.getProvider(providerName)
	if err != nil {
		return false
	}
	return p.SupportsFeature(feature)
}

// Models returns the available models for a provider.
func (r *Router) Models(providerName types.Provider) ([]string, error) {
	p, err := r.getProvider(providerName)
	if err != nil {
		return nil, err
	}
	return p.Models(), nil
}

// getProvider returns the provider for the given name.
func (r *Router) getProvider(name types.Provider) (provider.Provider, error) {
	p, ok := r.providers[name]
	if !ok {
		return nil, errors.ErrProviderUnavailable(name, "provider not configured")
	}
	return p, nil
}

// checkFeatureSupport checks if the provider supports the features required by the request.
func (r *Router) checkFeatureSupport(p provider.Provider, req *types.CompletionRequest) error {
	// Check structured output support
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_schema" {
		if !p.SupportsFeature(types.FeatureStructuredOutput) {
			return r.handleUnsupportedFeature(p.Name(), types.FeatureStructuredOutput)
		}
	}

	// Check JSON mode support
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json" {
		if !p.SupportsFeature(types.FeatureJSON) {
			return r.handleUnsupportedFeature(p.Name(), types.FeatureJSON)
		}
	}

	// Check tools support
	if len(req.Tools) > 0 {
		if !p.SupportsFeature(types.FeatureTools) {
			return r.handleUnsupportedFeature(p.Name(), types.FeatureTools)
		}
	}

	// Check vision support (detect images in messages)
	for _, msg := range req.Messages {
		for _, block := range msg.Content {
			if block.Type == types.ContentTypeImage {
				if !p.SupportsFeature(types.FeatureVision) {
					return r.handleUnsupportedFeature(p.Name(), types.FeatureVision)
				}
				break
			}
		}
	}

	return nil
}

// handleUnsupportedFeature handles an unsupported feature based on policy.
func (r *Router) handleUnsupportedFeature(providerName types.Provider, feature types.Feature) error {
	switch r.config.OnUnsupportedFeature {
	case PolicyError:
		return errors.ErrUnsupportedFeature(providerName, feature)
	case PolicyWarn:
		// TODO: Add logging
		return nil
	case PolicyIgnore:
		return nil
	default:
		return errors.ErrUnsupportedFeature(providerName, feature)
	}
}

# Agent Router

A unified Go library for making LLM inference requests across multiple providers (OpenAI, Anthropic, Google/Gemini) with a single, consistent interface.

## Features

- **Unified Interface** - Same API for all providers
- **Streaming** - Real-time streaming responses with normalized events
- **Structured Output** - JSON Schema support with automatic translation between provider formats
- **Tool/Function Calling** - Define tools once, use with any provider
- **Batch Processing** - Unified batch API for all providers (50% cost reduction)
- **Vision/Multimodal** - Support for image inputs
- **Feature Detection** - Check provider capabilities at runtime

## Installation

```bash
go get github.com/Chloe199719/agent-router
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "os"

    router "github.com/Chloe199719/agent-router"
    "github.com/Chloe199719/agent-router/pkg/types"
)

func main() {
    // Create router with multiple providers
    r, err := router.New(
        router.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
        router.WithAnthropic(os.Getenv("ANTHROPIC_API_KEY")),
        router.WithGoogle(os.Getenv("GOOGLE_API_KEY")),
    )
    if err != nil {
        panic(err)
    }

    ctx := context.Background()

    // Make a completion request - same interface for all providers
    resp, err := r.Complete(ctx, &types.CompletionRequest{
        Provider: types.ProviderOpenAI,  // or ProviderAnthropic, ProviderGoogle
        Model:    "gpt-4o-mini",
        Messages: []types.Message{
            types.NewTextMessage(types.RoleUser, "Hello!"),
        },
    })
    if err != nil {
        panic(err)
    }

    fmt.Println(resp.Text())
    fmt.Printf("Tokens: %d input, %d output\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}
```

## Providers

| Provider | Completion | Streaming | Structured Output | Tools | Batch |
|----------|------------|-----------|-------------------|-------|-------|
| OpenAI   | Yes | Yes | Yes | Yes | Yes |
| Anthropic | Yes | Yes | Yes | Yes | Yes |
| Google/Gemini | Yes | Yes | Yes | Yes | Yes |

All providers support batch processing at 50% reduced cost with 24-hour turnaround.

### Provider-Specific Configuration

```go
import "github.com/Chloe199719/agent-router/pkg/provider"

// Custom base URL (e.g., for Azure OpenAI or proxies)
router.WithOpenAI(apiKey, provider.WithBaseURL("https://your-endpoint.com"))

// Custom HTTP client
router.WithAnthropic(apiKey, provider.WithHTTPClient(customClient))
```

## Streaming

```go
stream, err := r.Stream(ctx, &types.CompletionRequest{
    Provider: types.ProviderOpenAI,
    Model:    "gpt-4o-mini",
    Messages: []types.Message{
        types.NewTextMessage(types.RoleUser, "Tell me a story"),
    },
})
if err != nil {
    panic(err)
}
defer stream.Close()

for {
    event, err := stream.Next()
    if err != nil {
        panic(err)
    }
    if event == nil {
        break // Stream complete
    }

    switch event.Type {
    case types.StreamEventContentDelta:
        fmt.Print(event.Delta.Text)
    case types.StreamEventDone:
        fmt.Printf("\n[Done, tokens: %d]\n", event.Usage.OutputTokens)
    }
}
```

### Stream Event Types

| Event Type | Description |
|------------|-------------|
| `StreamEventStart` | Stream started |
| `StreamEventContentDelta` | Text content chunk |
| `StreamEventToolCallStart` | Tool call began |
| `StreamEventToolCallDelta` | Tool call input chunk |
| `StreamEventToolCallEnd` | Tool call finished |
| `StreamEventDone` | Stream completed |
| `StreamEventError` | Error occurred |

## Structured Output (JSON Schema)

All providers support structured output with automatic schema translation:

```go
schema := types.JSONSchema{
    Type: "object",
    Properties: map[string]types.JSONSchema{
        "name": {Type: "string", Description: "Person's name"},
        "age":  {Type: "integer", Description: "Person's age"},
    },
    Required: []string{"name", "age"},
}

resp, err := r.Complete(ctx, (&types.CompletionRequest{
    Provider: types.ProviderOpenAI,
    Model:    "gpt-4o-mini",
    Messages: []types.Message{
        types.NewTextMessage(types.RoleUser, "Extract: John Smith is 42 years old"),
    },
}).WithJSONSchema("person", schema))

// resp.Text() = `{"name":"John Smith","age":42}`
```

The library automatically handles provider-specific schema requirements:
- **OpenAI**: Adds `additionalProperties: false` and uses `response_format.json_schema`
- **Anthropic**: Wraps schema in `output_config.format` with proper structure
- **Google**: Converts types to uppercase (STRING, INTEGER, etc.) for Gemini API

## Tool Calling

Define tools once and use them with any provider:

```go
tools := []types.Tool{
    {
        Name:        "get_weather",
        Description: "Get the current weather for a location",
        Parameters: types.JSONSchema{
            Type: "object",
            Properties: map[string]types.JSONSchema{
                "location": {Type: "string", Description: "City name"},
                "unit":     {Type: "string", Enum: []any{"celsius", "fahrenheit"}},
            },
            Required: []string{"location"},
        },
    },
}

resp, err := r.Complete(ctx, (&types.CompletionRequest{
    Provider: types.ProviderOpenAI,
    Model:    "gpt-4o-mini",
    Messages: []types.Message{
        types.NewTextMessage(types.RoleUser, "What's the weather in Tokyo?"),
    },
}).WithTools(tools...))

if resp.HasToolCalls() {
    for _, tc := range resp.ToolCalls {
        fmt.Printf("Tool: %s, Input: %v\n", tc.Name, tc.Input)
    }
}
```

### Continuing After Tool Calls

```go
// Add assistant response with tool calls
messages := append(messages, types.Message{
    Role:    types.RoleAssistant,
    Content: resp.Content,
})

// Add tool result
messages = append(messages, types.NewToolResultMessage(
    toolCall.ID,
    `{"temperature": 22, "condition": "sunny"}`,
    false, // isError
))

// Continue conversation
resp, err = r.Complete(ctx, (&types.CompletionRequest{
    Provider: types.ProviderOpenAI,
    Model:    "gpt-4o-mini",
    Messages: messages,
}).WithTools(tools...))
```

## Batch Processing

Process many requests asynchronously at reduced cost (50% off for most providers):

```go
import "github.com/Chloe199719/agent-router/pkg/batch"

// Create batch requests
requests := []batch.Request{
    {CustomID: "req-1", Request: &types.CompletionRequest{...}},
    {CustomID: "req-2", Request: &types.CompletionRequest{...}},
}

// Submit batch
job, err := r.Batch().Create(ctx, types.ProviderOpenAI, requests)

// Wait for completion (or poll manually)
job, err = r.Batch().Wait(ctx, types.ProviderOpenAI, job.ID, 30*time.Second)

// Get results
results, err := r.Batch().GetResults(ctx, types.ProviderOpenAI, job.ID)
for _, result := range results {
    if result.Error != nil {
        fmt.Printf("%s failed: %v\n", result.CustomID, result.Error)
    } else {
        fmt.Printf("%s: %s\n", result.CustomID, result.Response.Text())
    }
}
```

### Batch Job States

| Status | Description |
|--------|-------------|
| `pending` | Job created, not yet processing |
| `validating` | Validating request format |
| `in_progress` | Processing requests |
| `finalizing` | Completing final requests |
| `completed` | All requests processed |
| `failed` | Job failed |
| `cancelled` | Job was cancelled |
| `expired` | Job expired before completion |

## Request Options

```go
req := &types.CompletionRequest{
    Provider:      types.ProviderOpenAI,
    Model:         "gpt-4o",
    Messages:      messages,
    MaxTokens:     types.Ptr(1000),     // Pointer helper
    Temperature:   types.Ptr(0.7),
    TopP:          types.Ptr(0.9),
    TopK:          types.Ptr(40),       // Anthropic/Google only
    StopSequences: []string{"END"},
}

// Or use builder methods
req = (&types.CompletionRequest{...}).
    WithMaxTokens(1000).
    WithTemperature(0.7).
    WithTools(tools...).
    WithJSONSchema("name", schema)
```

## Message Types

```go
// Simple text message
msg := types.NewTextMessage(types.RoleUser, "Hello")

// Tool result message
msg := types.NewToolResultMessage(toolID, resultJSON, false)

// Complex message with multiple content blocks
msg := types.Message{
    Role: types.RoleUser,
    Content: []types.ContentBlock{
        {Type: types.ContentTypeText, Text: "What's in this image?"},
        {Type: types.ContentTypeImage, ImageURL: "https://..."},
    },
}

// Image from base64
msg := types.Message{
    Role: types.RoleUser,
    Content: []types.ContentBlock{
        {Type: types.ContentTypeText, Text: "Describe this:"},
        {
            Type:        types.ContentTypeImage,
            ImageBase64: base64Data,
            MediaType:   "image/png",
        },
    },
}
```

## Feature Detection

Check provider capabilities at runtime:

```go
if r.SupportsFeature(types.ProviderGoogle, types.FeatureBatch) {
    // Use batch API
}

// Available features
types.FeatureStreaming        // Streaming responses
types.FeatureStructuredOutput // JSON Schema output
types.FeatureTools            // Function/tool calling
types.FeatureVision           // Image inputs
types.FeatureBatch            // Batch processing
types.FeatureJSON             // JSON mode (less strict than schema)
```

## Error Handling

```go
import "github.com/Chloe199719/agent-router/pkg/errors"

resp, err := r.Complete(ctx, req)
if err != nil {
    var routerErr *errors.RouterError
    if errors.As(err, &routerErr) {
        switch routerErr.Code {
        case errors.ErrCodeProviderUnavailable:
            // Provider not configured
        case errors.ErrCodeUnsupportedFeature:
            // Feature not supported by provider
        case errors.ErrCodeAPIError:
            // Provider API returned an error
            fmt.Println(routerErr.StatusCode) // HTTP status
        case errors.ErrCodeRateLimit:
            // Rate limited
        }
    }
}
```

## Configuration Options

```go
r, err := router.New(
    router.WithOpenAI(apiKey),
    router.WithAnthropic(apiKey),
    router.WithGoogle(apiKey),
    
    // How to handle unsupported features
    router.WithUnsupportedFeaturePolicy(router.PolicyWarn), // PolicyError (default), PolicyWarn, PolicyIgnore
    
    // Debug mode
    router.WithDebug(true),
)
```

## Models

Recommended models for each provider:

| Provider | Fast | Balanced | Powerful |
|----------|------|----------|----------|
| OpenAI | gpt-4o-mini | gpt-4o | gpt-4o |
| Anthropic | claude-3-5-haiku-20241022 | claude-sonnet-4-20250514 | claude-sonnet-4-20250514 |
| Google | gemini-2.0-flash | gemini-2.5-flash | gemini-2.5-pro |

```go
models, _ := r.Models(types.ProviderOpenAI)
// Returns list of available models
```

## Running Tests

```bash
# Unit tests
make test

# Integration tests (requires API keys)
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
make test-integration

# Provider-specific tests
make test-openai
make test-anthropic
make test-google

# Feature-specific tests
make test-streaming
make test-tools
make test-structured
```

## Examples

See the [examples](./examples) directory:

- [Basic usage](./examples/basic/main.go) - Simple completion requests
- [Streaming](./examples/streaming/main.go) - Real-time streaming
- [Tool calling](./examples/tools/main.go) - Function/tool usage

## License

MIT

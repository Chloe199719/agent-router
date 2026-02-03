// Example demonstrating basic usage of the agent router.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	router "github.com/Chloe199719/agent-router"
	"github.com/Chloe199719/agent-router/pkg/types"
	"github.com/joho/godotenv"
)

func main() {
	// Create a router with multiple providers
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, relying on environment variables")
	}
	r, err := router.New(
		router.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
		router.WithAnthropic(os.Getenv("ANTHROPIC_API_KEY")),
		router.WithGoogle(os.Getenv("GOOGLE_API_KEY")),
	)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Example 1: Basic completion with OpenAI
	fmt.Println("=== OpenAI Completion ===")
	resp, err := r.Complete(ctx, &types.CompletionRequest{
		Provider: types.ProviderOpenAI,
		Model:    "gpt-4o-mini",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Say hello in French"),
		},
	})
	if err != nil {
		log.Printf("OpenAI error: %v", err)
	} else {
		fmt.Printf("Response: %s\n", resp.Text())
		fmt.Printf("Tokens: %d input, %d output\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}

	// Example 2: Same request with Anthropic
	fmt.Println("\n=== Anthropic Completion ===")
	resp, err = r.Complete(ctx, &types.CompletionRequest{
		Provider:  types.ProviderAnthropic,
		Model:     "claude-3-5-haiku-20241022",
		MaxTokens: types.Ptr(100),
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Say hello in French"),
		},
	})
	if err != nil {
		log.Printf("Anthropic error: %v", err)
	} else {
		fmt.Printf("Response: %s\n", resp.Text())
		fmt.Printf("Tokens: %d input, %d output\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}

	// Example 3: Same request with Google
	fmt.Println("\n=== Google Completion ===")
	resp, err = r.Complete(ctx, &types.CompletionRequest{
		Provider: types.ProviderGoogle,
		Model:    "gemini-2.5-flash",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Say hello in French"),
		},
	})
	if err != nil {
		log.Printf("Google error: %v", err)
	} else {
		fmt.Printf("Response: %s\n", resp.Text())
		fmt.Printf("Tokens: %d input, %d output\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}

	// Example 4: Structured output (works with all providers)
	fmt.Println("\n=== Structured Output (OpenAI) ===")
	schema := types.JSONSchema{
		Type: "object",
		Properties: map[string]types.JSONSchema{
			"greeting": {Type: "string", Description: "The greeting in French"},
			"language": {Type: "string", Description: "The language code"},
		},
		Required: []string{"greeting", "language"},
	}

	resp, err = r.Complete(ctx, (&types.CompletionRequest{
		Provider: types.ProviderOpenAI,
		Model:    "gpt-4o-mini",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Give me a French greeting"),
		},
	}).WithJSONSchema("greeting_response", schema))

	if err != nil {
		log.Printf("Structured output error: %v", err)
	} else {
		fmt.Printf("Structured response: %s\n", resp.Text())
	}
}

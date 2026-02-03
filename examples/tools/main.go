// Example demonstrating tool/function calling with the agent router.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	router "github.com/Chloe199719/agent-router"
	"github.com/Chloe199719/agent-router/pkg/types"
	"github.com/joho/godotenv"
)

func main() {
	// Create a router with a provider
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, relying on environment variables")
	}
	r, err := router.New(
		router.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Define tools
	tools := []types.Tool{
		{
			Name:        "get_weather",
			Description: "Get the current weather for a location",
			Parameters: types.JSONSchema{
				Type: "object",
				Properties: map[string]types.JSONSchema{
					"location": {
						Type:        "string",
						Description: "The city and country, e.g., 'Paris, France'",
					},
					"unit": {
						Type:        "string",
						Enum:        []any{"celsius", "fahrenheit"},
						Description: "Temperature unit",
					},
				},
				Required: []string{"location"}, // unit is optional
			},
		},
		{
			Name:        "search_web",
			Description: "Search the web for information",
			Parameters: types.JSONSchema{
				Type: "object",
				Properties: map[string]types.JSONSchema{
					"query": {
						Type:        "string",
						Description: "The search query",
					},
				},
				Required: []string{"query"},
			},
		},
	}

	// Initial request
	fmt.Println("=== Tool Calling Example ===")
	resp, err := r.Complete(ctx, (&types.CompletionRequest{
		Provider: types.ProviderOpenAI,
		Model:    "gpt-4o-mini",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "What's the weather like in Tokyo?"),
		},
	}).WithTools(tools...))

	if err != nil {
		log.Fatal(err)
	}

	// Check if the model wants to use tools
	if resp.HasToolCalls() {
		fmt.Println("Model wants to use tools:")
		for _, tc := range resp.ToolCalls {
			inputJSON, _ := json.MarshalIndent(tc.Input, "", "  ")
			fmt.Printf("  - %s(%s)\n", tc.Name, string(inputJSON))
		}

		// Simulate tool execution and continue conversation
		messages := []types.Message{
			types.NewTextMessage(types.RoleUser, "What's the weather like in Tokyo?"),
		}

		// Add assistant's response with tool calls
		messages = append(messages, types.Message{
			Role:    types.RoleAssistant,
			Content: resp.Content,
		})

		// Add tool results
		for _, tc := range resp.ToolCalls {
			// Simulate tool execution
			var result string
			switch tc.Name {
			case "get_weather":
				result = `{"temperature": 22, "condition": "Partly cloudy", "humidity": 65}`
			case "search_web":
				result = `{"results": [{"title": "Example result", "url": "https://example.com"}]}`
			default:
				result = `{"error": "Unknown tool"}`
			}

			messages = append(messages, types.NewToolResultMessage(tc.ID, result, false))
		}

		// Continue conversation with tool results
		fmt.Println("\n=== Continuing with tool results ===")
		resp, err = r.Complete(ctx, (&types.CompletionRequest{
			Provider: types.ProviderOpenAI,
			Model:    "gpt-4o-mini",
			Messages: messages,
		}).WithTools(tools...))

		if err != nil {
			log.Fatal(err)
		}

		fmt.Printf("Final response: %s\n", resp.Text())
	} else {
		fmt.Printf("Response (no tool calls): %s\n", resp.Text())
	}
}

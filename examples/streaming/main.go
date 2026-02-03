// Example demonstrating streaming with the agent router.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	router "github.com/Chloe199719/agent-router"
	"github.com/Chloe199719/agent-router/pkg/types"
)

func main() {
	// Create a router with a provider
	r, err := router.New(
		router.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Stream a response
	fmt.Println("=== Streaming Response ===")
	stream, err := r.Stream(ctx, &types.CompletionRequest{
		Provider: types.ProviderOpenAI,
		Model:    "gpt-4o-mini",
		Messages: []types.Message{
			types.NewTextMessage(types.RoleUser, "Tell me a short story about a robot in 3 sentences."),
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer stream.Close()

	// Process streaming events
	for {
		event, err := stream.Next()
		if err != nil {
			log.Fatal(err)
		}
		if event == nil {
			break // Stream complete
		}

		switch event.Type {
		case types.StreamEventStart:
			fmt.Printf("[Started, model: %s]\n", event.Model)
		case types.StreamEventContentDelta:
			fmt.Print(event.Delta.Text)
		case types.StreamEventToolCallStart:
			fmt.Printf("\n[Tool call: %s]\n", event.ToolCall.Name)
		case types.StreamEventDone:
			fmt.Printf("\n[Done, stop reason: %s]\n", event.StopReason)
			if event.Usage != nil {
				fmt.Printf("[Tokens: %d input, %d output]\n", event.Usage.InputTokens, event.Usage.OutputTokens)
			}
		case types.StreamEventError:
			fmt.Printf("\n[Error: %v]\n", event.Error)
		}
	}

	// Get the accumulated response
	resp := stream.Response()
	if resp != nil {
		fmt.Printf("\nFull response text: %s\n", resp.Text())
	}
}

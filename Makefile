.PHONY: build test test-integration test-openai test-anthropic test-google lint clean

# Build the library
build:
	go build ./...

# Run unit tests (no API calls)
test:
	go test -v ./pkg/...

# Run integration tests (requires API keys)
test-integration:
	go test -v -tags=integration ./tests/...

# Run only OpenAI tests
test-openai:
	go test -v -tags=integration -run "OpenAI" ./tests/...

# Run only Anthropic tests
test-anthropic:
	go test -v -tags=integration -run "Anthropic" ./tests/...

# Run only Google tests
test-google:
	go test -v -tags=integration -run "Google" ./tests/...

# Run specific test categories
test-completion:
	go test -v -tags=integration -run "BasicCompletion" ./tests/...

test-streaming:
	go test -v -tags=integration -run "Streaming" ./tests/...

test-structured:
	go test -v -tags=integration -run "StructuredOutput" ./tests/...

test-tools:
	go test -v -tags=integration -run "ToolCalling" ./tests/...

# Lint
lint:
	go vet ./...
	@which staticcheck > /dev/null 2>&1 && staticcheck ./... || echo "staticcheck not installed"

# Clean
clean:
	go clean ./...

# Install development dependencies
dev-deps:
	go install honnef.co/go/tools/cmd/staticcheck@latest

# Show help
help:
	@echo "Available targets:"
	@echo "  build            - Build the library"
	@echo "  test             - Run unit tests"
	@echo "  test-integration - Run all integration tests (requires API keys)"
	@echo "  test-openai      - Run OpenAI tests only"
	@echo "  test-anthropic   - Run Anthropic tests only"
	@echo "  test-google      - Run Google tests only"
	@echo "  test-completion  - Run basic completion tests"
	@echo "  test-streaming   - Run streaming tests"
	@echo "  test-structured  - Run structured output tests"
	@echo "  test-tools       - Run tool calling tests"
	@echo "  lint             - Run linters"
	@echo "  clean            - Clean build artifacts"
	@echo ""
	@echo "Required environment variables for integration tests:"
	@echo "  OPENAI_API_KEY    - OpenAI API key"
	@echo "  ANTHROPIC_API_KEY - Anthropic API key"
	@echo "  GOOGLE_API_KEY    - Google API key (optional)"

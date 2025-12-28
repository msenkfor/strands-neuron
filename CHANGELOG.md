# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-10

### Added

- Initial release
- AWS vLLM on Neuron model provider for Strands Agents SDK
- Support for OpenAI-compatible API endpoints
- Full streaming support with async generators
- Tool/function calling support
- Structured output support via tool calls
- Comprehensive unit and integration tests
- Example implementations:
  - Weather agent example
  - MCP (Model Context Protocol) integration example
  - Streaming examples
- Configuration options for:
  - Model selection
  - Temperature, top_p, max_tokens
  - Stop sequences
  - Tensor parallel configuration
  - Neuron-specific settings
- Pyproject.toml with modern Python packaging
- Formatting and linting configuration (Ruff, Black, mypy)
- Docker infrastructure for vLLM Neuron server deployment


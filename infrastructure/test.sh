export VLLM_ENDPOINT="http://localhost:8080"

echo "Testing vLLM API with curl..."
curl -X POST "$VLLM_ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the significance of Bells inequality in quantum mechanics?"}],
    "max_tokens": 900,
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'

echo -e "\n\nBasic API tests completed!"
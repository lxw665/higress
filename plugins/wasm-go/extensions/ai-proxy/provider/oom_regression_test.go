package provider

import (
	"bytes"
	"fmt"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOmitClaudeInternalFieldsEquivalence(t *testing.T) {
	body := []byte(`{
		"model": "kimi-k3-gs",
		"max_tokens": 4096,
		"stream": true,
		"thinking": {"type": "enabled", "budget_tokens": 8000},
		"output_config": {"effort": "high"},
		"anthropic_beta": ["interleaved-thinking-2025-05-14"],
		"system": "x-anthropic-billing-header: cc_version=2.1.37; cch=abc123;",
		"messages": [
			{
				"role": "assistant",
				"content": [
					{"type": "thinking", "thinking": "plan edits", "signature": "sig-1"},
					{"type": "redacted_thinking", "data": "opaque"},
					{"type": "text", "text": "applying"},
					{"type": "tool_use", "id": "call_1", "name": "edit", "input": {"path": "a.go", "ok": true}},
					{"type": "tool_use", "id": "call_2", "name": "noop", "input": {}}
				]
			},
			{
				"role": "user",
				"content": [
					{"type": "tool_result", "tool_use_id": "call_1", "content": "done"},
					{"type": "tool_result", "tool_use_id": "call_2"}
				]
			}
		]
	}`)

	converter := &ClaudeToOpenAIConverter{}
	for _, preserveReasoning := range []bool{false, true} {
		withInternal, err := converter.ConvertClaudeRequestToOpenAIWithOptions(body, ClaudeToOpenAIConvertOptions{
			PreserveMessageReasoningContent: preserveReasoning,
			OmitClaudeInternalFields:        false,
		})
		require.NoError(t, err)
		stripped := stripClaudeInternalMessageFields(withInternal, preserveReasoning)

		withoutInternal, err := converter.ConvertClaudeRequestToOpenAIWithOptions(body, ClaudeToOpenAIConvertOptions{
			PreserveMessageReasoningContent: preserveReasoning,
			OmitClaudeInternalFields:        true,
		})
		require.NoError(t, err)
		assert.JSONEq(t, string(stripped), string(withoutInternal))
	}
}

func TestLargeToolUseConversionMemoryBound(t *testing.T) {
	var hunks bytes.Buffer
	hunks.WriteByte('[')
	filler := strings.Repeat("x", 160)
	for i := 0; i < 2000; i++ {
		if i > 0 {
			hunks.WriteByte(',')
		}
		fmt.Fprintf(&hunks, `{"file":"pkg/f_%d.go","old":"%s","new":"%s","meta":{"ok":true,"n":%d}}`, i, filler, filler, i)
	}
	hunks.WriteByte(']')
	req := []byte(fmt.Sprintf(`{"model":"kimi-k3-gs","max_tokens":256,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"patch","input":{"edits":%s}}]}]}`, hunks.String()))

	runtime.GC()
	var m0, m1 runtime.MemStats
	runtime.ReadMemStats(&m0)

	out, err := ReplaceByCustomSettings(req, nil)
	require.NoError(t, err)
	converter := &ClaudeToOpenAIConverter{}
	out, err = converter.ConvertClaudeRequestToOpenAIWithOptions(out, ClaudeToOpenAIConvertOptions{
		OmitClaudeInternalFields: true,
	})
	require.NoError(t, err)
	out, err = convertDeveloperRoleToSystem(out)
	require.NoError(t, err)

	runtime.ReadMemStats(&m1)
	totalAlloc := m1.TotalAlloc - m0.TotalAlloc
	assert.Less(t, totalAlloc, uint64(len(req))*15, "cumulative heap allocation should stay bounded (<15x vs legacy >400x)")
}

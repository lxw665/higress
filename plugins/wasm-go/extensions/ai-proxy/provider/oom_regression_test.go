package provider

import (
	"bytes"
	"encoding/json"
	"fmt"
	"runtime"
	"strings"
	"testing"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/proxytest"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
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

func TestBedrockToolUseInputPreservesRawJSONAndErrorsOnInvalid(t *testing.T) {
	bedrock := &bedrockProvider{}
	openaiBody, err := (&ClaudeToOpenAIConverter{}).ConvertClaudeRequestToOpenAI([]byte(`{
		"model": "claude-sonnet-4",
		"messages": [{
			"role": "assistant",
			"content": [
				{"type": "tool_use", "id": "toolu_1", "name": "measure", "input": {"unit": "m", "value": 1e400}}
			]
		}]
	}`))
	require.NoError(t, err)

	var openaiReq chatCompletionRequest
	require.NoError(t, json.Unmarshal(openaiBody, &openaiReq))

	bedrockBody, err := bedrock.buildBedrockTextGenerationRequest(&openaiReq, nil)
	require.NoError(t, err)
	assert.Equal(t, "m", gjson.GetBytes(bedrockBody, "messages.0.content.0.toolUse.input.unit").String())
	assert.Equal(t, "1e400", gjson.GetBytes(bedrockBody, "messages.0.content.0.toolUse.input.value").Raw)

	_, err = claudeToolUseInput(claudeRawJSONObject(`{"broken":`))
	assert.Error(t, err)
}

type testHttpCtx struct {
	types.DefaultHttpContext
	fn func(body []byte) (types.Action, error)
}

func (h *testHttpCtx) OnHttpRequestBody(bodySize int, endOfStream bool) types.Action {
	body, err := proxytestGetBody(bodySize)
	if err != nil {
		return types.ActionPause
	}
	action, err := h.fn(body)
	if err != nil {
		return types.ActionPause
	}
	return action
}

type testPluginCtx struct {
	types.DefaultPluginContext
	fn func(body []byte) (types.Action, error)
}

func (p *testPluginCtx) NewHttpContext(uint32) types.HttpContext { return &testHttpCtx{fn: p.fn} }

type testVMCtx struct {
	types.DefaultVMContext
	fn func(body []byte) (types.Action, error)
}

func (v *testVMCtx) NewPluginContext(uint32) types.PluginContext { return &testPluginCtx{fn: v.fn} }

func proxytestGetBody(size int) ([]byte, error) {
	return proxywasm.GetHttpRequestBody(0, size)
}

func TestOpenAIOnRequestBodyMultiTurnMemoryBound(t *testing.T) {
	var cfg ProviderConfig
	cfg.FromJson(gjson.Parse(`{
		"type": "openai",
		"apiTokens": ["sk-test-token"],
		"modelMapping": {"*": "kimi-k3-gs"}
	}`))
	require.NoError(t, cfg.Validate())
	p, err := CreateProvider(cfg)
	require.NoError(t, err)
	bodyHandler, ok := p.(RequestBodyHandler)
	require.True(t, ok)

	ctx := newMockMultipartHttpContext()
	ctx.SetContext("needClaudeResponseConversion", true)
	var handlerErr error

	opt := proxytest.NewEmulatorOption().WithVMContext(&testVMCtx{
		fn: func(body []byte) (types.Action, error) {
			var act types.Action
			act, handlerErr = bodyHandler.OnRequestBody(ctx, ApiNameChatCompletion, body)
			return act, handlerErr
		},
	})
	host, reset := proxytest.NewHostEmulator(opt)
	defer reset()
	require.Equal(t, types.OnPluginStartStatusOK, host.StartPlugin())
	contextID := host.InitializeHttpContext()
	host.CallOnRequestHeaders(contextID, [][2]string{
		{":path", "/v1/messages"},
		{"content-type", "application/json"},
	}, false)

	// Build a 16-round (34 messages after conversion) multi-turn Coding Agent history (~680 KB)
	var msgs bytes.Buffer
	msgs.WriteString(`[{"role":"user","content":[{"type":"text","text":"inspect and patch modules"}]}`)
	filler := strings.Repeat("x", 140)
	for r := 0; r < 16; r++ {
		var hunks bytes.Buffer
		hunks.WriteByte('[')
		for h := 0; h < 120; h++ {
			if h > 0 {
				hunks.WriteByte(',')
			}
			fmt.Fprintf(&hunks, `{"file":"pkg/m_%d/f_%d.go","old":"%s","new":"%s","meta":{"ok":true,"n":%d}}`, r, h, filler, filler, h)
		}
		hunks.WriteByte(']')
		fmt.Fprintf(&msgs, `,{"role":"assistant","content":[{"type":"thinking","thinking":"round %d plan","signature":"sig_%d"},{"type":"tool_use","id":"call_%d","name":"apply_patch","input":{"edits":%s}}]}`, r, r, r, hunks.String())
		fmt.Fprintf(&msgs, `,{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_%d","content":"ok %d"}]}`, r, r)
	}
	msgs.WriteByte(']')
	req := []byte(fmt.Sprintf(`{"model":"kimi-k3.1-pro","max_tokens":512,"thinking":{"type":"enabled","budget_tokens":4096},"system":"You are a coding agent.","messages":%s}`, msgs.String()))

	runtime.GC()
	var m0, m1 runtime.MemStats
	runtime.ReadMemStats(&m0)

	action := host.CallOnRequestBody(contextID, req, true)
	require.NoError(t, handlerErr)
	assert.Equal(t, types.ActionContinue, action)

	runtime.ReadMemStats(&m1)
	totalAlloc := m1.TotalAlloc - m0.TotalAlloc

	replacedBody := host.GetCurrentRequestBody(contextID)
	require.NotEmpty(t, replacedBody)
	assert.Equal(t, "kimi-k3-gs", gjson.GetBytes(replacedBody, "model").String())
	assert.Equal(t, int64(34), gjson.GetBytes(replacedBody, "messages.#").Int())
	assert.False(t, gjson.GetBytes(replacedBody, "claude_thinking").Exists())
	assert.False(t, gjson.GetBytes(replacedBody, "messages.2.claude_content_blocks").Exists())

	t.Logf("16-round OnRequestBody: input=%.2f KB, totalAlloc=%.2f KB (%.1fx)",
		float64(len(req))/1024, float64(totalAlloc)/1024, float64(totalAlloc)/float64(len(req)))
	assert.Less(t, totalAlloc, uint64(len(req))*16,
		"multi-turn OpenAI OnRequestBody heap allocation must stay <16x input size (reverted strip loop exceeds >100x)")
}

func TestConvertDeveloperRoleToSystemEarlyStopAndZeroAlloc(t *testing.T) {
	var tail bytes.Buffer
	chunk := strings.Repeat("a", 5000)
	for i := 0; i < 50; i++ {
		fmt.Fprintf(&tail, `,{"role":"user","content":"%s-%d"}`, chunk, i)
	}

	noDevBody := []byte(fmt.Sprintf(`{"model":"gpt-4o","messages":[{"role":"system","content":"init"}%s]}`, tail.String()))
	noDevAllocs := testing.AllocsPerRun(50, func() {
		out, err := convertDeveloperRoleToSystem(noDevBody)
		if err != nil || &out[0] != &noDevBody[0] {
			t.Fatal(err)
		}
	})
	assert.LessOrEqual(t, noDevAllocs, 2.0, "no-developer fast path should perform at most scanner allocation without copying body")

	withDevBody := []byte(fmt.Sprintf(`{"model":"gpt-4o","messages":[{"role":"developer","content":"init"}%s]}`, tail.String()))
	baselineAllocs := testing.AllocsPerRun(20, func() {
		req := &chatCompletionRequest{}
		if err := json.Unmarshal(withDevBody, req); err != nil {
			t.Fatal(err)
		}
		req.Messages[0].Role = roleSystem
		if _, err := json.Marshal(req); err != nil {
			t.Fatal(err)
		}
	})
	actualAllocs := testing.AllocsPerRun(20, func() {
		out, err := convertDeveloperRoleToSystem(withDevBody)
		if err != nil || len(out) == 0 {
			t.Fatal(err)
		}
	})
	t.Logf("256 KiB developer-at-index-0 allocs: baseline=%.1f, actual=%.1f", baselineAllocs, actualAllocs)
	assert.LessOrEqual(t, actualAllocs, baselineAllocs+2, "precheck should stop at first developer message without scanning remaining messages")
}

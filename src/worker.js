const DEFAULT_UPSTREAM_BASE_URL = "https://apihub.agnes-ai.com";
const DEFAULT_OPENAI_MODEL = "agnes-5";
const DEFAULT_CLAUDE_MODEL = "agnes-5";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
  "Access-Control-Allow-Headers": "authorization,content-type,x-api-key,anthropic-api-key,anthropic-version,anthropic-beta,openai-beta",
  "Access-Control-Expose-Headers": "content-type,request-id,x-request-id",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const path = normalizePath(url.pathname);

    // 公开端点无需认证
    if (path === "/" || path === "/health") {
      return jsonResponse(serviceInfo(request, env));
    }

    try {
      const authError = validateWorkerApiKey(request, env);
      if (authError) return authError;

      if (path.startsWith("/api/")) {
        return proxyUpstream(request, env, path);
      }

      if (path === "/mcp" || path === "/v1/mcp" || path === "/anthropic/mcp" || path === "/anthropic/v1/mcp") {
        return jsonResponse(mcpInfo(request));
      }

      if (path === "/codex" || path === "/v1/codex" || path === "/anthropic/codex" || path === "/anthropic/v1/codex") {
        return textResponse(codexSetup(request), "text/plain; charset=utf-8");
      }

      if (path === "/v1/setup" || path === "/anthropic/setup" || path === "/anthropic/v1/setup") {
        return textResponse(agentSetup(request), "text/plain; charset=utf-8");
      }

      if (path === "/v1/messages" || (path === "/v1/models" && looksLikeAnthropicRequest(request)) || path.startsWith("/anthropic/")) {
        return handleAnthropic(request, env, path);
      }

      if (path.startsWith("/v1/")) {
        return handleOpenAI(request, env, path);
      }

      return errorResponse(404, "not_found", `No route for ${path}`);
    } catch (error) {
      return errorResponse(500, "internal_error", error && error.message ? error.message : String(error));
    }
  },
};

async function handleOpenAI(request, env, path) {
  if ((path === "/v1/key" || path === "/v1/auth-key" || path === "/v1/usage") && request.method === "GET") {
    const rawPath = path === "/v1/usage" ? "/api/usage" : "/api/key";
    return proxyUpstream(request, env, rawPath);
  }

  if (path === "/v1/models" && request.method === "GET") {
    if (looksLikeAnthropicRequest(request)) {
      return anthropicModels(request, env);
    }
    return openAIModels(request, env);
  }

  if (path === "/v1/search" && request.method === "POST") {
    const body = await readJson(request);
    return openAIDirectCapability(request, env, body, "/api/search");
  }

  if (path === "/v1/merge" && request.method === "POST") {
    const body = await readJson(request);
    return openAIDirectCapability(request, env, body, "/api/merge");
  }

  if (path === "/v1/chat/completions" && request.method === "POST") {
    const body = await readJson(request);
    return openAIChatCompletions(request, env, body);
  }

  if (path === "/v1/responses" && request.method === "POST") {
    const body = await readJson(request);
    return openAIResponses(request, env, body);
  }

  if (path === "/v1/files" && request.method === "GET") {
    return jsonResponse({ object: "list", data: [], has_more: false });
  }

  if (path === "/v1/files" && request.method === "POST") {
    return openAIFileUpload(request, env);
  }

  if ((path === "/v1/files/extract" || path === "/v1/attachments/extract") && request.method === "POST") {
    const body = await readJson(request);
    const extracted = await callUpstreamJson(request, env, "/api/attachments/extract", body);
    return jsonResponse(extracted);
  }

  if (path.startsWith("/v1/files/") && request.method === "GET") {
    return errorResponse(404, "not_found", "This Worker is stateless. Bind KV/R2 if you need persisted OpenAI file retrieval.");
  }

  if (path === "/v1/embeddings" || path.startsWith("/v1/audio/") || path.startsWith("/v1/images/")) {
    return errorResponse(501, "unsupported_endpoint", `${path} is not exposed by agnes-ai and cannot be emulated faithfully.`);
  }

  return errorResponse(404, "not_found", `Unsupported OpenAI-compatible route ${path}`);
}

async function openAIDirectCapability(request, env, body, route) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `chatcmpl_${randomId()}`;
  const payload = buildUpstreamPayload({ ...body, web_search: route === "/api/search", merge: route === "/api/merge" }, route);

  if (body.stream !== false) {
    const upstream = await callUpstreamStream(request, env, route, payload);
    return sseResponse(streamOpenAIChat(upstream, { id, created, model }));
  }

  const result = await collectUpstreamText(request, env, route, payload);
  return jsonResponse({
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: result.text },
        logprobs: null,
        finish_reason: result.finishReason || "stop",
      },
    ],
    usage: usageFromText(payload.message || payload.query || "", result.text),
    system_fingerprint: `agnes-ai-worker:${route}`,
  });
}

async function openAIChatCompletions(request, env, body) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `chatcmpl_${randomId()}`;
  const route = chooseUpstreamRoute(body);
  const payload = buildUpstreamPayload(body, route);

  if (body.stream) {
    const upstream = await callUpstreamStream(request, env, route, payload);
    return sseResponse(streamOpenAIChat(upstream, { id, created, model }));
  }

  const result = await collectUpstreamText(request, env, route, payload);
  return jsonResponse({
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: result.text },
        logprobs: null,
        finish_reason: result.finishReason || "stop",
      },
    ],
    usage: usageFromText(payload.message || "", result.text),
    system_fingerprint: "agnes-ai-worker",
  });
}

async function openAIResponses(request, env, body) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `resp_${randomId()}`;
  const syntheticChatBody = responsesToChatBody(body, model);
  const route = chooseUpstreamRoute(syntheticChatBody);
  const payload = buildUpstreamPayload(syntheticChatBody, route);

  if (body.stream) {
    const upstream = await callUpstreamStream(request, env, route, payload);
    return sseResponse(streamOpenAIResponses(upstream, { id, created, model }));
  }

  const result = await collectUpstreamText(request, env, route, payload);
  return jsonResponse({
    id,
    object: "response",
    created_at: created,
    status: "completed",
    error: null,
    incomplete_details: null,
    instructions: body.instructions || null,
    max_output_tokens: body.max_output_tokens || body.max_tokens || null,
    model,
    output: [
      {
        id: `msg_${randomId()}`,
        type: "message",
        status: "completed",
        role: "assistant",
        content: [{ type: "output_text", text: result.text, annotations: [] }],
      },
    ],
    output_text: result.text,
    parallel_tool_calls: true,
    previous_response_id: body.previous_response_id || null,
    reasoning: body.reasoning || null,
    store: body.store || false,
    temperature: body.temperature || null,
    text: body.text || { format: { type: "text" } },
    tool_choice: body.tool_choice || "auto",
    tools: body.tools || [],
    top_p: body.top_p || null,
    truncation: body.truncation || "disabled",
    usage: responseUsageFromText(payload.message || "", result.text),
  });
}

async function openAIFileUpload(request, env) {
  const body = await readJson(request);
  const fileData = body.file || body;
  const route = "/api/attachments/upload";
  const result = await callUpstreamJson(request, env, route, fileData);
  return jsonResponse(result);
}

async function handleAnthropic(request, env, path) {
  const actualPath = path.startsWith("/anthropic/") ? path.replace(/^\/anthropic/, "") : path;
  const upstreamPath = actualPath.startsWith("/") ? actualPath : `/${actualPath}`;

  if (path === "/v1/models" && request.method === "GET") {
    return anthropicModels(request, env);
  }

  if (path === "/v1/messages" && request.method === "POST") {
    const body = await readJson(request);
    return anthropicMessages(request, env, body);
  }

  if (path.startsWith("/anthropic/")) {
    const route = actualPath;
    if (route === "/messages" && request.method === "POST") {
      const body = await readJson(request);
      return anthropicMessages(request, env, body);
    }
    if (route === "/messages/count_tokens" && request.method === "POST") {
      const body = await readJson(request);
      return anthropicCountTokens(request, env, body);
    }
  }

  return proxyUpstream(request, env, upstreamPath);
}

async function anthropicMessages(request, env, body) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_CLAUDE_MODEL;
  const stream = body.stream !== false;
  const route = "/api/chat";
  const payload = buildAnthropicPayload(body);

  if (stream) {
    const upstream = await callUpstreamStream(request, env, route, payload);
    return sseResponse(streamAnthropic(upstream, { model }));
  }

  const result = await collectUpstreamText(request, env, route, payload);
  return jsonResponse({
    id: `msg_${randomId()}`,
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: result.text }],
    stop_reason: result.finishReason || "end_turn",
    stop_sequence: null,
    usage: { input_tokens: 0, output_tokens: result.outputTokens || 0 },
  });
}

async function anthropicCountTokens(request, env, body) {
  return jsonResponse({ input_tokens: 0 });
}

async function anthropicModels(request, env) {
  return jsonResponse({
    data: [{ id: DEFAULT_CLAUDE_MODEL, type: "model", name: DEFAULT_CLAUDE_MODEL }],
    stop_reason: null,
    trim_transcript_usd: null,
    type: "list",
  });
}

async function openAIModels(request, env) {
  return jsonResponse({
    data: [{ id: DEFAULT_OPENAI_MODEL, object: "model", created: nowSeconds(), owned_by: "agnes-ai" }],
    object: "list",
  });
}

// ========== 辅助函数 ==========

async function proxyUpstream(request, env, path) {
  const upstreamUrl = `${DEFAULT_UPSTREAM_BASE_URL}${path}`;
  const apiKey = env.AGNES_API_KEY || env.UNLIMITED_SURF_API_KEY || "";

  const headers = new Headers(request.headers);
  headers.set("Authorization", `Bearer ${apiKey}`);
  headers.delete("host");

  try {
    const response = await fetch(upstreamUrl, {
      method: request.method,
      headers,
      body: request.method !== "GET" ? request.body : null,
    });

    const responseHeaders = new Headers(response.headers);
    Object.entries(CORS_HEADERS).forEach(([key, value]) => {
      responseHeaders.set(key, value);
    });

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    return errorResponse(502, "upstream_error", `Upstream request failed: ${error.message}`);
  }
}

async function callUpstreamJson(request, env, route, body) {
  const upstreamUrl = `${DEFAULT_UPSTREAM_BASE_URL}${route}`;
  const apiKey = env.AGNES_API_KEY || env.UNLIMITED_SURF_API_KEY || "";

  const headers = {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${apiKey}`,
    ...Object.fromEntries(request.headers),
  };

  const response = await fetch(upstreamUrl, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Upstream ${response.status}: ${errorText}`);
  }

  return response.json();
}

async function callUpstreamStream(request, env, route, body) {
  const upstreamUrl = `${DEFAULT_UPSTREAM_BASE_URL}${route}`;
  const apiKey = env.AGNES_API_KEY || env.UNLIMITED_SURF_API_KEY || "";

  const headers = {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${apiKey}`,
    "Accept": "text/event-stream",
    ...Object.fromEntries(request.headers),
  };

  return fetch(upstreamUrl, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
}

async function collectUpstreamText(request, env, route, body) {
  const upstream = await callUpstreamStream(request, env, route, body);
  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let text = "";
  let finishReason = "stop";
  let outputTokens = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split("\n");

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;

      try {
        const json = JSON.parse(data);
        if (json.choices?.[0]?.delta?.content) {
          text += json.choices[0].delta.content;
        }
        if (json.choices?.[0]?.finish_reason) {
          finishReason = json.choices[0].finish_reason;
        }
        if (json.usage?.completion_tokens) {
          outputTokens = json.usage.completion_tokens;
        }
      } catch {
        // ignore
      }
    }
  }

  return { text, finishReason, outputTokens };
}

function buildUpstreamPayload(body, route) {
  const payload = { ...body };

  if (route === "/api/search") {
    payload.web_search = true;
    payload.query = body.messages?.[body.messages.length - 1]?.content || body.query || "";
  } else if (route === "/api/merge") {
    payload.merge = true;
    payload.models = body.models || [DEFAULT_OPENAI_MODEL];
  } else {
    payload.message = body.messages?.[body.messages.length - 1]?.content || body.message || "";
  }

  return payload;
}

function buildAnthropicPayload(body) {
  const messages = body.messages || [];
  const lastMessage = messages[messages.length - 1] || {};

  return {
    message: lastMessage.content,
    model: body.model || DEFAULT_CLAUDE_MODEL,
    stream: body.stream,
  };
}

function responsesToChatBody(body, model) {
  const messages = [];

  if (body.instructions) {
    messages.push({ role: "system", content: body.instructions });
  }

  const input = Array.isArray(body.input) ? body.input : [body.input];
  for (const item of input) {
    if (item.type === "message") {
      messages.push({ role: item.role, content: item.content });
    } else if (item.type === "function_call_output") {
      messages.push({ role: "tool", content: JSON.stringify(item.output) });
    }
  }

  return { model, messages, stream: body.stream };
}

function chooseUpstreamRoute(body) {
  if (body.web_search_options || body.query) return "/api/search";
  if (body.merge || (Array.isArray(body.models) && body.models.length > 1)) return "/api/merge";
  return "/api/chat";
}

function streamOpenAIChat(upstream, config) {
  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  const { id, created, model } = config;

  return new ReadableStream({
    async start(controller) {
      try {
        let buffer = "";
        let firstChunk = true;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (data === "[DONE]") {
              controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
              return;
            }

            try {
              const json = JSON.parse(data);
              const content = json.choices?.[0]?.delta?.content || "";
              const finishReason = json.choices?.[0]?.finish_reason;

              if (firstChunk || content) {
                const chunk = {
                  id,
                  object: "chat.completion.chunk",
                  created,
                  model,
                  choices: [{ index: 0, delta: content ? { content } : {}, finish_reason: finishReason }],
                };
                controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`));
                firstChunk = false;
              }
            } catch {
              // skip
            }
          }
        }
      } catch (error) {
        controller.error(error);
      } finally {
        controller.close();
      }
    },
  });
}

function streamOpenAIResponses(upstream, config) {
  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  const { id, created, model } = config;

  return new ReadableStream({
    async start(controller) {
      try {
        let buffer = "";
        let outputText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (data === "[DONE]") continue;

            try {
              const json = JSON.parse(data);
              const content = json.choices?.[0]?.delta?.content || "";
              outputText += content;

              const chunk = { id, object: "response.output_text.delta", delta: content };
              controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`));
            } catch {
              // skip
            }
          }
        }

        const finalChunk = {
          id,
          object: "response.output_item.added",
          output: [{ id: `msg_${randomId()}`, type: "message", role: "assistant", content: [{ type: "output_text", text: outputText }] }],
        };
        controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(finalChunk)}\n\n`));
        controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
      } catch (error) {
        controller.error(error);
      } finally {
        controller.close();
      }
    },
  });
}

function streamAnthropic(upstream, config) {
  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  const { model } = config;

  return new ReadableStream({
    async start(controller) {
      try {
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            try {
              const json = JSON.parse(data);
              const eventType = json.type;

              if (eventType === "content_block_delta") {
                const text = json.delta?.text || "";
                const chunk = { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } };
                controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`));
              } else if (eventType === "message_delta") {
                const chunk = { type: "message_delta", delta: json.delta, index: 0 };
                controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`));
              }
            } catch {
              // skip
            }
          }
        }

        const stopChunk = { type: "message_stop" };
        controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(stopChunk)}\n\n`));
        controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
      } catch (error) {
        controller.error(error);
      } finally {
        controller.close();
      }
    },
  });
}

// ========== 工具函数 ==========

function sseResponse(stream) {
  return new Response(stream, {
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive", ...CORS_HEADERS },
  });
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

function textResponse(text, contentType = "text/plain; charset=utf-8") {
  return new Response(text, {
    headers: { "Content-Type": contentType, ...CORS_HEADERS },
  });
}

function errorResponse(status, errorCode, errorMessage) {
  return jsonResponse({
    error: { message: errorMessage, type: errorCode, code: status },
  }, status);
}

function normalizePath(path) {
  return path.replace(/\/+$/, "") || "/";
}

async function readJson(request) {
  const contentType = request.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return {};
  try { return await request.json(); } catch { return {}; }
}

function validateWorkerApiKey(request, env) {
  const workerApiKey = env.WORKER_API_KEY;
  if (!workerApiKey) return null;

  const url = new URL(request.url);
  const path = normalizePath(url.pathname);
  const publicPaths = ["/", "/health"];
  
  if (publicPaths.includes(path)) return null;

  const authHeader = request.headers.get("authorization") || "";
  const token = authHeader.replace(/^Bearer\s+/i, "").trim();

  if (!token || token !== workerApiKey) {
    return errorResponse(401, "invalid_request_error", "Invalid or missing API key");
  }
  return null;
}

function serviceInfo(request, env) {
  return {
    ok: true,
    service: "agnes-ai-transfer-worker",
    upstream: DEFAULT_UPSTREAM_BASE_URL,
    timestamp: new Date().toISOString(),
  };
}

function nowSeconds() { return Math.floor(Date.now() / 1000); }

function randomId(length = 21) {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  for (let i = 0; i < length; i++) result += chars.charAt(Math.floor(Math.random() * chars.length));
  return result;
}

function looksLikeAnthropicRequest(request) {
  const auth = request.headers.get("authorization") || "";
  const apiKey = request.headers.get("x-api-key") || "";
  const anthropicVersion = request.headers.get("anthropic-version") || "";
  return auth.includes("sk-ant") || apiKey.startsWith("sk-ant") || anthropicVersion !== "";
}

function usageFromText(input, output) {
  return { prompt_tokens: input.length, completion_tokens: output.length, total_tokens: input.length + output.length };
}

function responseUsageFromText(input, output) {
  return { input_tokens: input.length, output_tokens: output.length, total_tokens: input.length + output.length };
}

function mcpInfo(request) {
  return jsonResponse({
    message: "MCP servers must be configured in your local agent, IDE, or Claude Code/Codex environment.",
    documentation: `${DEFAULT_UPSTREAM_BASE_URL}/mcp`,
    setup: { note: "This Worker only provides the model API endpoint.", limitation: "It does not read or modify local files from Cloudflare edge." },
  });
}

function codexSetup(request) {
  return `# Codex CLI Setup\n\nSet the following environment variables before running Codex:\n\n\`\`\`bash\nexport OPENAI_BASE_URL="${DEFAULT_UPSTREAM_BASE_URL}/v1"\nexport OPENAI_API_KEY="<your WORKER_API_KEY>"\nexport OPENAI_MODEL="${DEFAULT_OPENAI_MODEL}"\n\`\`\`\n\nOr use directly:\n\`\`\`bash\ncodex --api-base-url "${DEFAULT_UPSTREAM_BASE_URL}/v1" --api-key "<your WORKER_API_KEY>"\n\`\`\``;
}

function agentSetup(request) {
  return `# Agent Setup Guide\n\n## OpenAI Compatible (recommended)\nBase URL: \`${DEFAULT_UPSTREAM_BASE_URL}/v1\`\nAPI Key: Your WORKER_API_KEY\n\n## Anthropic/Claude Compatible\nBase URL: \`${DEFAULT_UPSTREAM_BASE_URL}\`\nAPI Key: Your WORKER_API_KEY\n\n## Models Available\nUse any model ID supported by ${DEFAULT_UPSTREAM_BASE_URL}.\nCheck \`/v1/models\` for the full list.`;
}

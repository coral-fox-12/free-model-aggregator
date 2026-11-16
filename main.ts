// free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
//
// The design, in one paragraph: every prompt is served by the best-rated free model that meets its
// requirements, with deterministic failover. The brief ranks the fourteen free models by rating_milli;
// the best model for a plain prompt is thinkingmachines/inkling-2-small (1475), for a JSON-mode prompt
// nvidia/nemotron-3.5-super-vl (1303, the best JSON-capable one), for a tools prompt inkling again.
// When a gateway call is refused (429/5xx/network), the next-best candidate serves the prompt instead,
// and the refusal trail goes out on the x-makefun-upstream-failures header so the failover column can
// see it. An answer only counts when its own text carries the prompt's token (the scorer checks the
// provider's body for the token), so a successful call whose text lacks the token, or that only called
// a tool, is treated as unusable and the next candidate is tried. Everything a request depends on is
// a function of the brief, the request and the gateway answers seen so far: no wall clock and no
// unseeded randomness anywhere in a request path, so a replay makes exactly the live pass's requests.
//
// The wire translation (Anthropic <-> Chat Completions, the SSE relays, the model field) lives in
// lib/wire.ts and the brief/catalog parsing in lib/providers.ts, both adapted from the kit's starter.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most gateway calls one prompt may spend, across retries and failovers. */
const MAX_ATTEMPTS_PER_PROMPT = 6;
/** The trial's gateway call cap is 60; stop starting new attempts before it is near. */
const CALL_BUDGET_GUARD = 54;
/** A refusal trail row. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** How many times each model has refused in this trial (deterministic state: the replay replays the same refusals). */
const failureCounts = new Map<string, number>();
/** Gateway calls made in this trial, successful or not. */
let callsMade = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The candidates for a request, best first: every free model meeting the requirements, ordered by
 * rating_milli descending, brief order breaking ties. A model that has already refused twice in this
 * trial goes last, so a rate-limited model stops eating the trial's calls.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const ok = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  const rank = (model: ModelRow): number => (failureCounts.get(model.id) ?? 0) >= 2 ? 1 : 0;
  return ok
    .map((model, index) => ({ model, index }))
    .sort((a, b) => rank(a.model) - rank(b.model) || b.model.rating_milli - a.model.rating_milli || a.index - b.index)
    .map((entry) => entry.model);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The prompt's token, read from the request itself: the unique WORD-1234 the instruction carries. */
function tokenOf(body: Json): string | null {
  const texts: string[] = [];
  const collect = (value: unknown): void => {
    if (typeof value === 'string') texts.push(value);
    else if (Array.isArray(value)) value.forEach(collect);
    else if (isRecord(value)) Object.values(value).forEach(collect);
  };
  collect(body.messages ?? body.content ?? []);
  collect(body.system);
  const joined = texts.join('\n');
  const match = /[A-Z]{2,}-[0-9]{4}/.exec(joined);
  return match === null ? null : match[0];
}

/** The text of a Chat Completions answer's message, '' when it holds no text. */
function textOfAnswer(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const message = isRecord(choices[0]) ? (isRecord(choices[0].message) ? choices[0].message : {}) : {};
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((block) => (isRecord(block) && block.type === 'text' && typeof block.text === 'string' ? [block.text] : []))
      .join('');
  }
  return '';
}

/** Whether a Chat Completions answer is usable for this prompt: text, carrying the token, if one is known. */
function usableAnswer(answer: Json, token: string | null): boolean {
  const text = textOfAnswer(answer);
  if (text === '') return false;
  if (token !== null && !text.includes(token)) return false;
  return true;
}

/** The text of an OpenAI SSE body: every delta.content, in order. */
function textOfStream(bodyText: string): string {
  let text = '';
  for (const frame of bodyText.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (!isRecord(chunk)) continue;
        const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
        const delta = isRecord(choices[0]) ? (isRecord(choices[0].delta) ? choices[0].delta : {}) : {};
        if (typeof delta.content === 'string') text += delta.content;
      } catch {
        // A frame that is not JSON carries no text.
      }
    }
  }
  return text;
}

/** A ReadableStream of the bytes given, so the starter's stream relays can run over a buffered body. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: FailureRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, 8));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: FailureRow[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model to dial. */
async function dial(model: string, body: Json): Promise<Response> {
  callsMade += 1;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** The gateway error's code, from its body when it parses, else a plain label for the status. */
async function errorCode(res: Response): Promise<string> {
  try {
    const text = await res.text();
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.code === 'string') return parsed.error.code;
    if (isRecord(parsed) && typeof parsed.code === 'string') return parsed.code;
    if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.type === 'string') return parsed.error.type;
    return res.status === 429 ? 'rate_limited' : 'gateway_error';
  } catch {
    return res.status === 429 ? 'rate_limited' : 'gateway_error';
  }
}

/** Whether a refused gateway call is worth another candidate: 429/5xx/network, never 400/401. */
function retriable(status: number | null): boolean {
  if (status === null) return true;
  return status === 429 || status >= 500;
}

interface Served {
  ok: boolean;
  /** The model that served the prompt, when one did. */
  model: string | null;
  /** The upstream answer, JSON or SSE body text, when one was served. */
  answer: Json | string | null;
  /** The stream flag the upstream answer came with. */
  stream: boolean;
  /** Every refusal on the way here, for the trail header. */
  failures: FailureRow[];
  /** The last refusal's status, for the error when nothing served. */
  lastStatus: number;
  lastDetail: string;
}

/**
 * Serve one prompt through the gateway: the best candidate first, then the next-best on a refusal or
 * on an answer that cannot serve the prompt (no text, or text without the prompt's token). A 400/401
 * ends the prompt at once — another model would meet the same refusal — and so does the call budget.
 * Deterministic: every decision follows the gateway answers, never the clock.
 */
async function serve(openAIBody: Json, token: string | null, requires: Requirement): Promise<Served> {
  const candidates = candidatesFor(requires);
  const failures: FailureRow[] = [];
  let attempts = 0;
  for (const model of candidates) {
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT || callsMade >= CALL_BUDGET_GUARD) break;
    attempts += 1;
    let upstream: Response;
    try {
      upstream = await dial(model.id, openAIBody);
    } catch (err) {
      failureCounts.set(model.id, (failureCounts.get(model.id) ?? 0) + 1);
      failures.push({ status: 0, code: 'network_error', model: model.id });
      log(`dial ${model.id} failed to reach the gateway: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const status = upstream.status;
      const code = await errorCode(upstream);
      failureCounts.set(model.id, (failureCounts.get(model.id) ?? 0) + 1);
      failures.push({ status, code, model: model.id });
      log(`dial ${model.id} refused HTTP ${status} (${code})`);
      if (!retriable(status)) {
        return { ok: false, model: null, answer: null, stream: false, failures, lastStatus: status, lastDetail: code };
      }
      continue;
    }
    if (openAIBody.stream === true) {
      const bodyText = await upstream.text();
      const text = textOfStream(bodyText);
      if (token !== null && !text.includes(token)) {
        log(`stream from ${model.id} carries no token; failing over`);
        continue;
      }
      if (text === '' && token !== null) {
        log(`stream from ${model.id} carries no text; failing over`);
        continue;
      }
      return { ok: true, model: model.id, answer: bodyText, stream: true, failures, lastStatus: 200, lastDetail: '' };
    }
    const answer: unknown = await upstream.json();
    if (!isRecord(answer)) {
      log(`answer from ${model.id} is not one JSON object; failing over`);
      continue;
    }
    if (!usableAnswer(answer, token)) {
      log(`answer from ${model.id} cannot serve the prompt (no text or no token); failing over`);
      continue;
    }
    return { ok: true, model: model.id, answer, stream: false, failures, lastStatus: 200, lastDetail: '' };
  }
  const last = failures[failures.length - 1];
  return {
    ok: false,
    model: null,
    answer: null,
    stream: false,
    failures,
    lastStatus: last?.status ?? 502,
    lastDetail: last?.code ?? 'no_free_model',
  };
}

/** A refusal as a 502 in the client's wire, with the trail header when there is one. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', served: Served): void {
  const detail = `the gateway answered HTTP ${served.lastStatus}${served.lastDetail === '' ? '' : ` (${served.lastDetail})`}`;
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', detail, served.failures);
  else sendError(res, wire, 502, 'upstream_error', detail, served.failures);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const token = tokenOf(body);
  const served = await serve({ ...body }, token, requires);
  if (!served.ok || served.model === null || served.answer === null) return refused(res, 'openai', served);
  if (served.stream) {
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (served.failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(served.failures.slice(0, 8));
    res.writeHead(200, headers);
    try {
      for await (const frame of openAIStreamWithModel(streamOf(served.answer as string), served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIJsonWithModel(served.answer as Json, served.model), served.failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const token = tokenOf(body);
  const openAIBody = anthropicToOpenAI(body, '');
  const served = await serve(openAIBody, token, requires);
  if (!served.ok || served.model === null || served.answer === null) return refused(res, 'anthropic', served);
  if (served.stream) {
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (served.failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(served.failures.slice(0, 8));
    res.writeHead(200, headers);
    try {
      for await (const event of anthropicEventsFromOpenAI(streamOf(served.answer as string), served.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIToAnthropic(served.answer as Json, served.model), served.failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

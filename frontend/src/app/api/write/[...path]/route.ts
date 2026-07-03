export const runtime = "nodejs";

type WriteMethod = "POST" | "PUT" | "DELETE";
type WriteRouteContext = {
  params: Promise<{ path: string[] }>;
};

const ALLOWED_TARGETS: Readonly<Record<string, string>> = {
  "POST employees": "/api/employees",
  "DELETE employees": "/api/employees",
  "PUT shift-settings": "/api/shift-settings",
};

function configurationError() {
  return Response.json(
    { detail: "Server proxy not configured" },
    { status: 503 },
  );
}

async function proxyWrite(
  request: Request,
  method: WriteMethod,
  context: WriteRouteContext,
) {
  // CSRF guard: this proxy injects the API key for any caller, so a cross-site
  // form-data/POST (CORS "simple request", no preflight) could forge writes.
  // Reject when the browser sends an Origin that isn't our own host.
  const origin = request.headers.get("origin");
  if (origin) {
    const host = request.headers.get("host");
    let originHost: string | null = null;
    try {
      originHost = new URL(origin).host;
    } catch {
      originHost = null;
    }
    if (!host || originHost !== host) {
      return Response.json(
        { detail: "Cross-origin write blocked" },
        { status: 403 },
      );
    }
  }

  const { path } = await context.params;
  const targetPath = ALLOWED_TARGETS[`${method} ${path.join("/")}`];
  if (!targetPath) {
    return Response.json({ detail: "Not found" }, { status: 404 });
  }

  const apiKey = process.env.API_KEY;
  const backendInternalUrl = process.env.BACKEND_INTERNAL_URL;
  if (!apiKey || !backendInternalUrl) {
    return configurationError();
  }

  let targetUrl: URL;
  try {
    targetUrl = new URL(targetPath, backendInternalUrl);
  } catch {
    return configurationError();
  }
  if (targetUrl.protocol !== "http:" && targetUrl.protocol !== "https:") {
    return configurationError();
  }
  targetUrl.search = new URL(request.url).search;

  const headers = new Headers({ "X-API-Key": apiKey });
  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers.set("Content-Type", contentType);
  }

  let backendResponse: Response;
  try {
    backendResponse = await fetch(targetUrl, {
      method,
      headers,
      body: request.body ? await request.arrayBuffer() : undefined,
      cache: "no-store",
      redirect: "manual",
    });
  } catch {
    return Response.json({ detail: "Backend unavailable" }, { status: 502 });
  }

  const responseHeaders = new Headers();
  const responseContentType = backendResponse.headers.get("content-type");
  if (responseContentType) {
    responseHeaders.set("Content-Type", responseContentType);
  }

  return new Response(backendResponse.body, {
    status: backendResponse.status,
    statusText: backendResponse.statusText,
    headers: responseHeaders,
  });
}

export function POST(request: Request, context: WriteRouteContext) {
  return proxyWrite(request, "POST", context);
}

export function PUT(request: Request, context: WriteRouteContext) {
  return proxyWrite(request, "PUT", context);
}

export function DELETE(request: Request, context: WriteRouteContext) {
  return proxyWrite(request, "DELETE", context);
}

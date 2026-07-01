export const runtime = "nodejs";

type WriteMethod = "POST" | "PUT" | "DELETE";
type WriteRouteContext = {
  params: Promise<{ path: string[] }>;
};

const ALLOWED_TARGETS: Readonly<Record<string, string>> = {
  "POST employees": "/api/employees",
  "DELETE employees": "/api/employees",
  "PUT shift-settings": "/api/shift-settings",
  "POST attendance/checkin": "/api/attendance/checkin",
  "POST attendance/checkout": "/api/attendance/checkout",
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

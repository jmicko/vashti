import { invokeNative, isNativeRuntime } from "./runtime";

type ApiError = {
  error?: {
    code?: string;
    message?: string;
  };
};

type NativeHttpResponse = {
  status: number;
  headers: Record<string, string>;
  body_text: string | null;
  body_base64: string | null;
};

type NativeMultipartPart =
  | { kind: "text"; name: string; value: string }
  | {
      kind: "file";
      name: string;
      filename: string;
      mime_type: string;
      data_base64: string;
    };

export async function requestJson<T = unknown>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  if (isNativeRuntime()) {
    const bodyText = requestBodyText(options.body);
    const headers = headersRecord(options.headers);
    if (bodyText !== null && !headers["content-type"]) {
      headers["content-type"] = "application/json";
    }
    const response = await invokeNative<NativeHttpResponse>("native_http_request", {
      request: {
        method: options.method ?? "GET",
        path,
        headers,
        body_text: bodyText,
        response_type: "text"
      }
    });
    if (response.status < 200 || response.status >= 300) {
      throw new Error(nativeResponseErrorMessage(response));
    }
    return JSON.parse(response.body_text ?? "null") as T;
  }

  const response = await fetch(path, {
    credentials: "include",
    headers: {
      "content-type": "application/json",
      ...options.headers
    },
    ...options
  });

  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }

  return response.json() as Promise<T>;
}

export async function requestBlob(path: string, options: RequestInit = {}) {
  if (isNativeRuntime()) {
    const response = await invokeNative<NativeHttpResponse>("native_http_request", {
      request: {
        method: options.method ?? "GET",
        path,
        headers: headersRecord(options.headers),
        body_text: requestBodyText(options.body),
        response_type: "base64"
      }
    });
    if (response.status < 200 || response.status >= 300) {
      throw new Error(nativeResponseErrorMessage(response));
    }
    const mimeType = response.headers["content-type"] ?? "application/octet-stream";
    return new Blob([base64ToBytes(response.body_base64 ?? "")], { type: mimeType });
  }

  const response = await fetch(path, { credentials: "include", ...options });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }
  return response.blob();
}

export async function requestMultipartJson<T = unknown>(
  path: string,
  formData: FormData,
  options: Omit<RequestInit, "body"> = {}
): Promise<T> {
  if (isNativeRuntime()) {
    const parts = await nativeMultipartParts(formData);
    const response = await invokeNative<NativeHttpResponse>("native_http_multipart", {
      request: {
        method: options.method ?? "POST",
        path,
        headers: headersRecord(options.headers),
        parts
      }
    });
    if (response.status < 200 || response.status >= 300) {
      throw new Error(nativeResponseErrorMessage(response));
    }
    return JSON.parse(response.body_text ?? "null") as T;
  }

  const response = await fetch(path, {
    credentials: "include",
    ...options,
    body: formData
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }
  return response.json() as Promise<T>;
}

export async function responseErrorMessage(response: Response) {
  let message = `Request failed with ${response.status}`;
  try {
    const payload = (await response.json()) as ApiError;
    message = payload.error?.message ?? message;
  } catch {
    // Keep the status-derived message when the body is not JSON.
  }
  return message;
}

function requestBodyText(body: BodyInit | null | undefined) {
  if (body === undefined || body === null) {
    return null;
  }
  if (typeof body !== "string") {
    throw new Error("This native request body type is not supported");
  }
  return body;
}

function headersRecord(headers: HeadersInit | undefined) {
  const output: Record<string, string> = {};
  new Headers(headers).forEach((value, key) => {
    output[key] = value;
  });
  return output;
}

function nativeResponseErrorMessage(response: NativeHttpResponse) {
  let message = `Request failed with ${response.status}`;
  try {
    const payload = JSON.parse(response.body_text ?? "") as ApiError;
    return payload.error?.message ?? message;
  } catch {
    return message;
  }
}

async function nativeMultipartParts(formData: FormData): Promise<NativeMultipartPart[]> {
  const parts: NativeMultipartPart[] = [];
  for (const [name, value] of formData.entries()) {
    if (typeof value === "string") {
      parts.push({ kind: "text", name, value });
      continue;
    }
    parts.push({
      kind: "file",
      name,
      filename: value.name,
      mime_type: value.type || "application/octet-stream",
      data_base64: bytesToBase64(new Uint8Array(await value.arrayBuffer()))
    });
  }
  return parts;
}

function bytesToBase64(bytes: Uint8Array) {
  const chunkSize = 0x8000;
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function base64ToBytes(value: string) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

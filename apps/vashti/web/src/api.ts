type ApiError = {
  error?: {
    code?: string;
    message?: string;
  };
};

export async function requestJson<T = unknown>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
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

#[cfg(target_os = "android")]
mod connections;
#[cfg(target_os = "android")]
mod secrets;
#[cfg(target_os = "android")]
mod transport;
#[cfg(any(target_os = "android", test))]
mod validation;

#[cfg(target_os = "android")]
#[tauri::mobile_entry_point]
pub fn run() {
    use tauri::Manager;

    tauri::Builder::default()
        .register_asynchronous_uri_scheme_protocol("vashtiasset", |context, request, responder| {
            let app = context.app_handle().clone();
            let method = request.method().clone();
            let path = request.uri().path().trim_start_matches('/').to_string();
            tauri::async_runtime::spawn(async move {
                let response = if method != tauri::http::Method::GET {
                    media_error_response(
                        tauri::http::StatusCode::METHOD_NOT_ALLOWED,
                        "Only GET is allowed",
                    )
                } else if let Some((namespace, encoded_path)) = path.split_once('/') {
                    let state = app.state::<connections::NativeState>();
                    match transport::resolve_authenticated_media(
                        state.inner(),
                        namespace,
                        encoded_path,
                    )
                    .await
                    {
                        Ok((status, headers, body)) => {
                            let mut response = tauri::http::Response::builder()
                                .status(status)
                                .header("content-security-policy", "default-src 'none'; sandbox")
                                .header("referrer-policy", "no-referrer")
                                .header("x-content-type-options", "nosniff");
                            for (name, value) in headers {
                                response = response.header(name, value);
                            }
                            response.body(body).unwrap_or_else(|_| {
                                media_error_response(
                                    tauri::http::StatusCode::INTERNAL_SERVER_ERROR,
                                    "Could not build media response",
                                )
                            })
                        }
                        Err(error) => {
                            media_error_response(tauri::http::StatusCode::BAD_REQUEST, &error)
                        }
                    }
                } else {
                    media_error_response(
                        tauri::http::StatusCode::BAD_REQUEST,
                        "Media URL is invalid",
                    )
                };
                responder.respond(response);
            });
        })
        .setup(|app| {
            let state = connections::NativeState::load(app.handle())
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            connections::native_list_connections,
            connections::native_add_connection,
            connections::native_update_connection,
            connections::native_remove_connection,
            connections::native_select_connection,
            connections::native_sync_active_identity,
            transport::native_http_request,
            transport::native_http_multipart,
            transport::native_http_stream,
            transport::native_cancel_request,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Vashti Android");
}

#[cfg(target_os = "android")]
fn media_error_response(
    status: tauri::http::StatusCode,
    message: &str,
) -> tauri::http::Response<Vec<u8>> {
    tauri::http::Response::builder()
        .status(status)
        .header("content-type", "text/plain; charset=utf-8")
        .header("cache-control", "no-store")
        .header("content-security-policy", "default-src 'none'; sandbox")
        .header("referrer-policy", "no-referrer")
        .header("x-content-type-options", "nosniff")
        .body(message.as_bytes().to_vec())
        .expect("static media error response is valid")
}

#[cfg(not(target_os = "android"))]
pub fn run() {
    eprintln!("vashti-android is built and run through the Android target");
}

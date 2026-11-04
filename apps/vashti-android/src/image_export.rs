use serde::Serialize;
use tauri::{
    Manager, Runtime,
    plugin::{Builder, PluginHandle, TauriPlugin},
};

struct ImageExport<R: Runtime>(PluginHandle<R>);

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("image-export")
        .setup(|app, api| {
            app.manage(ImageExport(
                api.register_android_plugin("chat.vashti.app", "ImageExportPlugin")?,
            ));
            Ok(())
        })
        .build()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExportRequest {
    filename: String,
    mime_type: String,
    data: String,
}

#[tauri::command]
pub async fn native_save_image(
    app: tauri::AppHandle,
    filename: String,
    mime_type: String,
    data: String,
) -> Result<serde_json::Value, String> {
    if !matches!(
        mime_type.as_str(),
        "image/png" | "image/jpeg" | "image/gif" | "image/webp"
    ) {
        return Err("Invalid image export".into());
    }
    native_save_file(app, filename, mime_type, data).await
}

#[tauri::command]
pub async fn native_save_file(
    app: tauri::AppHandle,
    filename: String,
    mime_type: String,
    data: String,
) -> Result<serde_json::Value, String> {
    crate::validation::validate_file_export(&filename, &mime_type, data.len())?;
    app.state::<ImageExport<tauri::Wry>>()
        .0
        .run_mobile_plugin_async(
            "save",
            ExportRequest {
                filename,
                mime_type,
                data,
            },
        )
        .await
        .map_err(|_| "Could not save file".into())
}

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
    if filename.len() > 160
        || filename.contains(['/', '\\'])
        || !matches!(
            mime_type.as_str(),
            "image/png" | "image/jpeg" | "image/gif" | "image/webp"
        )
        || data.len() > 32 * 1024 * 1024
    {
        return Err("Invalid image export".into());
    }
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
        .map_err(|_| "Could not save image".into())
}

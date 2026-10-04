package chat.vashti.app

import android.app.Activity
import android.content.Intent
import android.provider.DocumentsContract
import android.util.Base64
import androidx.activity.result.ActivityResult
import app.tauri.annotation.ActivityCallback
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import java.util.concurrent.Executors

@InvokeArg
class ImageExportArgs {
  lateinit var filename: String
  lateinit var mimeType: String
  lateinit var data: String
}

@TauriPlugin
class ImageExportPlugin(private val activity: Activity) : Plugin(activity) {
  private var pending = false
  private val writer = Executors.newSingleThreadExecutor()

  @Command
  fun save(invoke: Invoke) {
    if (pending) {
      invoke.reject("An image save is already open")
      return
    }
    try {
      val args = invoke.parseArgs(ImageExportArgs::class.java)
      pending = true
      val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
        addCategory(Intent.CATEGORY_OPENABLE)
        type = args.mimeType
        putExtra(Intent.EXTRA_TITLE, args.filename)
      }
      startActivityForResult(invoke, intent, "saved")
    } catch (_: Exception) {
      pending = false
      invoke.reject("Could not open image save dialog")
    }
  }

  @ActivityCallback
  fun saved(invoke: Invoke, result: ActivityResult) {
    val uri = result.data?.data
    if (result.resultCode != Activity.RESULT_OK || uri == null) {
      pending = false
      invoke.resolve(JSObject().put("saved", false))
      return
    }
    writer.execute {
      try {
        val args = invoke.parseArgs(ImageExportArgs::class.java)
        val bytes = Base64.decode(args.data, Base64.DEFAULT)
        activity.contentResolver.openOutputStream(uri, "w")?.use { it.write(bytes) }
          ?: throw IllegalStateException("No output stream")
        invoke.resolve(JSObject().put("saved", true))
      } catch (_: Exception) {
        // Do not leave a corrupt or empty document behind on failure.
        try { DocumentsContract.deleteDocument(activity.contentResolver, uri) } catch (_: Exception) {}
        invoke.reject("Could not save image")
      } finally {
        activity.runOnUiThread { pending = false }
      }
    }
  }
}

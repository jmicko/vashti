# Add project specific ProGuard rules here.
# Native startup looks up this method by name. Keep it even when a shared Cargo
# cache skips regenerating proguard-tauri.pro in a fresh checkout.
-keep class chat.vashti.app.TauriActivity {
  public app.tauri.plugin.PluginManager getPluginManager();
}

# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

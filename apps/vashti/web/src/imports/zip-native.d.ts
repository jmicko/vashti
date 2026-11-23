// zip.js publishes these types through package exports, which the app's
// legacy TypeScript resolver does not follow for this codec entry point.
declare module "@zip.js/zip.js/lib/zip-core-native.js" {
  export { BlobReader, ZipReader } from "@zip.js/zip.js";
  export type { FileEntry } from "@zip.js/zip.js";
}

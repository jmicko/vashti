# Custom model profile images

## Goal

Allow a custom model to have a profile image that appears consistently in the
model picker, custom-model settings, chat messages, and chat history.

Profile images are cosmetic model metadata. They are not chat attachments and
must never be copied into each chat or message.

## Storage model

### Hosted custom models

Store the exact uploaded file as an immutable avatar asset:

* Bytes live in `data/persona-avatars/<owner-user-id>/<asset-id>`.
* SQLite stores the asset ID, owner, filename, MIME type, size, and storage
  path.
* `persona_versions` references the asset ID and stores crop coordinates.
* Messages continue to reference `persona_version_id`; they do not copy image
  data or avatar metadata.

The original upload is the source of truth. Vashti does not convert it to WebP,
resize it, or repeatedly recompress it. Rendering uses `object-fit: cover` and
saved `object-position` coordinates. A future thumbnail pipeline can always
derive new files from the unchanged original.

Accepted formats for the first implementation are JPEG and PNG. Animated
formats and WebP are deliberately excluded until there is a clear product need
and consistent rendering policy.

### Device custom models

Device-only custom models use a dedicated encrypted IndexedDB avatar store.
Each original image is stored once and referenced by private custom-model
versions. It is not duplicated into private messages.

## Version behavior

Changing the underlying image changes the identity of the custom-model
presentation and creates a new model version. This includes:

* adding an image;
* replacing an image;
* removing an image.

Adjusting only the crop position does not create a new version. Crop
coordinates are editable presentation metadata on the existing version. This
is an intentional exception to otherwise immutable custom-model versions:
recropping does not change model behavior, prompt history, or the original
image.

Any behavioral edit, such as changing the name, base model, system prompt, or
tool policy, still creates a new version. The new version carries forward the
current avatar asset and crop coordinates unless the image was explicitly
changed.

## Upload and access API

Hosted assets use dedicated endpoints:

* `POST /api/persona-avatars` uploads one original JPEG or PNG.
* `GET /api/persona-avatars/{asset_id}` serves an authorized asset.
* `DELETE /api/persona-avatars/{asset_id}` removes an unused asset owned by the
  current user.

The upload response contains the asset ID and metadata. Persona create/update
requests carry:

* `avatar_asset_id`
* `avatar_crop_x`
* `avatar_crop_y`

The server validates that an assigned asset belongs to the custom-model owner.
Readers may fetch an asset only when they can access a custom-model version
that references it.

Uploaded-but-unused assets are deleted when the editor is cancelled or a
replacement is discarded. A later maintenance pass may remove abandoned
assets left by interrupted browser sessions.

## Editor

The custom-model editor provides:

* a square preview;
* choose/replace and remove controls;
* horizontal and vertical crop-position controls;
* a reset-crop control.

The browser previews the original with CSS. No canvas export or client-side
recompression is required.

Saving behavior:

1. Upload a newly selected original, if any.
2. Create or update the custom model with the returned asset ID and crop.
3. Delete a superseded asset only when it is no longer referenced.

## Rendering and caching

A shared `ModelAvatar` component renders hosted and device assets. It falls
back to a stable initial when no image exists or loading fails.

Hosted images use normal authenticated HTTP responses with cache validators.
The browser cache is sufficient; Vashti does not maintain a second manual
IndexedDB cache for server-hosted images. Device assets are read from their
existing encrypted local store.

## Non-goals

* Built-in Ollama model images.
* Image editing beyond crop position.
* Generated thumbnails or multiple resolutions.
* A public CDN or unauthenticated avatar URLs.
* Storing binary image data in SQLite.
* Copying avatar data into chats, messages, or attachment rows.

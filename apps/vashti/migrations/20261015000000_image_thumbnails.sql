CREATE TABLE image_thumbnails (
    asset_id TEXT PRIMARY KEY REFERENCES image_assets(id) ON DELETE CASCADE,
    body BLOB NOT NULL
);

-- Inference no longer uses the privileged Kuno control credential.
UPDATE image_service_settings SET token = '';
ALTER TABLE image_service_settings DROP COLUMN token;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub struct SessionStore {
    memory: Arc<Mutex<HashMap<String, String>>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            memory: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn get(&self, connection_id: &str) -> Result<Option<String>, String> {
        let store = self.clone();
        let connection_id = connection_id.to_string();
        tokio::task::spawn_blocking(move || store.get_blocking(&connection_id))
            .await
            .map_err(|error| format!("Secure session task failed: {error}"))?
    }

    pub async fn set(&self, connection_id: &str, cookie: &str) -> Result<(), String> {
        let store = self.clone();
        let connection_id = connection_id.to_string();
        let cookie = cookie.to_string();
        tokio::task::spawn_blocking(move || store.set_blocking(&connection_id, &cookie))
            .await
            .map_err(|error| format!("Secure session task failed: {error}"))?
    }

    pub async fn delete(&self, connection_id: &str) -> Result<(), String> {
        let store = self.clone();
        let connection_id = connection_id.to_string();
        tokio::task::spawn_blocking(move || store.delete_blocking(&connection_id))
            .await
            .map_err(|error| format!("Secure session task failed: {error}"))?
    }

    fn get_blocking(&self, connection_id: &str) -> Result<Option<String>, String> {
        if let Some(value) = self
            .memory
            .lock()
            .map_err(|_| "Session store lock failed".to_string())?
            .get(connection_id)
            .cloned()
        {
            return Ok(Some(value));
        }
        let stored = platform_get(connection_id)?;
        if let Some(value) = stored.as_ref() {
            self.memory
                .lock()
                .map_err(|_| "Session store lock failed".to_string())?
                .insert(connection_id.to_string(), value.clone());
        }
        Ok(stored)
    }

    fn set_blocking(&self, connection_id: &str, cookie: &str) -> Result<(), String> {
        platform_set(connection_id, cookie)?;
        self.memory
            .lock()
            .map_err(|_| "Session store lock failed".to_string())?
            .insert(connection_id.to_string(), cookie.to_string());
        Ok(())
    }

    fn delete_blocking(&self, connection_id: &str) -> Result<(), String> {
        platform_delete(connection_id)?;
        self.memory
            .lock()
            .map_err(|_| "Session store lock failed".to_string())?
            .remove(connection_id);
        Ok(())
    }
}

#[cfg(target_os = "android")]
fn platform_entry(connection_id: &str) -> Result<keyring_core::Entry, String> {
    use keyring_core::api::CredentialStoreApi;

    let store = android_native_keyring_store::Store::new().map_err(|error| error.to_string())?;
    store
        .build("chat.vashti.app.session", connection_id, None)
        .map_err(|error| error.to_string())
}

#[cfg(target_os = "android")]
fn platform_get(connection_id: &str) -> Result<Option<String>, String> {
    match platform_entry(connection_id)?.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring_core::Error::NoEntry) => Ok(None),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(not(target_os = "android"))]
fn platform_get(_connection_id: &str) -> Result<Option<String>, String> {
    Ok(None)
}

#[cfg(target_os = "android")]
fn platform_set(connection_id: &str, cookie: &str) -> Result<(), String> {
    platform_entry(connection_id)?
        .set_password(cookie)
        .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "android"))]
fn platform_set(_connection_id: &str, _cookie: &str) -> Result<(), String> {
    Ok(())
}

#[cfg(target_os = "android")]
fn platform_delete(connection_id: &str) -> Result<(), String> {
    match platform_entry(connection_id)?.delete_credential() {
        Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(not(target_os = "android"))]
fn platform_delete(_connection_id: &str) -> Result<(), String> {
    Ok(())
}

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, oneshot};
use uuid::Uuid;

use crate::error::ApiError;

const COMPLETED_CALL_TTL: Duration = Duration::from_secs(5 * 60);
pub const CLIENT_TOOL_TIMEOUT: Duration = Duration::from_secs(2 * 60);

#[derive(Clone, Default)]
pub struct ClientToolBroker {
    state: Arc<Mutex<BrokerState>>,
}

#[derive(Default)]
struct BrokerState {
    pending: HashMap<String, PendingCall>,
    completed: HashMap<String, CompletedCall>,
}

struct PendingCall {
    generation_id: String,
    user_id: String,
    session_binding: String,
    resume_token: String,
    sender: oneshot::Sender<String>,
}

struct CompletedCall {
    generation_id: String,
    user_id: String,
    session_binding: String,
    resume_token: String,
    completed_at: Instant,
}

pub struct RegisteredClientToolCall {
    pub call_id: String,
    pub resume_token: String,
    pub receiver: oneshot::Receiver<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolveClientToolCall {
    Accepted,
    AlreadyCompleted,
}

impl ClientToolBroker {
    pub async fn register(
        &self,
        generation_id: &str,
        user_id: &str,
        session_binding: &str,
    ) -> RegisteredClientToolCall {
        let call_id = Uuid::new_v4().to_string();
        let resume_token = Uuid::new_v4().to_string();
        let (sender, receiver) = oneshot::channel();
        let mut state = self.state.lock().await;
        state.remove_expired_completed();
        state.pending.insert(
            call_id.clone(),
            PendingCall {
                generation_id: generation_id.to_string(),
                user_id: user_id.to_string(),
                session_binding: session_binding.to_string(),
                resume_token: resume_token.clone(),
                sender,
            },
        );

        RegisteredClientToolCall {
            call_id,
            resume_token,
            receiver,
        }
    }

    pub async fn resolve(
        &self,
        identity: &ClientToolCallIdentity<'_>,
        result: String,
    ) -> Result<ResolveClientToolCall, ApiError> {
        let mut state = self.state.lock().await;
        state.remove_expired_completed();

        if let Some(completed) = state.completed.get(identity.call_id) {
            return if completed.matches(identity) {
                Ok(ResolveClientToolCall::AlreadyCompleted)
            } else {
                Err(invalid_result_error())
            };
        }

        let matches = state
            .pending
            .get(identity.call_id)
            .is_some_and(|pending| pending.matches(identity));
        if !matches {
            return Err(invalid_result_error());
        }

        let pending = state
            .pending
            .remove(identity.call_id)
            .expect("matched pending client tool call");
        state.completed.insert(
            identity.call_id.to_string(),
            CompletedCall {
                generation_id: pending.generation_id.clone(),
                user_id: pending.user_id.clone(),
                session_binding: pending.session_binding.clone(),
                resume_token: pending.resume_token.clone(),
                completed_at: Instant::now(),
            },
        );
        if pending.sender.send(result).is_err() {
            state.completed.remove(identity.call_id);
            return Err(ApiError::conflict(
                "client_tool_call_closed",
                "The private generation is no longer waiting for this tool result",
            ));
        }
        Ok(ResolveClientToolCall::Accepted)
    }

    pub async fn cancel(&self, call_id: &str) {
        self.state.lock().await.pending.remove(call_id);
    }
}

pub struct ClientToolCallIdentity<'a> {
    pub generation_id: &'a str,
    pub call_id: &'a str,
    pub user_id: &'a str,
    pub session_binding: &'a str,
    pub resume_token: &'a str,
}

impl PendingCall {
    fn matches(&self, identity: &ClientToolCallIdentity<'_>) -> bool {
        self.generation_id == identity.generation_id
            && self.user_id == identity.user_id
            && self.session_binding == identity.session_binding
            && self.resume_token == identity.resume_token
    }
}

impl CompletedCall {
    fn matches(&self, identity: &ClientToolCallIdentity<'_>) -> bool {
        self.generation_id == identity.generation_id
            && self.user_id == identity.user_id
            && self.session_binding == identity.session_binding
            && self.resume_token == identity.resume_token
    }
}

impl BrokerState {
    fn remove_expired_completed(&mut self) {
        let now = Instant::now();
        self.completed.retain(|_, completed| {
            now.saturating_duration_since(completed.completed_at) < COMPLETED_CALL_TTL
        });
    }
}

pub fn session_binding(session_id: &str) -> String {
    let digest = Sha256::digest(session_id.as_bytes());
    format!("{digest:x}")
}

fn invalid_result_error() -> ApiError {
    ApiError::forbidden(
        "invalid_client_tool_result",
        "This client tool result does not match an active private generation",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity<'a>(
        registered: &'a RegisteredClientToolCall,
        generation_id: &'a str,
        user_id: &'a str,
        session: &'a str,
    ) -> ClientToolCallIdentity<'a> {
        ClientToolCallIdentity {
            generation_id,
            call_id: &registered.call_id,
            user_id,
            session_binding: session,
            resume_token: &registered.resume_token,
        }
    }

    #[tokio::test]
    async fn resolves_only_the_matching_user_generation_and_session() {
        let broker = ClientToolBroker::default();
        let registered = broker.register("generation", "user", "session").await;

        let wrong = ClientToolCallIdentity {
            user_id: "other-user",
            ..identity(&registered, "generation", "user", "session")
        };
        assert_eq!(
            broker
                .resolve(&wrong, "ignored".to_string())
                .await
                .unwrap_err()
                .code(),
            "invalid_client_tool_result"
        );

        let matched = identity(&registered, "generation", "user", "session");
        assert_eq!(
            broker
                .resolve(&matched, "result".to_string())
                .await
                .unwrap(),
            ResolveClientToolCall::Accepted
        );
        assert_eq!(registered.receiver.await.unwrap(), "result");
    }

    #[tokio::test]
    async fn duplicate_matching_results_are_idempotent() {
        let broker = ClientToolBroker::default();
        let registered = broker.register("generation", "user", "session").await;
        let matched = identity(&registered, "generation", "user", "session");

        assert_eq!(
            broker
                .resolve(&matched, "result".to_string())
                .await
                .unwrap(),
            ResolveClientToolCall::Accepted
        );
        assert_eq!(
            broker
                .resolve(&matched, "result".to_string())
                .await
                .unwrap(),
            ResolveClientToolCall::AlreadyCompleted
        );
    }

    #[tokio::test]
    async fn completed_call_rejects_a_different_identity() {
        let broker = ClientToolBroker::default();
        let registered = broker.register("generation", "user", "session").await;
        let matched = identity(&registered, "generation", "user", "session");

        assert_eq!(
            broker
                .resolve(&matched, "result".to_string())
                .await
                .unwrap(),
            ResolveClientToolCall::Accepted
        );

        let wrong_generation = identity(&registered, "other-generation", "user", "session");
        assert_eq!(
            broker
                .resolve(&wrong_generation, "ignored".to_string())
                .await
                .unwrap_err()
                .code(),
            "invalid_client_tool_result"
        );
    }
}

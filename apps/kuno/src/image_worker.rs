//! Single-threaded, private model-retaining worker entrypoint support.
//! No Tokio, files, request diagnostics, or content-bearing process arguments.
use kuno_image::worker_protocol::{self as protocol, Request, TransientBytes};
use kuno_protocol::ImageRecipe;
use std::io::{Read, Write};

/// `generate` must clear all native request-derived state before returning.
/// It may keep immutable model weights; its previews use the same stdout pipe.
pub fn serve(
    input: &mut impl Read,
    output: &mut impl Write,
    mut generate: impl FnMut(&ImageRecipe) -> Result<Vec<u8>, &'static str>,
) -> Result<(), &'static str> {
    protocol::write_response(output, protocol::READY, &[])?;
    loop {
        let request = match protocol::read_request(input) {
            Ok(request) => request,
            Err(code) => return fail(output, code),
        };
        let Some(Request::Generate(recipe)) = request else {
            return Ok(());
        };
        let png = match generate(&recipe) {
            Ok(png) => TransientBytes(png),
            Err(code) => {
                drop(recipe);
                return fail(output, code);
            }
        };
        if let Err(code) = kuno_image::validate_output(&png.0, recipe.width, recipe.height) {
            drop(png);
            drop(recipe);
            return fail(output, code);
        }
        protocol::write_response(output, protocol::FINAL, &png.0)?;
        // Do not acknowledge CLEAN until every worker-side Rust content owner
        // is cleared, in addition to the generator's native cleanup contract.
        drop(png);
        drop(recipe);
        protocol::write_response(output, protocol::CLEAN, &[])?;
    }
}

fn fail(output: &mut impl Write, code: &'static str) -> Result<(), &'static str> {
    let code = match code {
        "image_request_read_failed"
        | "image_worker_request_invalid"
        | "image_request_invalid"
        | "image_request_too_large"
        | "image_physical_gpu_not_found"
        | "image_model_load_failed"
        | "image_gpu_memory_unavailable"
        | "image_cancelled"
        | "image_vulkan_device_lost"
        | "image_gpu_int8_unsupported"
        | "image_attention_unsupported"
        | "image_native_runtime_failed"
        | "image_native_output_invalid"
        | "image_worker_output_invalid"
        | "image_encoding_failed"
        | "image_generation_failed"
        | "image_engine_unavailable" => code,
        _ => "image_generation_failed",
    };
    let _ = protocol::write_response(output, protocol::ERROR, code.as_bytes());
    Err(code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};
    fn recipe() -> ImageRecipe {
        ImageRecipe {
            workflow: kuno_protocol::QWEN21_RECIPE.into(),
            prompt: "transient synthetic prompt".into(),
            negative_prompt: String::new(),
            seed: "1".into(),
            width: 64,
            height: 64,
            steps: 1,
            cfg: 1.0,
            attention: Default::default(),
            flow_shift: None,
            reference_images: vec![],
        }
    }
    fn kinds(bytes: &[u8]) -> Vec<u8> {
        let mut at = 0;
        let mut kinds = Vec::new();
        while at < bytes.len() {
            assert_eq!(&bytes[at..at + 4], protocol::RESPONSE_MAGIC);
            kinds.push(bytes[at + 4]);
            let size = u32::from_be_bytes(bytes[at + 5..at + 9].try_into().unwrap()) as usize;
            at += 9 + size;
        }
        kinds
    }
    #[test]
    fn consecutive_jobs_keep_session_but_clear_request_before_clean() {
        let mut input = Vec::new();
        protocol::write_generate(&mut input, &recipe()).unwrap();
        protocol::write_generate(&mut input, &recipe()).unwrap();
        protocol::write_shutdown(&mut input).unwrap();
        let clears = Rc::new(Cell::new(0));
        let jobs = Rc::new(Cell::new(0));
        struct Writer {
            bytes: Vec<u8>,
            clears: Rc<Cell<usize>>,
            clean_frames: usize,
        }
        impl Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes == [protocol::CLEAN] {
                    assert!(self.clears.get() > self.clean_frames);
                    self.clean_frames += 1;
                }
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut output = Writer {
            bytes: Vec::new(),
            clears: Rc::clone(&clears),
            clean_frames: 0,
        };
        serve(&mut input.as_slice(), &mut output, |recipe| {
            struct NativeRequest(Rc<Cell<usize>>);
            impl Drop for NativeRequest {
                fn drop(&mut self) {
                    self.0.set(self.0.get() + 1);
                }
            }
            let _derived_state = NativeRequest(Rc::clone(&clears));
            jobs.set(jobs.get() + 1);
            assert_eq!(recipe.prompt, "transient synthetic prompt");
            Ok(include_bytes!("../../vashti/src/images/fixtures/64.png").to_vec())
        })
        .unwrap();
        assert_eq!(jobs.get(), 2);
        assert_eq!(
            kinds(&output.bytes),
            [
                protocol::READY,
                protocol::FINAL,
                protocol::CLEAN,
                protocol::FINAL,
                protocol::CLEAN
            ]
        );
    }
    #[test]
    fn failed_job_never_acknowledges_retention_or_runs_next_request() {
        let mut input = Vec::new();
        protocol::write_generate(&mut input, &recipe()).unwrap();
        protocol::write_generate(&mut input, &recipe()).unwrap();
        let mut output = Vec::new();
        let mut calls = 0;
        let result = serve(&mut input.as_slice(), &mut output, |_| {
            calls += 1;
            Err("unclassified private backend details")
        });
        assert_eq!(result, Err("image_generation_failed"));
        assert_eq!(calls, 1);
        assert_eq!(kinds(&output), [protocol::READY, protocol::ERROR]);
        assert!(!output.windows(7).any(|b| b == b"private"));
    }
    #[test]
    fn eof_and_shutdown_release_worker_without_cleanup_for_unfinished_job() {
        let mut output = Vec::new();
        serve(&mut &[][..], &mut output, |_| panic!("unexpected job")).unwrap();
        assert_eq!(kinds(&output), [protocol::READY]);
        let mut input = Vec::new();
        protocol::write_shutdown(&mut input).unwrap();
        input.extend_from_slice(b"not consumed after shutdown");
        output.clear();
        serve(&mut input.as_slice(), &mut output, |_| {
            panic!("unexpected job")
        })
        .unwrap();
        assert_eq!(kinds(&output), [protocol::READY]);
    }
    #[test]
    fn malformed_result_cannot_become_reusable_session() {
        let mut input = Vec::new();
        protocol::write_generate(&mut input, &recipe()).unwrap();
        let mut output = Vec::new();
        assert_eq!(
            serve(&mut input.as_slice(), &mut output, |_| Ok(
                b"not png".to_vec()
            )),
            Err("image_worker_output_invalid")
        );
        assert_eq!(kinds(&output), [protocol::READY, protocol::ERROR]);
    }
}

// CPU source regressions only; no client construction, device or runtime work.
fn section<'a>(source: &'a str, begin: &str, end: &str) -> &'a str {
    source
        .split_once(begin)
        .unwrap()
        .1
        .split_once(end)
        .unwrap()
        .0
}

#[test]
fn same_eight_full_gdn_owners_keep_every_original_guard() {
    let body = section(
        include_str!("gdn_chunk.rs"),
        "pub fn validate_bindings(",
        "\npub fn enqueue(",
    );
    assert!(!body.contains(".get_resource("));
    for guard in [
        "buffers.0[..index].contains(&id)",
        "arena.layout(id).map_err(|_| fault())?.logical_words != n",
        "h.offset_start.unwrap_or(0) != 0",
        "h.offset_end.unwrap_or(0) != 0",
        "h.stream != arena.handle(buffers.0[0]).map_err(|_| fault())?.stream",
        "region.offset != 0 || region.size != h.size() || region.buffer.size() != h.size()",
        "resources[..index]",
        "old.resource().buffer == region.buffer",
    ] {
        assert!(
            body.contains(guard),
            "missing original owner guard: {guard}"
        );
    }
    let capacity = body.find("handles.capacity() != 8").unwrap();
    let populate = body.find("handles.push(arena.handle(id)").unwrap();
    let lookup = body
        .find("crate::resident_util::resources(client, handles)")
        .unwrap();
    let shape = body
        .find("resources.len() != 8 || resources.capacity() != 8")
        .unwrap();
    let physical = body
        .find("for (index, r) in resources.iter().enumerate()")
        .unwrap();
    assert!(capacity < populate && populate < lookup && lookup < shape && shape < physical);
    assert!(body.contains("HEADER_BYTES <= 4096"));
    assert!(body.contains("37 * 2 * 512 + HEADER_BYTES <= 64 * 2 * 512"));
}

#[test]
fn selected_metadata_and_default_order_share_the_existing_authority() {
    let helper = section(
        include_str!("resident_util.rs"),
        "pub fn resources(",
        "\nstruct ScrubBytes",
    );
    let (selected, default) = helper.split_once("} else {").unwrap();
    assert!(selected.contains("RETAINED_METADATA.load(Ordering::Acquire)"));
    assert!(selected.contains(".retained_resource_metadata(handles)"));
    assert!(default.contains("handles\n            .into_iter()"));
    assert!(default.contains(".map(|h| client.get_resource(h).map_err(fault))"));
    assert!(
        include_str!("resident_util.rs")
            .contains("static RETAINED_METADATA: AtomicBool = AtomicBool::new(false)")
    );

    let server = include_str!("../vendor/cubecl-wgpu/src/compute/server.rs");
    let default = section(
        server,
        "    fn get_resource(\n",
        "    fn retained_resource_metadata(\n",
    );
    assert!(default.contains("self.scheduler.execute_streams(streams)"));
    let selected = section(
        server,
        "    fn retained_resource_metadata(\n",
        "    fn clear_private_owners(\n",
    );
    assert!(selected.contains("retained_metadata_bindings(&bindings, stream_id)?"));
    assert!(selected.contains("stream.private_io_active()?"));
    assert!(selected.contains("stream.mem_manage.get_resource(binding)?"));
    assert!(!selected.contains("self.scheduler.execute_streams("));
    assert!(!selected.contains(".queue.submit("));
    assert!(!selected.contains(".sync("));
}

#[test]
fn all_actual_gdn_recurrences_still_validate_before_launch() {
    let source = include_str!("gdn_chunk.rs");
    for (begin, end, launch) in [
        (
            "pub(crate) fn enqueue_with_policy(",
            "pub(crate) fn enqueue_shared_count1(",
            "$kernel::launch::<WgpuRuntime>",
        ),
        (
            "pub(crate) fn enqueue_shared_count1(",
            "#[cube",
            "crate::next_shared::gdn(",
        ),
    ] {
        let body = section(source, begin, end);
        let validate = body
            .find("validate_bindings(arena, buffers, spans.words())?")
            .unwrap();
        let launch = body.find(launch).unwrap();
        assert!(validate < launch);
    }
    let native = include_str!("resident_gdn_batch.rs");
    assert!(native.contains("gdn_chunk::enqueue_with_policy("));
    assert!(native.contains("gdn_chunk::enqueue_shared_count1("));
    assert!(native.contains("INFO_BYTES_PER_STAGE: u64 = 64 * 2 * 512"));
}

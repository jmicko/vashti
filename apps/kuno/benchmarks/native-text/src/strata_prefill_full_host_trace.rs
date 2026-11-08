// Fixed-public host attribution only. No client calls, queries, files or fences.
pub(super) fn full_host_selected(
    enabled: bool,
    authorized: bool,
    host_only: bool,
    request: u64,
    start: usize,
    count: usize,
    layer: usize,
) -> bool {
    enabled && authorized && host_only && request == 3 && layer < 48
        && matches!((start,count),(0,4096)|(4096,148))
}
type Triples = BTreeMap<&'static str,(u64,u64,u64)>;
fn delta(before: &Triples, after: &Triples) -> Triples {
    after.iter().filter_map(|(&name,&a)| {
        let b=before.get(name).copied().unwrap_or_default();
        let d=(a.0.saturating_sub(b.0),a.1.saturating_sub(b.1),a.2.saturating_sub(b.2));
        (d!=(0,0,0)).then_some((name,d))
    }).collect()
}
pub(crate) struct FullHostLayer(Option<(Triples,Instant,usize,usize,usize)>);
impl FullHostLayer {
    pub(crate) fn new(start:usize,count:usize,layer:usize)->Self {
        // Private/old modes return before reading any diagnostic environment or
        // retaining geometry, clocks, counters or maps.
        if !ENABLED.load(Ordering::Acquire)||!FULL_HOST_TRACE.load(Ordering::Acquire){return Self(None);}
        if !full_host_selected(true,
            std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref()==Ok("1"),
            std::env::var("KUNO_STRATA_NATIVE_PROFILE").as_deref()==Ok("prefill-host-layer"),
            REQUEST.load(Ordering::Relaxed),start,count,layer){return Self(None);}
        Self(Some((TOTALS.with(|t|t.borrow().clone()),Instant::now(),start,count,layer)))
    }
    pub(crate) fn finish(mut self) {
        if let Some((before,clock,start,count,layer))=self.0.take(){
            let ns=clock.elapsed().as_nanos().min(u64::MAX as u128)as u64;
            let changes=TOTALS.with(|t|delta(&before,&t.borrow()));
            eprintln!("public_prefill_full_host_layer {}",serde_json::json!({
                "compiled_public_fixture_only":true,"content_saved":false,
                "request_index":3,"start":start,"count":count,"layer":layer,
                "layer_wall_ns":ns,"original_counter_triples":changes,
                "original_counter_field_semantics_unchanged":true,
                "native_timestamp_sampling":false,"normal_native_rolling_policy":true,
                "existing_coarse_profile_fences_retained":true,"additional_gpu_fences":false,
                "not_a_clean_timing_reference":true
            }));
        }
    }
}
#[cfg(test)]mod tests {
    use super::*;
    #[test]fn only_exact_fixed_public_host_scope_accepts_all96_real_layer_positions(){
        let mut selected=0;
        for enabled in [false,true]{for authorized in [false,true]{for host in [false,true]{
          for request in 0..=4{for(start,count)in [(0,4096),(4096,148),(0,148),(4096,149),(0,4095),(1,4096)]{
            for layer in 0..=48{
                let got=full_host_selected(enabled,authorized,host,request,start,count,layer);
                let expected=enabled&&authorized&&host&&request==3&&layer<48&&[(0,4096),(4096,148)].contains(&(start,count));
                assert_eq!(got,expected);selected+=usize::from(got);
            }
          }}
        }}}assert_eq!(selected,96);
    }
    #[test]fn delta_retains_original_static_triples_and_never_fabricates_clock_or_counter_values(){
        let before=BTreeMap::from([("experts",(1,100,40)),("counter",(9,0,0))]);
        let after=BTreeMap::from([("experts",(2,160,70)),("counter",(12,0,0)),("new",(3,0,0))]);
        assert_eq!(delta(&before,&after),BTreeMap::from([("experts",(1,60,30)),("counter",(3,0,0)),("new",(3,0,0))]));
        assert!(delta(&before,&before).is_empty());
    }
    #[test]fn private_and_old_modes_keep_trace_maps_geometry_and_clocks_absent(){
        assert!(!FULL_HOST_TRACE.load(Ordering::Acquire));
        let before=TOTALS.with(|t|t.borrow().clone());
        assert!(FullHostLayer::new(0,4096,24).0.is_none());
        FullHostLayer::new(4096,148,47).finish();
        assert_eq!(TOTALS.with(|t|t.borrow().clone()),before);
    }
}

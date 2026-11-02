// Additive contiguous raw GU and pending ordered-DOWN cohort publication.
// All original group guards, K16 products, source owners and launch ABI remain.
use super::*;
#[path="strata_matrix_large_wide_products.rs"] mod wide_products;
#[path="strata_matrix_large_tail_products.rs"] mod tail_products;
pub(crate) fn part_rows(words:&[u32;48],tile:usize)->u32 {
    (0..words[9] as usize).map(|i| {
        let n=words[17+i*4];let tail=n%128;
        match tile {128=>n/128,8 if (1..=8).contains(&tail)=>1,
            16 if (9..32).contains(&tail)=>tail.div_ceil(16),
            32 if tail>=32=>tail.div_ceil(32),_=>0}
    }).fold(0,|a,b|if words[13]==1 {a+b}else{a.max(b)})
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_large_cohort(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    h0: u32,
    h1: u32,
    h2: u32,
    h3: u32,
    h4: u32,
    h5: u32,
    h6: u32,
    h7: u32,
    h8: u32,
    h9: u32,
    h10: u32,
    h11: u32,
    h12: u32,
    h13: u32,
    h14: u32,
    h15: u32,
    m0: u32,
    m1: u32,
    m2: u32,
    m3: u32,
    m4: u32,
    m5: u32,
    m6: u32,
    m7: u32,
    m8: u32,
    m9: u32,
    m10: u32,
    m11: u32,
    m12: u32,
    m13: u32,
    m14: u32,
    m15: u32,
    m16: u32,
    m17: u32,
    m18: u32,
    m19: u32,
    m20: u32,
    m21: u32,
    m22: u32,
    m23: u32,
    m24: u32,
    m25: u32,
    m26: u32,
    m27: u32,
    m28: u32,
    m29: u32,
    m30: u32,
    m31: u32,
    #[comptime] down: bool,
    #[comptime] tile: usize,
) {
    let mut slot = usize::cast_from(CUBE_POS_Z);
    let mut tile_index=usize::cast_from(CUBE_POS_Y);
    if h13==1 && (CUBE_POS_Z!=0 || h9<1 || h9>8) {
        if UNIT_POS==0 {fault[0].fetch_or(128);}terminate!();
    }
    if h13==1 {
        let mut remaining=tile_index;slot=8;
        if h9>0 {
            let n=m1;let tail=n%128;
            let mut tiles=0u32;
            if comptime!(tile==128) {tiles=n/128;}
            if comptime!(tile==8) {if tail>0 && tail<=8 {tiles=1;}}
            if comptime!(tile==16) {if tail>8 && tail<32 {tiles=(tail+15)/16;}}
            if comptime!(tile==32) {if tail>=32 {tiles=(tail+31)/32;}}
            if slot==8 {if remaining<usize::cast_from(tiles) {slot=0;tile_index=remaining;}else{remaining-=usize::cast_from(tiles);}}
        }
        if h9>1 {
            let n=m5;let tail=n%128;
            let mut tiles=0u32;
            if comptime!(tile==128) {tiles=n/128;}
            if comptime!(tile==8) {if tail>0 && tail<=8 {tiles=1;}}
            if comptime!(tile==16) {if tail>8 && tail<32 {tiles=(tail+15)/16;}}
            if comptime!(tile==32) {if tail>=32 {tiles=(tail+31)/32;}}
            if slot==8 {if remaining<usize::cast_from(tiles) {slot=1;tile_index=remaining;}else{remaining-=usize::cast_from(tiles);}}
        }
        if h9>2 {
            let n=m9;let tail=n%128;
            let mut tiles=0u32;
            if comptime!(tile==128) {tiles=n/128;}
            if comptime!(tile==8) {if tail>0 && tail<=8 {tiles=1;}}
            if comptime!(tile==16) {if tail>8 && tail<32 {tiles=(tail+15)/16;}}
            if comptime!(tile==32) {if tail>=32 {tiles=(tail+31)/32;}}
            if slot==8 {if remaining<usize::cast_from(tiles) {slot=2;tile_index=remaining;}else{remaining-=usize::cast_from(tiles);}}
        }
        if h9>3 {
            let n=m13;let tail=n%128;
            let mut tiles=0u32;
            if comptime!(tile==128) {tiles=n/128;}
            if comptime!(tile==8) {if tail>0 && tail<=8 {tiles=1;}}
            if comptime!(tile==16) {if tail>8 && tail<32 {tiles=(tail+15)/16;}}
            if comptime!(tile==32) {if tail>=32 {tiles=(tail+31)/32;}}
            if slot==8 {if remaining<usize::cast_from(tiles) {slot=3;tile_index=remaining;}else{remaining-=usize::cast_from(tiles);}}
        }
        if h9>4 {
            let n=m17;let tail=n%128;
            let mut tiles=0u32;
            if comptime!(tile==128) {tiles=n/128;}
            if comptime!(tile==8) {if tail>0 && tail<=8 {tiles=1;}}
            if comptime!(tile==16) {if tail>8 && tail<32 {tiles=(tail+15)/16;}}
            if comptime!(tile==32) {if tail>=32 {tiles=(tail+31)/32;}}
            if slot==8 {if remaining<usize::cast_from(tiles) {slot=4;tile_index=remaining;}else{remaining-=usize::cast_from(tiles);}}
        }
        if h9>5 {
            let n=m21;let tail=n%128;
            let mut tiles=0u32;
            if comptime!(tile==128) {tiles=n/128;}
            if comptime!(tile==8) {if tail>0 && tail<=8 {tiles=1;}}
            if comptime!(tile==16) {if tail>8 && tail<32 {tiles=(tail+15)/16;}}
            if comptime!(tile==32) {if tail>=32 {tiles=(tail+31)/32;}}
            if slot==8 {if remaining<usize::cast_from(tiles) {slot=5;tile_index=remaining;}else{remaining-=usize::cast_from(tiles);}}
        }
        if h9>6 {
            let n=m25;let tail=n%128;
            let mut tiles=0u32;
            if comptime!(tile==128) {tiles=n/128;}
            if comptime!(tile==8) {if tail>0 && tail<=8 {tiles=1;}}
            if comptime!(tile==16) {if tail>8 && tail<32 {tiles=(tail+15)/16;}}
            if comptime!(tile==32) {if tail>=32 {tiles=(tail+31)/32;}}
            if slot==8 {if remaining<usize::cast_from(tiles) {slot=6;tile_index=remaining;}else{remaining-=usize::cast_from(tiles);}}
        }
        if h9>7 {
            let n=m29;let tail=n%128;
            let mut tiles=0u32;
            if comptime!(tile==128) {tiles=n/128;}
            if comptime!(tile==8) {if tail>0 && tail<=8 {tiles=1;}}
            if comptime!(tile==16) {if tail>8 && tail<32 {tiles=(tail+15)/16;}}
            if comptime!(tile==32) {if tail>=32 {tiles=(tail+31)/32;}}
            if slot==8 {if remaining<usize::cast_from(tiles) {slot=7;tile_index=remaining;}else{remaining-=usize::cast_from(tiles);}}
        }
    }
    if h13==1 && slot==8 {
        if UNIT_POS==0 {fault[0].fetch_or(128);}terminate!();
    }
    let mut group = 0u32;
    let mut count = 0u32;
    let mut prefix = 0u32;
    let mut raw = 0u32;
    if slot == 0 {
        group = m0;
        count = m1;
        prefix = m2;
        raw = m3;
    }
    if slot == 1 {
        group = m4;
        count = m5;
        prefix = m6;
        raw = m7;
    }
    if slot == 2 {
        group = m8;
        count = m9;
        prefix = m10;
        raw = m11;
    }
    if slot == 3 {
        group = m12;
        count = m13;
        prefix = m14;
        raw = m15;
    }
    if slot == 4 {
        group = m16;
        count = m17;
        prefix = m18;
        raw = m19;
    }
    if slot == 5 {
        group = m20;
        count = m21;
        prefix = m22;
        raw = m23;
    }
    if slot == 6 {
        group = m24;
        count = m25;
        prefix = m26;
        raw = m27;
    }
    if slot == 7 {
        group = m28;
        count = m29;
        prefix = m30;
        raw = m31;
    }

    let columns = comptime!(if down { 640usize } else { 2560usize });
    let rows = comptime!(if down { 2560usize } else { 1280usize });
    let members = usize::cast_from(count);
    let group = usize::cast_from(group);
    let prefix = usize::cast_from(prefix);
    if CUBE_DIM_X != comptime!(if tile==128 {256u32}else{64u32}) || CUBE_DIM_Y != 1u32 || CUBE_DIM_Z != 1u32 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    // Distinct uniform guards precede every untrusted map index and padded
    // arithmetic; never rely on eager boolean lowering to suppress a read.
    if h9 < 1
        || h9 > 8
        || slot >= usize::cast_from(h9)
        || h4 > 512 - h9
        || h11 != comptime!(tile as u32)
        || h0 < 1
        || h0 > comptime!(crate::strata_prompt::groups::CAP_U32)
        || h1 != comptime!(if down { 640u32 } else { 2560u32 })
        || h2 != comptime!(if down { 2560u32 } else { 1280u32 })
        || h5 != comptime!(if down { 2u32 } else { 0u32 })
        || group >= 512
        || group != usize::cast_from(h4) + slot
        || members < 256
        || members > comptime!(crate::strata_prompt::groups::CAPACITY)
        || members > usize::cast_from(h0)
        || prefix > comptime!(crate::strata_prompt::groups::CAPACITY)
        || prefix % 32 != 0
        || raw > 8
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    let padded = (members + 31) / 32 * 32;
    if padded > comptime!(crate::strata_prompt::groups::CAPACITY) - prefix || map.len() != 65536 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if map[0] != h0
        || map[1] != 0
        || map[2] != h0
        || map[3] < 1
        || map[3] > 512
        || group >= usize::cast_from(map[3])
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if map[4 + group] >= 512
        || map[516 + group] != count
        || usize::cast_from(map[1028 + group]) > usize::cast_from(h0) * 10
        || members > usize::cast_from(h0) * 10 - usize::cast_from(map[1028 + group])
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    let base = members / 128 * 128;
    let tail = members-base;
    let piece = if comptime!(tile==128) {base}else{tail};
    if comptime!(tile==8) {if tail==0 || tail>8 {terminate!();}}
    if comptime!(tile==16) {if tail<=8 || tail>=32 {terminate!();}}
    if comptime!(tile==32) {if tail<32 {terminate!();}}
    if tile_index*comptime!(tile)>=piece {terminate!();}
    let mut input_prefix=prefix;
    if h13>comptime!(if cfg!(feature="strata-layer-gather") {1u32}else{0u32}) || h14!=0 || h15!=0 {
        if UNIT_POS==0 {fault[0].fetch_or(128);}terminate!();
    }
    // Only GU reads global X. Keep each terminating guard at the outer
    // kernel scope: nested early exits otherwise form invalid SPIR-V CFG.
    if comptime!(!down) {
        let global=usize::cast_from(map[comptime!(crate::strata_prompt::groups::LAYER_PREFIX)+group]);
        let total=usize::cast_from(map[comptime!(crate::strata_prompt::groups::LAYER_PREFIX+512)]);
        let extent=(members+31)/32*32;
        if h13==1 && (total>comptime!(crate::strata_prompt::groups::PADDED_ROWS) || global%32!=0 || global>total) {
            if UNIT_POS==0 {fault[0].fetch_or(128);}terminate!();
        }
        if h13==1 && extent>total-global {
            if UNIT_POS==0 {fault[0].fetch_or(128);}terminate!();
        }
        let mut next=total;
        if group+1<usize::cast_from(map[3]) {next=usize::cast_from(map[comptime!(crate::strata_prompt::groups::LAYER_PREFIX)+group+1]);}
        if h13==1 && ((group==0 && global!=0) || global+extent!=next || input.len()!=comptime!(crate::strata_prompt::groups::PADDED_ROWS*640)) {
            if UNIT_POS==0 {fault[0].fetch_or(128);}terminate!();
        }
        if h13==1 {input_prefix=global;}
    }
    let weight_first = slot*4915200 + comptime!(if down {3276800usize}else{0usize});
    if comptime!(tile==128) {
        wide_products::wide(input,weight,output,map,status,fault,
            u32::cast_from(columns),u32::cast_from(rows),count,u32::cast_from(weight_first),
            u32::cast_from(group),u32::cast_from(prefix),u32::cast_from(input_prefix),u32::cast_from(tile_index),u32::cast_from(base),down);
    } else if comptime!(tile==32) {
        tail_products::macro_product::<f16>(input,weight,columns,rows,tail,weight_first,
            output,map,status,fault,group,base,members,prefix,input_prefix,tile_index,down);
    } else {
        tail_products::narrow_product::<f16>(input,weight,columns,rows,tail,weight_first,
            tile,false,output,map,status,fault,group,base,members,prefix,input_prefix,tile_index,down);
    }
}

pub(crate) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    [
        ("large_cohort128_gu",false,128),("large_cohort128_down",true,128),
        ("large_cohort8_gu",false,8),("large_cohort8_down",true,8),
        ("large_cohort16_gu",false,16),("large_cohort16_down",true,16),
        ("large_cohort32_gu",false,32),("large_cohort32_down",true,32),
    ].map(|(name,down,tile)|(name,definition(down,tile))).into_iter().collect()
}

fn definition(down: bool, tile: usize) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    AddressType::U32.register(&mut b.scope);
    let t = Vector::<f16, Const<4>>::as_type(&b.scope);
    let input = b.input_array(t);
    let weight = b.input_array(t);
    let output = b.output_array(f32::as_type(&b.scope));
    let map = b.input_array(u32::as_type(&b.scope));
    let status = b.output_array(u32::as_type(&b.scope));
    let fault =
        b.output_array(cubecl::ir::StorageType::Atomic(ElemType::UInt(UIntKind::U32)).into());
    let scalars: [cubecl::ir::ManagedVariable; 48] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    expert_large_cohort::expand(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        map.into(),
        status.into(),
        fault.into(),
        scalars[0].clone().into(),
        scalars[1].clone().into(),
        scalars[2].clone().into(),
        scalars[3].clone().into(),
        scalars[4].clone().into(),
        scalars[5].clone().into(),
        scalars[6].clone().into(),
        scalars[7].clone().into(),
        scalars[8].clone().into(),
        scalars[9].clone().into(),
        scalars[10].clone().into(),
        scalars[11].clone().into(),
        scalars[12].clone().into(),
        scalars[13].clone().into(),
        scalars[14].clone().into(),
        scalars[15].clone().into(),
        scalars[16].clone().into(),
        scalars[17].clone().into(),
        scalars[18].clone().into(),
        scalars[19].clone().into(),
        scalars[20].clone().into(),
        scalars[21].clone().into(),
        scalars[22].clone().into(),
        scalars[23].clone().into(),
        scalars[24].clone().into(),
        scalars[25].clone().into(),
        scalars[26].clone().into(),
        scalars[27].clone().into(),
        scalars[28].clone().into(),
        scalars[29].clone().into(),
        scalars[30].clone().into(),
        scalars[31].clone().into(),
        scalars[32].clone().into(),
        scalars[33].clone().into(),
        scalars[34].clone().into(),
        scalars[35].clone().into(),
        scalars[36].clone().into(),
        scalars[37].clone().into(),
        scalars[38].clone().into(),
        scalars[39].clone().into(),
        scalars[40].clone().into(),
        scalars[41].clone().into(),
        scalars[42].clone().into(),
        scalars[43].clone().into(),
        scalars[44].clone().into(),
        scalars[45].clone().into(),
        scalars[46].clone().into(),
        scalars[47].clone().into(),
        down,
        tile,
    );
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(if tile==128 {256}else{64})))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    words: &[u32; 48],
    down: bool,
    tile: usize,
) -> Result<()> {
    crate::strata_matrix::native::compatible(client, false)?;
    let padded = crate::strata_prompt::experts::large::validate_words(words)?;
    let [input, weight, output, map, status, fault] = h;
    if ![8,16,32,128].contains(&tile)
        || words[11] != tile as u32
        || words[1] != if down { 640 } else { 2560 }
        || words[2] != if down { 2560 } else { 1280 }
        || words[5] != if down { 2 } else { 0 }
        || words[3] != 0
        || input.size_in_used() < (padded * words[1] as usize * 2) as u64
        || weight.size_in_used()
            < u64::from(words[9]) * crate::strata_prompt::experts::cohorts::IMAGE_BYTES
        || output.size_in_used()
            != if down {
                crate::strata_prompt::groups::PART_BYTES
            } else {
                crate::strata_prompt::groups::GU_BYTES
            }
        || map.size_in_used() != (crate::strata_prompt::groups::MAP_WORDS * 4) as u64
        || status.size_in_used() != (crate::strata_prompt::groups::STATUS_WORDS * 4) as u64
        || fault.size_in_used() != 4
        || [&input, &weight, &output, &map, &status, &fault]
            .iter()
            .any(|v| v.stream != input.stream)
    {
        return Err(Error::InvalidBufferLength);
    }

    for v in [&input, &weight] {
        if v.size_in_used() % 8 != 0 || v.offset_start.unwrap_or(0) % 8 != 0 {
            return Err(Error::InvalidBufferLength);
        }
    }
    let ia = unsafe { ArrayArg::from_raw_parts(input.clone(), input.size_in_used() as usize / 2) };
    let wa =
        unsafe { ArrayArg::from_raw_parts(weight.clone(), weight.size_in_used() as usize / 2) };
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let ma = crate::resident_util::raw_arg(map.clone(), map.size_in_used() as usize / 4)?;
    let sa = crate::resident_util::raw_arg(status.clone(), status.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    let hw = &client.properties().hardware;
    if hw.max_units_per_cube < 256 || hw.max_cube_dim.0 < 256 || hw.max_shared_memory_size < 26624 {
        return Err(Error::LimitExceeded("large cohort original wide/tail geometry"));
    }
    let part_rows=part_rows(words,tile);
    if part_rows==0 {return Err(Error::InvalidBufferLength);}
    let settings = matrix_settings().cube_dim(CubeDim::new_1d(if tile==128 {256}else{64}));
    let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
    launcher.with_scope(|scope| scope.device_properties(client.properties()));
    let [ia, wa, oa, ma, sa, fa] = register_arrays(&mut launcher, [ia, wa, oa, ma, sa, fa]);
    register_words(&mut launcher, words);
    // The original generated kernel's runtime-u32 compilation argument is ().
    // Construct that SAME kernel with the same authentic array compilation
    // arguments and settings; only scalar serialization is performed in bulk.
    let kernel = expert_large_cohort::ExpertLargeCohort::<WgpuRuntime>::new(
        settings,
        client.clone(),
        ia,
        wa,
        oa,
        ma,
        sa,
        fa,
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        down,
        tile,
    );
    unsafe {
        launcher.launch_unchecked(
            CubeCount::Static(
                (if down { 2560usize } else { 1280usize } / if tile==128 {128}else{64}) as u32,
                part_rows,
                if words[13]==1 {1}else{words[9]},
            ),
            kernel,
            client,
        );
    }
    crate::strata_prefill_profile::count("expert_large_cohort_matrix_launches",1);
    crate::strata_prefill_profile::count(if tile==128 {"expert_large_cohort_prefix_launches"}else{"expert_large_cohort_tail_launches"},1);
    crate::strata_prefill_profile::count("expert_large_cohort_matrix_workgroups",
        u64::from(words[2])/if tile==128 {128}else{64}*u64::from(part_rows)*u64::from(if words[13]==1 {1}else{words[9]}));
    Ok(())
}

#[path="strata_matrix_large_cohort_validation.rs"]
mod validation;
pub(crate) use validation::validate_sources;

#[cfg(all(test,feature="strata-layer-gather"))]
mod layer_tests {
 use super::*;
 #[test] fn layer_gather_flat_tiles_cover_only_real_prefix_and_tail_pairs() {
  for counts in [vec![256usize,257,264,265,287,288,383,511],vec![512;8],vec![4095],vec![4096]] {
   let mut words=[0u32;48];words[9]=counts.len() as u32;
   for(i,&n)in counts.iter().enumerate(){words[17+4*i]=n as u32;}
   for tile in [128usize,8,16,32] {
    let per=|n:usize|match tile {128=>n/128,8 if (1..=8).contains(&(n%128))=>1,16 if (9..32).contains(&(n%128))=>(n%128).div_ceil(16),32 if n%128>=32=>(n%128).div_ceil(32),_=>0};
    words[13]=0;assert_eq!(part_rows(&words,tile) as usize,counts.iter().copied().map(per).max().unwrap());
    words[13]=1;let actual=part_rows(&words,tile) as usize;let expected=counts.iter().copied().map(per).sum::<usize>();assert_eq!(actual,expected);
    let mut visits=Vec::new();
    for y in 0..actual {let mut remaining=y;let mut found=None;for(slot,&n)in counts.iter().enumerate(){let n=per(n);if remaining<n {found=Some((slot,remaining));break;}remaining-=n;}visits.push(found.unwrap());}
    let oracle=counts.iter().enumerate().flat_map(|(s,&n)|(0..per(n)).map(move|i|(s,i))).collect::<Vec<_>>();assert_eq!(visits,oracle);
   }
  }
 }
}

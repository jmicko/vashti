    // Exact original eight-lane 4/2/1 tree, including every RHS component.
    // Collective calls stay outside the per-lane addition predicate.
    for(var stride=4u;stride>0u;stride/=2u) {
        let other_gates=subgroupShuffleDown(gates,stride);
        let other_ups=subgroupShuffleDown(ups,stride);
        if lane<stride {gates+=other_gates;ups+=other_ups;}
    }

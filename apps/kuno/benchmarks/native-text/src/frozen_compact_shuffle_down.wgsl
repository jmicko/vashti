    // Exact original eight-lane 4/2/1 tree; all four RHS remain independent.
    for(var stride=4u;stride>0u;stride/=2u) {
        let other_values=subgroupShuffleDown(values,stride);
        if lane<stride {values+=other_values;}
    }

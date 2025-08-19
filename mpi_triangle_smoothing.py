def triangle_smoothing(data: np.ndarray, rect: list, nrep=1):
    """
    Multi-dimensional triangle smoothing - like repeatedly folding a piece of paper
    to blend pencil marks (data points) into smooth gradients. Each axis direction is smoothed separately.
    
    Parameters:
        data : Input array (1D line, 2D image, or 3D volume)
        rect : List of smoothing "strengths" for each dimension
               [radius_axis0, radius_axis1, ...]
        nrep : How many times to repeat the smoothing process
    Returns:
        Smoothed array with same shape as input
    """
    ndim = data.ndim  # How many dimensions our data has (1D, 2D, 3D)
    
    # Start with a clean copy (don't mess with original data)
    result = data.copy()

    # Repeat smoothing multiple times (like multiple coats of paint)
    for _ in range(nrep):
        # Process each dimension one-by-one (like smoothing first left-right, then up-down)
        for axis in range(ndim):
            # Skip if radius too small (no smoothing needed)
            if rect[axis] <= 1:
                continue
            # Apply 1D smoothing along current axis direction
            result = _smooth_axis(result, axis, rect[axis])
    return result

def _smooth_axis(data, axis, nbox):
    """
    Apply triangle smoothing along one dimension - think of dragging a triangular
    stencil along a row of numbers, averaging values through the stencil window.
    
    Visualize: A triangular stencil with peak at center, tapering to zero at edges.
    nbox = width of stencil base (radius from center to edge)
    """
    nx = data.shape[axis]     # Length of current dimension
    npad = 2 * nbox           # Padding needed for edge handling
    np_tot = nx + npad        # Total working length (data + padding)
    
    # Prepare output canvas (use float32 for precision during calculations)
    smoothed = np.zeros_like(data, dtype=np.float32)

    # Create iterator: Processes all lines perpendicular to current axis
    # Example: For a 2D image and horizontal axis, processes each row separately
    it = np.nditer(
        np.zeros(data.shape[:axis] + data.shape[axis+1:]),
        flags=['multi_index']
    )
    
    # Process each 1D line (like processing each row in an image)
    while not it.finished:
        # Build index to extract current 1D line
        idx = list(it.multi_index)
        idx.insert(axis, slice(None))
        # Grab the 1D data line (convert to float for precision)
        x = data[tuple(idx)].astype(np.float32)

        # Temporary workspace (like a scratchpad for calculations)
        tmp = np.zeros(np_tot, dtype=np.float32)
        
        # ===== STEP 1: Apply triangle weights =====
        # Imagine placing three copies of your data line with weights:
        # 1. Original position: Negative weights (like cutting a hole)
        # 2. Shifted right by nbox: Positive weights ×2 (filling the hole)
        # 3. Shifted right by 2*nbox: Negative weights (trimming excess)
        wt = 1.0 / (nbox * nbox)  # Normalization factor (area of triangle)
        
        # Apply weights at different positions:
        tmp[0:nx] += -wt * x          # Left: negative stamp
        tmp[nbox:nbox+nx] += 2 * wt * x  # Center: positive stamp (2× stronger)
        tmp[2*nbox:2*nbox+nx] += -wt * x  # Right: negative stamp

        # ===== STEP 2: Double integration =====
        # Think of this as two smoothing passes:
        # 1st pass: Forward accumulation (like water flowing downhill)
        t = 0.0
        for i in range(np_tot):
            t += tmp[i]          # Cumulative sum (water collecting)
            tmp[i] = t           # Store current water level
            
        # 2nd pass: Backward accumulation (water flowing uphill)
        t = 0.0
        for i in reversed(range(np_tot)):
            t += tmp[i]          # Cumulative sum in reverse
            tmp[i] = t           # Final accumulation level

        # ===== STEP 3: Edge folding =====
        # Handle edges by "mirroring" data (like folding paper at edges)
        y = np.copy(x)  # Start with original values
        
        # Copy center region (away from edges)
        y[:] = tmp[nbox:nbox+nx]  # Middle part needs no adjustment

        # Handle right edges (beyond original data length)
        j = nbox + nx  # Start beyond data end
        while j < np_tot:
            # Fold right: Mirror data back into visible area
            for i in range(min(nx, np_tot - j)):
                # Mirror position: Counting backwards from end
                y[nx - 1 - i] += tmp[j + i]  # Add folded value
            j += nx  # Jump to next reflection segment
            
            # After first fold, add direct reflection
            if j < np_tot:
                for i in range(min(nx, np_tot - j)):
                    y[i] += tmp[j + i]  # Add direct reflection
                j += nx

        # Handle left edges (before data start)
        j = nbox  # Start before data begins
        while j >= 0:
            # Fold left: Mirror data into visible area
            for i in range(min(nx, j)):
                y[i] += tmp[j - 1 - i]  # Add folded value
            j -= nx  # Jump to previous reflection segment
            
            # After first fold, add reverse reflection
            if j >= 0:
                for i in range(min(nx, j)):
                    y[nx - 1 - i] += tmp[j - 1 - i]  # Add reverse reflection
                j -= nx

        # Save smoothed line to output
        smoothed[tuple(idx)] = y
        it.iternext()  # Move to next line

    return smoothed

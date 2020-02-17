def get_bhs(bodies):
    """
    Finds the IMBH and BH particles.
    """
    imbh = None
    i_imbh = None
    bh = None
    i_bh = None
    for i, b in enumerate(bodies):
        if imbh is None and b.name == "IMBH":
            imbh = b
            i_imbh = i
        elif bh is None and b.name == "BH":
            bh = b
            i_bh = i

    if imbh is None or bh is None:
        raise ValueError("IMBH and/or BH not found!")

    return (imbh, bh), (i_imbh, i_bh)

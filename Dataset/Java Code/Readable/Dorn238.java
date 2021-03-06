     */
    public void init(CipherParameters params)
        throws IllegalArgumentException;

    /**
     * Return the name of the algorithm the MAC implements.
     *
     * @return the name of the algorithm the MAC implements.
     */
    public String getAlgorithmName();

    /**
     * Return the block size for this cipher (in bytes).
     *
     * @return the block size for this cipher in bytes.
     */
	public int getMacSize();

    /**
     * add a single byte to the mac for processing.
     *
     * @param in the byte to be processed.
     * @exception IllegalStateException if the MAC is not initialised.
     */
	public void update(byte in)
        throws IllegalStateException;

    /**
     * @param in the array containing the input.
     * @param inOff the index in the array the data begins at.
     * @param len the length of the input starting at inOff.
     * @exception IllegalStateException if the MAC is not initialised.
     * @exception DataLengthException if there isn't enough data in in.
     */
	public void update(byte[] in, int inOff, int len)
        throws DataLengthException, IllegalStateException;

    /**
     * Compute the final statge of the MAC writing the output to the out
     * parameter.
     * <p>
     * doFinal leaves the MAC in the same state it was after the last init.
     *
     * @param out the array the MAC is to be output to.
     * @param outOff the offset into the out buffer the output is to start at.
     * @exception DataLengthException if there isn't enough space in out.
     * @exception IllegalStateException if the MAC is not initialised.
     */
	public int doFinal(byte[] out, int outOff)
        throws DataLengthException, IllegalStateException;

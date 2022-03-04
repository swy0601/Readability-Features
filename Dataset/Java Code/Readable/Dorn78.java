	public void setCompanyId(long companyId);

	/**
	 * Returns the user ID of this meetups registration.
	 *
	 * @return the user ID of this meetups registration
	 */
	public long getUserId();

	/**
	 * Sets the user ID of this meetups registration.
	 *
	 * @param userId the user ID of this meetups registration
	 */
	public void setUserId(long userId);

	/**
	 * Returns the user uuid of this meetups registration.
	 *
	 * @return the user uuid of this meetups registration
	 * @throws SystemException if a system exception occurred
	 */
	public String getUserUuid() throws SystemException;

	/**
	 * Sets the user uuid of this meetups registration.
	 *
	 * @param userUuid the user uuid of this meetups registration
	 */
	public void setUserUuid(String userUuid);

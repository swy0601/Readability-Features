		assertEquals(existingEntryId, newEntryId);
	}

	public void testDynamicQueryByProjectionMissing() throws Exception {
		DynamicQuery dynamicQuery = DynamicQueryFactoryUtil.forClass(RatingsEntry.class,
				RatingsEntry.class.getClassLoader());

		dynamicQuery.setProjection(ProjectionFactoryUtil.property("entryId"));

		dynamicQuery.add(RestrictionsFactoryUtil.in("entryId",
				new Object[] { nextLong() }));

		List<Object> result = _persistence.findWithDynamicQuery(dynamicQuery);

		assertEquals(0, result.size());
	}

	public void testResetOriginalValues() throws Exception {
		if (!PropsValues.HIBERNATE_CACHE_USE_SECOND_LEVEL_CACHE) {
			return;
		}

		RatingsEntry newRatingsEntry = addRatingsEntry();

		_persistence.clearCache();

		RatingsEntryModelImpl existingRatingsEntryModelImpl = (RatingsEntryModelImpl)_persistence.findByPrimaryKey(newRatingsEntry.getPrimaryKey());

		assertEquals(existingRatingsEntryModelImpl.getUserId(),
			existingRatingsEntryModelImpl.getOriginalUserId());
		assertEquals(existingRatingsEntryModelImpl.getClassNameId(),
			existingRatingsEntryModelImpl.getOriginalClassNameId());
		assertEquals(existingRatingsEntryModelImpl.getClassPK(),
			existingRatingsEntryModelImpl.getOriginalClassPK());
	}

	protected RatingsEntry addRatingsEntry() throws Exception {
		long pk = nextLong();

		RatingsEntry ratingsEntry = _persistence.create(pk);

		ratingsEntry.setCompanyId(nextLong());

		ratingsEntry.setUserId(nextLong());

		ratingsEntry.setUserName(randomString());

		ratingsEntry.setCreateDate(nextDate());

		ratingsEntry.setModifiedDate(nextDate());

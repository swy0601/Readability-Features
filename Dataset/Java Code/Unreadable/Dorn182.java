		com.liferay.portal.service.ServiceContext serviceContext)
		throws com.liferay.portal.kernel.exception.PortalException,
			com.liferay.portal.kernel.exception.SystemException {
		try {
			MethodKey methodKey = new MethodKey(LayoutSetPrototypeServiceUtil.class.getName(),
					"updateLayoutSetPrototype",
					_updateLayoutSetPrototypeParameterTypes4);

			MethodHandler methodHandler = new MethodHandler(methodKey,
					layoutSetPrototypeId, nameMap, description, active,
					layoutsUpdateable, serviceContext);

			Object returnObj = null;

			try {
				returnObj = TunnelUtil.invoke(httpPrincipal, methodHandler);
			}
			catch (Exception e) {
				if (e instanceof com.liferay.portal.kernel.exception.PortalException) {
					throw (com.liferay.portal.kernel.exception.PortalException)e;
				}

				if (e instanceof com.liferay.portal.kernel.exception.SystemException) {
					throw (com.liferay.portal.kernel.exception.SystemException)e;
				}

				throw new com.liferay.portal.kernel.exception.SystemException(e);
			}

			return (com.liferay.portal.model.LayoutSetPrototype)returnObj;
		}
		catch (com.liferay.portal.kernel.exception.SystemException se) {
			_log.error(se, se);

			throw se;
		}
	}

	public static com.liferay.portal.model.LayoutSetPrototype updateLayoutSetPrototype(
		HttpPrincipal httpPrincipal, long layoutSetPrototypeId,
		java.lang.String settings)
		throws com.liferay.portal.kernel.exception.PortalException,
			com.liferay.portal.kernel.exception.SystemException {
		try {
			MethodKey methodKey = new MethodKey(LayoutSetPrototypeServiceUtil.class.getName(),
					"updateLayoutSetPrototype",
					_updateLayoutSetPrototypeParameterTypes5);

			MethodHandler methodHandler = new MethodHandler(methodKey,
					layoutSetPrototypeId, settings);

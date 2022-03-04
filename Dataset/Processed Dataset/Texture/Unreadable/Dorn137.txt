			layoutRevision.setCreateDate(serviceContext.getCreateDate(now));
			layoutRevision.setModifiedDate(serviceContext.getModifiedDate(now));
			layoutRevision.setLayoutSetBranchId(
				oldLayoutRevision.getLayoutSetBranchId());
			layoutRevision.setParentLayoutRevisionId(
				oldLayoutRevision.getLayoutRevisionId());
			layoutRevision.setHead(false);
			layoutRevision.setLayoutBranchId(layoutBranchId);
			layoutRevision.setPlid(oldLayoutRevision.getPlid());
			layoutRevision.setPrivateLayout(
				oldLayoutRevision.isPrivateLayout());
			layoutRevision.setName(name);
			layoutRevision.setTitle(title);
			layoutRevision.setDescription(description);
			layoutRevision.setKeywords(keywords);
			layoutRevision.setRobots(robots);
			layoutRevision.setTypeSettings(typeSettings);

			if (iconImage) {
				layoutRevision.setIconImage(iconImage);
				layoutRevision.setIconImageId(iconImageId);
			}

			layoutRevision.setThemeId(themeId);
			layoutRevision.setColorSchemeId(colorSchemeId);
			layoutRevision.setWapThemeId(wapThemeId);
			layoutRevision.setWapColorSchemeId(wapColorSchemeId);
			layoutRevision.setCss(css);
			layoutRevision.setStatus(WorkflowConstants.STATUS_DRAFT);
			layoutRevision.setStatusDate(serviceContext.getModifiedDate(now));

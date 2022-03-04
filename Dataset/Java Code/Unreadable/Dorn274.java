			selenium.getValue("//input[@value='New Message']"));
		selenium.clickAt("//input[@value='New Message']",
			RuntimeVariables.replace("New Message"));

		for (int second = 0;; second++) {
			if (second >= 90) {
				fail("timeout");
			}

			try {
				if (selenium.isVisible("//span[2]/span/button")) {
					break;
				}
			}
			catch (Exception e) {
			}

			Thread.sleep(1000);
		}

		selenium.clickAt("//span[2]/span/button",
			RuntimeVariables.replace("Dropdown"));
		assertEquals(RuntimeVariables.replace(
				"socialofficefriendfn socialofficefriendmn socialofficefriendln"),
			selenium.getText("//div[8]/div/div/ul/li[1]"));
		selenium.clickAt("//div[8]/div/div/ul/li[1]",
			RuntimeVariables.replace(
				"socialofficefriendfn socialofficefriendmn socialofficefriendln"));
		assertEquals("socialofficefriendfn socialofficefriendmn socialofficefriendln <socialofficefriendsn>,",
			selenium.getValue("//span/input"));
		assertTrue(selenium.isVisible("//span[1]/span/span/input"));
		selenium.type("//span[1]/span/span/input",
			RuntimeVariables.replace("Message3 Subject"));
		assertTrue(selenium.isVisible("//textarea"));
		selenium.type("//textarea", RuntimeVariables.replace("Message3 Body"));
		selenium.clickAt("//input[@value='Send']",
			RuntimeVariables.replace("Send"));

		for (int second = 0;; second++) {
			if (second >= 90) {
				fail("timeout");
			}

			try {
				if (selenium.isVisible("//div[@class='portlet-msg-success']")) {
					break;
				}
			}
			catch (Exception e) {
			}

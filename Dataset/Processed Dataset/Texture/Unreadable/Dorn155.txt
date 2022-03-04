
				boolean suborganization2Present = selenium.isElementPresent(
						"//td[4]/span/ul/li/strong/a");

				if (!suborganization2Present) {
					label = 3;

					continue;
				}

				selenium.clickAt("//input[@name='_125_allRowIds']",
					RuntimeVariables.replace("All Rows"));
				selenium.click(RuntimeVariables.replace(
						"//input[@value='Delete']"));
				selenium.waitForPageToLoad("30000");
				loadRequiredJavaScriptModules();
				assertTrue(selenium.getConfirmation()
								   .matches("^Are you sure you want to delete this[\\s\\S]$"));

			case 3:
				selenium.clickAt("link=Users and Organizations",
					RuntimeVariables.replace("Users and Organizations"));
				selenium.waitForPageToLoad("30000");
				loadRequiredJavaScriptModules();
				selenium.type("//input[@id='_125_keywords']",
					RuntimeVariables.replace("Selenium"));
				selenium.click(RuntimeVariables.replace(
						"//input[@value='Search']"));
				selenium.waitForPageToLoad("30000");
				loadRequiredJavaScriptModules();
				selenium.clickAt("//input[@name='_125_rowIds']",
					RuntimeVariables.replace("Row"));
				selenium.click(RuntimeVariables.replace(
						"//input[@value='Delete']"));
				selenium.waitForPageToLoad("30000");
				loadRequiredJavaScriptModules();
				assertTrue(selenium.getConfirmation()
								   .matches("^Are you sure you want to delete this[\\s\\S]$"));
				selenium.type("//input[@id='_125_keywords']",
					RuntimeVariables.replace("Test"));
				selenium.click(RuntimeVariables.replace(
						"//input[@value='Search']"));
				selenium.waitForPageToLoad("30000");
				loadRequiredJavaScriptModules();

				boolean organization3Present = selenium.isElementPresent(
						"//td[4]/span/ul/li/strong/a");

				if (!organization3Present) {
					label = 9;

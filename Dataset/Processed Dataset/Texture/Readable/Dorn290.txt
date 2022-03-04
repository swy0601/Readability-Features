/**
 * @author Brian Wing Shun Chan
 */
public class ViewSitePublicPageDropDownTests extends BaseTestSuite {
	public static Test suite() {
		TestSuite testSuite = new TestSuite();
		testSuite.addTestSuite(AddSitesTest.class);
		testSuite.addTestSuite(AddSitesPublicPageTest.class);
		testSuite.addTestSuite(ViewSitesPublicPageDropDownTest.class);
		testSuite.addTestSuite(TearDownSitesTest.class);

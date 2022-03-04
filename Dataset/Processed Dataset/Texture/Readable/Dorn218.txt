/**
 * @author Brian Wing Shun Chan
 */
public class DBUpgradeTags528TestSuite extends BaseTestSuite {

	public static Test suite() {
		TestSuite testSuite = new TestSuite();

		testSuite.addTest(LoginTests.suite());
		testSuite.addTest(TagsTestPlan.suite());

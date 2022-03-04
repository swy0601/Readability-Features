/**
 * @author Brian Wing Shun Chan
 */
public class UnitConverterTestPlan extends BaseTestSuite {

	public static Test suite() {
		TestSuite testSuite = new TestSuite();

		testSuite.addTest(PortletTestPlan.suite());
		testSuite.addTest(UnitTestPlan.suite());

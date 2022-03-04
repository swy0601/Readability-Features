		createGroup(model, "websites", parameter_group);

		
		return model;
	}
	
	private static BasicPluginConfigModel createInstallationConfigGroup(final JythonPlugin plugin, final JythonPluginCore core, final JythonPluginInitialiser jpi, final BasicPluginConfigModel parent, final boolean init_ok) {
		BasicPluginConfigModel model = core.plugin_interface.getUIManager().createBasicPluginConfigModel(parent.getSection(), "azjython.install");
		addJythonStatusParameter(init_ok, core, model, false);
		
		// Things to disable upon installing Jython.
		final List disable_on_install = new ArrayList();

		// We reuse this multiple times to store parameters in a group.
		ArrayList parameter_group = new ArrayList();
		
		LabelParameter lp = model.addLabelParameter2("azjython.config.auto_config.info");
		final ActionParameter start_param = model.addActionParameter2("azjython.blank", "azjython.config.auto_config.start");
		final ActionParameter stop_param = model.addActionParameter2("azjython.blank", "azjython.config.auto_config.stop");
		start_param.setEnabled(!init_ok);
		stop_param.setEnabled(false);
				
		parameter_group.add(lp);
		parameter_group.add(start_param);
		parameter_group.add(stop_param);
		disable_on_install.add(start_param);
		disable_on_install.add(stop_param);
		createGroup(model, "auto_config", parameter_group);
				
		final DirectoryParameter dm = model.addDirectoryParameter2("jython.path", "azjython.config.jythonpath", "");
		disable_on_install.add(dm);
		parameter_group.add(dm);
					
		ActionParameter am = model.addActionParameter2("azjython.config.install", "azjython.config.install.action");
		disable_on_install.add(am);
		parameter_group.add(am);
		
		final ParameterListener auto_install_listener = new ParameterListener() {
			public void parameterChanged(Parameter p) {
				boolean installed = jpi.installJython(true);
				if (installed) {
					for (int i=0; i < disable_on_install.size(); i++) {
						((Parameter)disable_on_install.get(i)).setEnabled(false);
					}
				}
			}
		};
		
		am.addListener(auto_install_listener);
		am.setEnabled(!init_ok);

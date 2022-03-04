
package org.gudy.azureus2.pluginsimpl.local.update;

/**
 * @author parg
 *
 */

import java.io.*;

import org.gudy.azureus2.platform.PlatformManager;
import org.gudy.azureus2.platform.PlatformManagerCapabilities;
import org.gudy.azureus2.platform.PlatformManagerFactory;
import org.gudy.azureus2.plugins.update.*;

import org.gudy.azureus2.core3.util.*;
import org.gudy.azureus2.core3.internat.MessageText;
import org.gudy.azureus2.core3.logging.*;

import com.aelitis.azureus.core.update.AzureusRestarter;
import com.aelitis.azureus.core.update.AzureusRestarterFactory;

public class 
UpdateInstallerImpl
	implements UpdateInstaller
{
		// change these and you'll need to change the Updater!!!!
	
	protected static final String	UPDATE_DIR 	= "updates";
	protected static final String	ACTIONS		= "install.act";
	
	protected static AEMonitor	class_mon 	= new AEMonitor( "UpdateInstaller:class" );

	private UpdateManagerImpl	manager;
	private File				install_dir;
	
	protected static void
	checkForFailedInstalls(
		UpdateManagerImpl	manager )
	{
		try{
			File	update_dir = new File( manager.getUserDir() + File.separator + UPDATE_DIR );
			
			File[]	dirs = update_dir.listFiles();
			
			if ( dirs != null ){
				
				boolean	found_failure = false;
				
				String	files = "";

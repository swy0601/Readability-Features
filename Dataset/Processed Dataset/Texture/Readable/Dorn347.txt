package org.gudy.azureus2.countrylocator.statisticsProviders;

import java.util.Map;

import org.gudy.azureus2.plugins.peers.Peer;

/**
 * IStatisticsProvider (see {@link #getStatistics(Peer[])})
 * 
 * @author gooogelybear
 * @author free_lancer
 */
public interface IStatisticsProvider {
		
	/**
	 * Calculates stats values for a column in the stats table. <br />
	 * 
	 * @param peers the peers for which the stats will be calculated
	 * @return A map indexed by country code which contains the stats values as values
	 */
	public Map getStatistics(Peer[] peers);
	
	public String getName();
	
	/**
	 * 
	 * @return The measurement unit used for the stats values, e.g. "kB/s"
	 */
	public String getUnit();

			sum_activity    += g_activity[index]*exp(-sum_attenuation);
			index += pixelNumber;
			}
		g_sinogram[tid]=sum_activity;
		//g_sinogram[tid]=g_activity[tid+128*128*50];
	}
	return; 	
}



		proportionRandomValuesUsed = 0;
	}

	int connections = w.Elements();

	float * rnd = (v_reconstructed == nullptr) ? nullptr : (randomValues.Pointer() + (proportionRandomValuesUsed * randomValuesNeededPerEpoch));

	if(connections > MAX_THREADS_PER_BLOCK) {
		KernelComputeStatusHiddenUnitsRBM(dimJsamples, inputsBlockSize, v, w.DevicePointer(), b.DevicePointer(), h, rnd, I);
	} else {
		ComputeStatusHiddenUnitsSmallRBM<<<samples, dimIJ, connections * sizeof(cudafloat)>>>(v, w.DevicePointer(), b.DevicePointer(), h, rnd);
	}

	if (v_reconstructed != nullptr) {
		rnd = (useBinaryValuesVisibleReconstruction) ? (rnd + J * samples) : nullptr;

		if(connections > MAX_THREADS_PER_BLOCK) {
			KernelComputeStatusVisibleUnitsRBM(dimIsamples, hiddenUnitsBlockSize, h, w.DevicePointer(), a.DevicePointer(), v_reconstructed, rnd, J);
		} else {		
			ComputeStatusVisibleUnitsSmallRBM<<<samples, dimJI, connections * sizeof(cudafloat)>>>(h, w.DevicePointer(), a.DevicePointer(), v_reconstructed, rnd);
		}
	}
	
	proportionRandomValuesUsed++;
}

void RBM::ContrastiveDivergence(int n) {
	ComputeStatusUnits(v.Pointer(), h_data.Pointer(), v_recon.Pointer());
	for (int k = 1; k < n; k++) ComputeStatusUnits(v_recon.Pointer(), h_recon.Pointer(), v_recon.Pointer());
	ComputeStatusUnits(v_recon.Pointer(), h_recon.Pointer(), nullptr);

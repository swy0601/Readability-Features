	d_localGradient.ResizeWithoutPreservingData(neurons * patterns);
		
	sharedMemFire = connections * sizeof(cudafloat);
	sharedMemGradients = (nextLayerNeurons * (neurons + 1)) * sizeof(cudafloat);

	dimNeuronsPatterns.x = neurons;
	dimNeuronsPatterns.y = patterns;

	dimInputsNeurons.x = inputs;
	dimInputsNeurons.y = neurons;

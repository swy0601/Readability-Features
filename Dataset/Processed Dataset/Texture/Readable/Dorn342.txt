curandRngType_t Random::randomGeneratorType = CURAND_RNG_PSEUDO_DEFAULT;

curandGenerator_t Random::RandomGenerator() {
	if (randomGenerator == nullptr) {
		curandCreateGenerator(&randomGenerator, randomGeneratorType);
		atexit(&CleanUp);
	}

	return randomGenerator;
}

void Random::CleanUp() {
	if (randomGenerator != nullptr) {
		curandDestroyGenerator(randomGenerator);
		randomGenerator = nullptr;
	}
}

void Random::SetSeed(unsigned long long seed, curandRngType_t generatorType) {
	if (generatorType != randomGeneratorType) {
		randomGeneratorType = generatorType;
		CleanUp();		
	}

	curandSetPseudoRandomGeneratorSeed(RandomGenerator(), seed);
}

void Random::Fill(DeviceArray<float> & a) {
	curandGenerateUniform(RandomGenerator(), a.Pointer(), a.Length());
}

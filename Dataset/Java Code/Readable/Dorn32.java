		float trialFitness = posFitnesses[solutionPosID];

		if(trialFitness BETTER_THAN fitnesses[solutionPosID])
		{
			posBetter = true;
			bias = 0.2f*bias + 0.4f*(dif+bias);
		}
		else
		{
			trialFitness = negFitnesses[solutionPosID];
			if(trialFitness BETTER_THAN fitnesses[solutionPosID])
			{
				negBetter = true;
				bias = bias - 0.4f*(dif+bias);
			}
		}

		if(posBetter || negBetter)
		{
			successes[solutionPosID]++;
			fails[solutionPosID] = 0;
			biases[solutionPosID] = bias;
			fitnesses[solutionPosID] = trialFitness;
			
		}
		else
		{
			successes[solutionPosID] = 0;
			fails[solutionPosID]++;
		}

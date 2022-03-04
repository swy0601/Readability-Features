
		// No computation is performed if any of the point is part of the background
        // The two is added because the image is resample between 2 and bin +2
        // if 64 bins are used the histogram will have 68 bins et the image will be between 2 and 65
        if( targetImageValue>0.0f &&
            resultImageValue>0.0f &&
            targetImageValue<c_Binning &&
            resultImageValue<c_Binning &&
            targetImageValue==targetImageValue &&
            resultImageValue==resultImageValue){

            targetImageValue = floor(targetImageValue); // Parzen window filling of the joint histogram is approximated
            resultImageValue = floor(resultImageValue);

			float3 resDeriv = make_float3(
				resultImageGradient.x,
				resultImageGradient.y,
				resultImageGradient.z);
				
			if( resultImageGradient.x==resultImageGradient.x &&
				resultImageGradient.y==resultImageGradient.y &&
				resultImageGradient.z==resultImageGradient.z){
					
				float jointEntropyDerivative_X = 0.0f;
				float movingEntropyDerivative_X = 0.0f;
				float fixedEntropyDerivative_X = 0.0f;
						
				float jointEntropyDerivative_Y = 0.0f;
				float movingEntropyDerivative_Y = 0.0f;
				float fixedEntropyDerivative_Y = 0.0f;
						
				float jointEntropyDerivative_Z = 0.0f;
				float movingEntropyDerivative_Z = 0.0f;
				float fixedEntropyDerivative_Z = 0.0f;
						
				for(int t=(int)(targetImageValue-1.0f); t<(int)(targetImageValue+2.0f); t++){
					if(-1<t && t<c_Binning){
						for(int r=(int)(resultImageValue-1.0f); r<(int)(resultImageValue+2.0f); r++){
							if(-1<r && r<c_Binning){
								float commonValue = GetBasisSplineValue((float)t-targetImageValue) *
									GetBasisSplineDerivativeValue((float)r-resultImageValue);
		
								float jointLog = tex1Dfetch(histogramTexture, t*c_Binning+r);
								float targetLog = tex1Dfetch(histogramTexture, c_Binning*c_Binning+t);
								float resultLog = tex1Dfetch(histogramTexture, c_Binning*c_Binning+c_Binning+r);
		
								float temp = commonValue * resDeriv.x;
								jointEntropyDerivative_X -= temp * jointLog;
								fixedEntropyDerivative_X -= temp * targetLog;
								movingEntropyDerivative_X -= temp * resultLog;

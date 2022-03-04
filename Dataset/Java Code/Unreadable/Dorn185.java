				continue;
				
			}else if ( c == '\\' ){
				
				escape = true;
				
				continue;
			}
			
			if ( c == '"' || c == '\'' && ( i == 0 || chars[ i-1 ] != '\\' )){
				
				if ( quote == ' ' ){
						
					bit_contains_quotes = true;
					
					quote = c;
					
				}else if ( quote == c ){
										
					quote = ' ';
					
				}else{
					
					bit += c;
				}
			}else{
				
				if ( quote == ' ' ){
					
					if ( c == ' ' ){

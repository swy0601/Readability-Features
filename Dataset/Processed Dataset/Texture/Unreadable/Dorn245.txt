            {
               continue;
            }

            SchemaLoadInfo schemaLoadInfo = new SchemaLoadInfo(addStringArrays(tableTypes, viewTypes));
            schemaLoadInfo.schemaName = _schemaDetails[i].getSchemaName();
            schemaLoadInfo.tableTypes = new String[0];

            if(SQLAliasSchemaDetailProperties.SCHEMA_LOADING_ID_DONT_LOAD !=_schemaDetails[i].getTable())
            {
               schemaLoadInfo.tableTypes = addStringArrays(schemaLoadInfo.tableTypes, tableTypes);
            }

            if(SQLAliasSchemaDetailProperties.SCHEMA_LOADING_ID_DONT_LOAD !=_schemaDetails[i].getView())
            {
               schemaLoadInfo.tableTypes = addStringArrays(schemaLoadInfo.tableTypes, viewTypes);
            }

            if(SQLAliasSchemaDetailProperties.SCHEMA_LOADING_ID_DONT_LOAD !=_schemaDetails[i].getProcedure())
            {
               schemaLoadInfo.loadProcedures = true;
            }
            else
            {
               schemaLoadInfo.loadProcedures = false;

            }

            schemaLoadInfos.add(schemaLoadInfo);
         }

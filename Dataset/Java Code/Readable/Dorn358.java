    PrimaryKeyInfo() {
        super(null, null, null, null, null);
    }
    
    /**
     * Create a new PrimaryKeyInfo object.
     * 
     * @param catalog catalog name
     * @param schema schema name
     * @param aColumnName the name of the column that either by itself or along
     *                    with others form(s) a unique index value for a single
     *                    row in a table. 
     * @param aKeySequence sequence number within primary key
     * @param aPrimaryKeyName the name of the primary key
     * @param md
     */
	public PrimaryKeyInfo(String catalog, 
                   String schema,
                   String aTableName,
                   String aColumnName, 
                   short aKeySequence, 
                   String aPrimaryKeyName,
                   ISQLDatabaseMetaData md)
	{
		super(catalog, schema, aPrimaryKeyName, DatabaseObjectType.PRIMARY_KEY, md);
        columnName = aColumnName;
        tableName = aTableName;
        keySequence = aKeySequence;
	}

    /**
     * @param columnName The columnName to set.
     */
    public void setColumnName(String columnName) {
        this.columnName = columnName;
    }

    /**
     * @return Returns the columnName.
     */
    public String getColumnName() {
        return columnName;
    }

    /**
     * @param keySequence The keySequence to set.
     */
    public void setKeySequence(short keySequence) {
        this.keySequence = keySequence;
    }

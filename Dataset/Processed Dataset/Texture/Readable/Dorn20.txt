		return listenerID;
	}

	public String getMessageID() {
		return messageID;
	}

	public String getOperationID() {
		return operationID;
	}

	protected long getSequenceNo() {
		return lSequenceNo;
	}

	protected void setSequenceNo(long sequenceNo) {
		lSequenceNo = sequenceNo;
	}

	public String toString() {
		String paramString = parameters.toString();
		return "PlaformMessage {"
				+ "cn"
				+ contentNetworkID
				+ ", "
				+ lSequenceNo
				+ ", "
				+ messageID
				+ ", "
				+ listenerID
				+ ", "
				+ operationID
				+ ", "
				+ (paramString.length() > 32767 ? paramString.substring(0, 32767)
						: paramString) + "}";
	}

	public String toShortString() {
		return (requiresAuthorization ? "AUTH: " : "") + getMessageID() + "."
				+ getListenerID() + "." + getOperationID();
	}

	/**
	 * @return
	 *
	 * @since 3.1.1.1
	 */
	public boolean sendAZID() {
		return sendAZID;
	}

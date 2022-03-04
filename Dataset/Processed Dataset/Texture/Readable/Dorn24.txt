		else if (RUBY.getValue().equals(value)) {
			return RUBY;
		}

		throw new IllegalArgumentException("Invalid value " + value);
	}

	public String getValue() {
		return _value;
	}

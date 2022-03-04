    }

    abstract void encode(DEROutputStream out)
        throws IOException;

    public String toString()
    {
      return "#"+new String(Hex.encode(string));
    }

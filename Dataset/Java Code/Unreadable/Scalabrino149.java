	/**
	 * As per sections 12.2.3.23.9, 12.2.4.8.9 and 12.2.5.3.6 of the JPA 2.0
	 * specification, the element-collection subelement completely overrides the
	 * mapping for the specified field or property.  Thus, any methods which
	 * might in some contexts merge with annotations must not do so in this
	 * context.
	 */
	private void getElementCollection(List<Annotation> annotationList, XMLContext.Default defaults) {
		for ( Element element : elementsForProperty ) {
			if ( "element-collection".equals( element.getName() ) ) {
				AnnotationDescriptor ad = new AnnotationDescriptor( ElementCollection.class );
				addTargetClass( element, ad, "target-class", defaults );
				getFetchType( ad, element );
				getOrderBy( annotationList, element );
				getOrderColumn( annotationList, element );
				getMapKey( annotationList, element );
				getMapKeyClass( annotationList, element, defaults );
				getMapKeyTemporal( annotationList, element );
				getMapKeyEnumerated( annotationList, element );
				getMapKeyColumn( annotationList, element );
				buildMapKeyJoinColumns( annotationList, element );
				Annotation annotation = getColumn( element.element( "column" ), false, element );
				addIfNotNull( annotationList, annotation );
				getTemporal( annotationList, element );
				getEnumerated( annotationList, element );
				getLob( annotationList, element );
				//Both map-key-attribute-overrides and attribute-overrides
				//translate into AttributeOverride annotations, which need
				//need to be wrapped in the same AttributeOverrides annotation.
				List<AttributeOverride> attributes = new ArrayList<AttributeOverride>();
				attributes.addAll( buildAttributeOverrides( element, "map-key-attribute-override" ) );
				attributes.addAll( buildAttributeOverrides( element, "attribute-override" ) );
				annotation = mergeAttributeOverrides( defaults, attributes, false );
				addIfNotNull( annotationList, annotation );
				annotation = getAssociationOverrides( element, defaults, false );
				addIfNotNull( annotationList, annotation );
				getCollectionTable( annotationList, element, defaults );
				annotationList.add( AnnotationFactory.create( ad ) );
				getAccessType( annotationList, element );
			}
		}
	}
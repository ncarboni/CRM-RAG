# Entity Documents

This directory contains individual markdown files for each entity processed from the RDF data.

## Purpose
- **Transparency**: View exactly what the system extracts and processes for each entity
- **Debugging**: Identify issues with relationship extraction or label processing
- **Reuse**: These documents can be reused for other purposes or analyses

## File Naming Convention
Files are named: `{label}_{hash}.md`
- `label`: Cleaned entity label (special chars removed, spaces replaced with underscores)
- `hash`: 8-character MD5 hash of the entity URI (ensures uniqueness)

## File Structure
Each file contains:
1. **Metadata header**: URI, label, generation timestamp
2. **Types**: RDF types of the entity
3. **Properties**: All literal values (labels, descriptions, WKT geometries, dates, etc.)
4. **Relationships**: Filtered CIDOC-CRM relationships in natural language

## Notes
- Schema-level predicates (rdf:type, rdfs:subClassOf, etc.) are filtered from relationships
- Self-referential relationships are removed
- Files are regenerated on each rebuild
